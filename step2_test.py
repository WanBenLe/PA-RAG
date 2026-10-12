# -*- coding: utf-8 -*-
"""
Decoupled Multi-Hop Financial RAG Evaluation Pipeline
模块功能:
    1. 基于监管报告 (10-K/10-Q) 的金融领域多跳 RAG 评测与消融实验。
    2. 支持多线程并发流水线评估、意图分解、语义推导树展开及多维审计评测。
"""

import os
import re
import json
import random
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures

import numpy as np
import openai
from tqdm.auto import tqdm
from pymilvus import MilvusClient
from datasets import load_dataset

# 假设这些来自内部模块 rag_util；单测时可通过 mock 模拟
from rag_util import (
    call_jina_embedding_batch_api,
    call_jina_embedding_api,
    call_vllm_rerank_jina_api,
    call_sglang_generation,
    realtime_eval_and_fix,
    year_filter,
    parse_hit_nodes,
    bridge_hit_chunks,
    fetch_and_normalized_serag_filter_chunks,
    run_accuracy_judge,
    run_recall_judge
)

# ==============================================================================
# 🛠️ 1. 全局配置与环境管理 (消除所有硬编码)
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FinancialRAG")

@dataclass
class RAGPipelineConfig:
    """系统运行环境与超参数配置中心"""
    mode: str = "linear"                           # 运行模式: 'pure', 'page_index', 'linear'
    env: str = "sharedStore"                       # 隔离环境: 'singleStore', 'sharedStore', 'inContext', 'inContext_reverse'
    top_k: int = 3                                 # 单次召回/重排截断值
    k_hop: int = 3                                 # 多跳检索深度
    window_size: int = 10                          # 物理上下文窗口前后扩展块数
    embedding_dim: int = 512                       # 向量维度
    max_workers: int = 16                          # 线程池最大并发数
    sample_fraction: Optional[float] = None        # 数据集采样率 (None 表示全量评测)
    random_seed: int = 42                          # 随机种子
    
    # 服务网络终端与模型
    vllm_api_base: str = field(default_factory=lambda: os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    intent_model_name: str = field(default_factory=lambda: os.getenv("INTENT_MODEL_NAME", "Qwen3.6-35B-A3B-AWQ"))
    
    # 路径与数据集配置 (支持环境变量覆盖)
    base_workspace_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_WORKSPACE_DIR", "/root/autodl-tmp/nanobot/workspace")))
    pdf_cache_dir: Path = field(default_factory=lambda: Path(os.getenv("PDF_CACHE_DIR", "/root/autodl-tmp/financebench/pdfs")))
    dataset_path: str = field(default_factory=lambda: os.getenv("DATASET_PATH", "./datasets/PatronusAI___financebench"))
    
    # 派生字段 (动态初始化)
    db_path: str = ""
    collection_name: str = ""
    col_chunks: str = ""
    col_assertions: str = ""
    report_path: str = ""

    def __post_init__(self):
        """根据 mode 动态解析数据库与输出文件路径"""
        memory_dir = self.base_workspace_dir / "memory"
        eval_output_dir = self.base_workspace_dir / "pure_rag_eval"

        if self.mode == "page_index":
            self.db_path = str(memory_dir / "milvus_bayes.db")
            self.collection_name = "bayes_knowledge"
            self.report_path = str(eval_output_dir / f"page_index_rag_{self.env}_results.json")
        elif self.mode == "pure":
            self.db_path = str(memory_dir / "pure_rag.db")
            self.collection_name = "pure_rag"
            self.report_path = str(eval_output_dir / f"pure_rag_flat_{self.env}_results.json")
        elif self.mode == "linear":
            self.db_path = str(memory_dir / "linear_rag_decoupled.db")
            self.col_chunks = "linear_rag_chunks"
            self.col_assertions = "linear_rag_assertions"
            self.report_path = str(eval_output_dir / f"linear_rag_decoupled_{self.env}_results.json")
        else:
            raise ValueError(f"❌ 运行模式未识别: {self.mode}. 仅支持 ['page_index', 'pure', 'linear']")


# ==============================================================================
# 🔒 2. 线程安全的 Milvus 客户端连接池
# ==============================================================================

class MilvusClientManager:
    """进程级单例连接池，确保多线程下对 SQLite/Milvus 文件的访问安全并复用 gRPC 通道"""
    _clients: Dict[str, MilvusClient] = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, db_path: str) -> MilvusClient:
        with cls._lock:
            if db_path not in cls._clients:
                cls._clients[db_path] = MilvusClient(db_path)
            return cls._clients[db_path]


# ==============================================================================
# 📦 3. 数据集与元数据准备
# ==============================================================================

def prepare_dataset(config: RAGPipelineConfig) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str], List[str]]:
    """加载评测数据集，提取文档元数据并建立归一化映射关系"""
    dataset = load_dataset(config.dataset_path, split="train")

    # 优先从 PDF 本地缓存读取文档列表，若无则从 Milvus 全量拉取
    if config.pdf_cache_dir.exists():
        raw_active_docs = {f.name for f in config.pdf_cache_dir.glob("*.pdf")}
    else:
        client = MilvusClientManager.get_client(config.db_path)
        col_name = config.col_chunks if config.mode == "linear" else config.collection_name
        all_entities = client.query(
            collection_name=col_name,
            filter="id > 0",
            output_fields=["source_doc"],
            limit=16384
        )
        raw_active_docs = {hit["source_doc"] for hit in all_entities}

    # 解析可检索的公司实体列表与报告年份
    company_list = sorted(list({doc.split('_')[0] for doc in raw_active_docs if "_" in doc}))
    year_list = sorted(list({re.search(r'\d{4}', doc).group() for doc in raw_active_docs if re.search(r'\d{4}', doc)}))

    def _get_norm_key(name: str) -> str:
        if not name:
            return ""
        return name.rsplit('.', 1)[0].lower() if '.' in name else name.lower()

    # 建立多格式名称到标准文档名的映射字典
    normalized_mapping: Dict[str, str] = {}
    for d in raw_active_docs:
        normalized_mapping[_get_norm_key(d)] = d
        normalized_mapping[d.lower()] = d

    # 过滤出当前知识库能覆盖的有效评测样本
    eval_rows = [
        r for r in dataset
        if _get_norm_key(r.get("doc_name", "")) in normalized_mapping or r.get("doc_name", "").lower() in normalized_mapping
    ]

    # 按需进行分层采样
    if config.sample_fraction and 0 < config.sample_fraction < 1.0:
        random.seed(config.random_seed)
        sample_size = max(1, int(len(eval_rows) * config.sample_fraction))
        eval_rows = random.sample(eval_rows, sample_size)
        logger.info(f"已启用采样评测: 抽取样本数 {len(eval_rows)} / 总可用数 {len(dataset)}")

    return eval_rows, normalized_mapping, company_list, year_list


# ==============================================================================
# 🧠 4. 检索策略路由实现
# ==============================================================================

def execute_linear_retrieval(
    row: Dict[str, Any],
    actual_db_doc_name: str,
    config: RAGPipelineConfig,
    company_list: List[str],
    year_list: List[str]
) -> List[Dict[str, str]]:
    """执行 Linear RAG: 包含金融意图解析、K-Hop 语义树迭代扩展、上下文窗口物理拉取及融合"""
    question = row.get("question", "")
    doc_name = row.get("doc_name", "")
    client_llm = openai.OpenAI(api_key="EMPTY", base_url=config.vllm_api_base)
    instruction = (
        "Given a complex financial query, retrieve relevant passages or factual assertions "
        "that provide the exact numbers, metrics, or corporate context to answer the query."
    )

    # 1. 结构化金融意图解构 (Prompt 驱动 CoT)
    intent_sys = f"""You are an Expert Financial Router for a RAG system analyzing regulatory corporate filings (e.g., 10-K, 10-Q, Annual and Interim Reports).
Valid Companies: {company_list}
Valid Years: {year_list}

Deconstruct the user's financial inquiry into its core accounting components and retrieval requirements adhering strictly to standard GAAP/IFRS rules.

Output ONLY a valid JSON object without markdown fences:
{{
  "chain_of_thought": "Rigorous step-by-step deconstruction of the inquiry.",
  "mathematical_formula": "The universal accounting formula expressed in LaTeX, or 'N/A'.",
  "company": "The exact matching company name from the valid list.",
  "year": ["A list of all unique calendar/fiscal years relevant to the required data window"],
  "query_type": "'Fact' or 'Opinion'",
  "query_list": ["Concise search strings targeting specific financial schedules or notes."]
}}"""

    response_intent = client_llm.chat.completions.create(
        model=config.intent_model_name,
        messages=[{"role": "system", "content": intent_sys}, {"role": "user", "content": question}],
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )

    intent_dict = realtime_eval_and_fix(response_intent.choices[0].message.content)
    target_company = intent_dict.get("company", doc_name.split('_')[0])
    intent_is_fact = intent_dict.get("query_type", "Fact") == "Fact"

    raw_years = intent_dict.get("year", [])
    target_years = [str(raw_years)] if isinstance(raw_years, (str, int)) else [str(y) for y in raw_years] if isinstance(raw_years, list) else []
    query_list = intent_dict.get("query_list", []) if isinstance(intent_dict.get("query_list"), list) else []

    client = MilvusClientManager.get_client(config.db_path)
    retrieved_items = []

    # 2. 构建向量检索约束条件
    expr_a = f'is_fact == {str(intent_is_fact).lower()}'
    if config.env == "singleStore":
        expr_a += f' and node_id LIKE "{actual_db_doc_name}%"'
    else:
        if target_years:
            or_expr = " or ".join([f'node_id LIKE "{target_company}%{y}%"' for y in target_years])
            expr_a += f' and ({or_expr})'
        else:
            expr_a += f' and node_id LIKE "{target_company}%"'

    global_tree_nodes = []
    existing_nids = set()

    search_queries = [question] + query_list[:5]
    batch_query_vectors = call_jina_embedding_batch_api(search_queries, is_query=True, dim=config.embedding_dim)

    # Hop 1 初始检索与候选召回
    hop1_candidates = []
    if batch_query_vectors:
        hop1_res = client.search(
            collection_name=config.col_assertions,
            data=batch_query_vectors,
            filter=expr_a,
            limit=config.top_k * 3,
            output_fields=["node_id", "tree_like", "is_fact"],
            metric_type="IP"
        )
        hop1_map = {hit["entity"]["node_id"]: hit["entity"] for res in hop1_res if res for hit in res}
        hop1_candidates = list(hop1_map.values())

    composite_rerank_query = (
        f"Main Question: {question} (CRUCIAL Core Components to find: {', '.join(query_list)})"
        if query_list else question
    )

    current_frontier = []
    if hop1_candidates:
        dynamic_top_k = max(config.top_k, len(search_queries) * 2)
        rerank_docs = [f"[Assertion]: {c['tree_like']}" for c in hop1_candidates]
        indices = call_vllm_rerank_jina_api(
            query=composite_rerank_query,
            documents=rerank_docs,
            top_k=dynamic_top_k,
            instruction=instruction
        )
        current_frontier = [hop1_candidates[i] for i in indices]

        for n in current_frontier:
            n["derivation_chain"] = f"[Context Block: {n['node_id']}]: {n['tree_like']}"
            if n["node_id"] not in existing_nids:
                global_tree_nodes.append(n)
                existing_nids.add(n["node_id"])

    # Hop 2 ~ K 迭代展开推导树
    for _ in range(2, config.k_hop + 1):
        if not current_frontier:
            break

        expansion_queries = [f"{question}. Background context: {node['tree_like']}" for node in current_frontier]
        batch_vectors = call_jina_embedding_batch_api(expansion_queries, is_query=True, dim=config.embedding_dim)
        if not batch_vectors:
            break

        hop_n_res = client.search(
            collection_name=config.col_assertions,
            data=batch_vectors,
            filter=expr_a,
            limit=3,
            output_fields=["node_id", "tree_like", "is_fact"],
            metric_type="IP"
        )

        cand_pool = []
        for idx, res_batch in enumerate(hop_n_res):
            if res_batch:
                parent_node = current_frontier[idx]
                for hit in res_batch:
                    cand_node = hit["entity"]
                    cand_node["derivation_chain"] = (
                        f"[Context Block: {cand_node['node_id']}]: {cand_node['tree_like']} -> {parent_node['derivation_chain']}"
                    )
                    if cand_node["node_id"] not in existing_nids:
                        cand_pool.append(cand_node)

        if not cand_pool:
            break

        cand_docs = [f"[Assertion]: {c['tree_like']}" for c in cand_pool]
        indices = call_vllm_rerank_jina_api(
            query=composite_rerank_query,
            documents=cand_docs,
            top_k=config.top_k,
            instruction=instruction
        )

        next_frontier = []
        for idx in indices:
            best_node = cand_pool[idx]
            next_frontier.append(best_node)
            if best_node["node_id"] not in existing_nids:
                global_tree_nodes.append(best_node)
                existing_nids.add(best_node["node_id"])
        current_frontier = next_frontier

    # 3. 物理行号滑动窗口恢复与原始切片拼装
    final_context_ids = set()
    golden_asts_map = {}
    for node in global_tree_nodes:
        golden_asts_map.setdefault(node["node_id"], []).append(node)
        match = re.search(r"^(.*)_LN_(\d+)$", node["node_id"])
        if match:
            filename, idx_str = match.groups()
            idx = int(idx_str)
            for i in range(max(0, idx - config.window_size), idx + config.window_size + 1):
                final_context_ids.add(f"{filename}_LN_{i:04d}")

    if final_context_ids:
        id_list_str = ", ".join([f'"{nid}"' for nid in final_context_ids])
        final_chk_res = client.query(
            collection_name=config.col_chunks,
            filter=f"node_id in [{id_list_str}]",
            output_fields=["node_id", "text"]
        )
        final_chk_map = {r["node_id"]: r["text"] for r in final_chk_res}

        golden_fact_blocks = [f"\n=== 🌳 Dynamic K-Hop Semantic Tree (Golden Facts, Depth: {config.k_hop}) ===\n"]
        golden_nid = []

        sorted_nids = sorted(
            list(golden_asts_map.keys()),
            key=lambda x: (x.split('_LN_')[0], int(x.split('_LN_')[-1]))
        )

        for nid in sorted_nids:
            for ast in golden_asts_map[nid]:
                t_str = "Fact" if ast.get("is_fact", True) else "Opinion"
                chain_display = ast.get("derivation_chain", f"[Context Block: {nid}]: {ast['tree_like']}")
                golden_nid.append(nid)
                golden_fact_blocks.append(f"  • [Golden {t_str}]: {chain_display}")

        golden_nid_dict = parse_hit_nodes(golden_nid)
        cluster_golden_nid = {}
        for k in golden_nid_dict.keys():
            cluster_golden_nid[k], _ = bridge_hit_chunks(golden_nid_dict[k])

        chunk_text = fetch_and_normalized_serag_filter_chunks(
            cluster_golden_nid, batch_query_vectors, golden_nid, client
        )
        complete_context = f"[origin source text]\n{chunk_text}" + "".join(golden_fact_blocks)
        retrieved_items.append({"text": complete_context})

    return retrieved_items


def execute_baseline_retrieval(
    row: Dict[str, Any],
    actual_db_doc_name: str,
    config: RAGPipelineConfig
) -> List[Dict[str, str]]:
    """执行经典平面向量检索 Baseline (Pure / Page Index)"""
    question = row.get("question", "")
    retrieved_items = []
    client = MilvusClientManager.get_client(config.db_path)
    collection_name = config.collection_name or "pure_rag"

    if config.env in ["singleStore", "sharedStore"]:
        query_vector = call_jina_embedding_api(question, dim=config.embedding_dim, is_query=True)
        filter_expr = f"source_doc == '{actual_db_doc_name}'" if config.env == "singleStore" else None
        search_res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=config.top_k,
            filter=filter_expr,
            output_fields=["text"]
        )
        if search_res and search_res[0]:
            retrieved_items.extend(hit["entity"] for hit in search_res[0])
    elif config.env in ["inContext", "inContext_reverse"]:
        query_res = client.query(
            collection_name=collection_name,
            filter=f"source_doc == '{actual_db_doc_name}'",
            output_fields=["text"],
            limit=150
        )
        if query_res:
            retrieved_items = sorted(query_res, key=lambda x: x.get("id", 0))
            if config.env == "inContext_reverse":
                retrieved_items.reverse()

    return retrieved_items


# ==============================================================================
# ⚖️ 5. QA 推理与裁判审计评估
# ==============================================================================

def generate_and_evaluate(
    question: str,
    combined_context: str,
    gold_answer: str,
    evidence_text: str
) -> Dict[str, Any]:
    """生成最终财务审计答案，解析 XML 标签并进行指标打分"""
    qa_system_prompt = """
You are an expert corporate financial analyst. Answer the user's question by rigorously analyzing the provided context.

Context Handling & Noise Reduction:
The provided context may contain large amounts of irrelevant information. Your primary task is to filter out the noise. ONLY extract data, formulas, and facts that directly address the user's specific question. Do not summarize the text or engage in tangential reasoning.

Crucial Execution Rules:
1. Targeted Extraction: Identify the exact metrics required. Ignore all unrelated financial data.
2. Explicit Calculation: If a requested metric is NOT explicitly stated but underlying components ARE, you MUST perform the calculation. Show concise, strictly necessary arithmetic steps and round the final answer properly.
3. Quantitative Backing: Cite only the most critical, specific quantitative facts that prove your point. Do not over-explain.
4. No Preambles: State your logic and conclusion directly. Do not include polite phrases or meta-commentary.

OUTPUT FORMAT PROTOCOL:
You MUST wrap your final output sections strictly using the following XML tags:
<analysis_and_steps>
[Provide a highly concise deduction. List: 1. Goal, 2. Variables, 3. Formula, 4. Calculation.]
</analysis_and_steps>

<final_answer>
[State the absolute final numeric, qualitative, or factual conclusion in 1 clear sentence.]
</final_answer>
"""
    user_prompt = f"Context Reference Material:\n{combined_context}\n\nQuestion: {question}\nAnswer:"
    gen_res = call_sglang_generation(user_prompt, qa_system_prompt, thinking=True)
    raw_content = gen_res.get("content", "").strip()

    # 结构化正则提纯
    clean_answer = ""
    answer_match = re.search(r"<final_answer>(.*?)</final_answer>", raw_content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        clean_answer = answer_match.group(1).strip()
    else:
        # 降级容错提取
        lines = [line.strip() for line in raw_content.split("\n") if line.strip()]
        if lines:
            clean_answer = re.sub(r"[Output Generation]|[Done]|\*Proceeds\*", "", lines[-1], flags=re.IGNORECASE).strip()

    is_correct = run_accuracy_judge(question=question, agent_ans=clean_answer, gold_ans=gold_answer)
    recall_res = run_recall_judge(question=question, evidence=evidence_text, combined_context=combined_context)

    return {
        "clean_answer": clean_answer,
        "is_correct": is_correct,
        "evidence_matched": bool(recall_res.get("evidence_matched", False)),
        "cot_aligned": "N/A",
        "cot_diff": "N/A"
    }


# ==============================================================================
# 🔄 6. 单行任务处理器
# ==============================================================================

def process_single_row(
    row: Dict[str, Any],
    config: RAGPipelineConfig,
    normalized_mapping: Dict[str, str],
    company_list: List[str],
    year_list: List[str]
) -> Dict[str, Any]:
    """单样本端到端评测任务单元"""
    doc_name = row.get("doc_name", "")
    question = row.get("question", "")
    gold_answer = row.get("answer", "")

    norm_key = (doc_name.rsplit('.', 1)[0].lower() if '.' in doc_name else doc_name.lower()) if doc_name else ""
    actual_db_doc_name = normalized_mapping.get(norm_key, normalized_mapping.get(doc_name.lower(), doc_name))

    # 提取基准证据文本
    evidence_text = ""
    if row.get("evidence_text"):
        evidence_text = str(row["evidence_text"]).strip()
    elif row.get("evidence") and isinstance(row["evidence"], list) and len(row["evidence"]) > 0:
        evidence_text = row["evidence"][0].get("evidence_text", "").strip()

    # 路由选择并执行检索
    if config.mode == "linear":
        retrieved_items = execute_linear_retrieval(row, actual_db_doc_name, config, company_list, year_list)
    else:
        retrieved_items = execute_baseline_retrieval(row, actual_db_doc_name, config)

    retrieved_contexts = [entity.get("text", "") if isinstance(entity, dict) else str(entity) for entity in retrieved_items]
    combined_context = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No reference available."

    eval_result = generate_and_evaluate(question, combined_context, gold_answer, evidence_text)

    # 统计 Token 级别的召回率重合度
    evidence_tokens = set(re.findall(r'\w+', evidence_text.lower()))
    pool_tokens = set(re.findall(r'\w+', combined_context.lower()))
    token_recall = len(evidence_tokens.intersection(pool_tokens)) / len(evidence_tokens) if evidence_tokens else 0.0

    return {
        "问题": question,
        "recall内容": combined_context,
        "recall中了没有": eval_result["evidence_matched"],
        "Refined_Recall_Score": token_recall,
        "COT差距": eval_result["cot_diff"],
        "思维链对齐": eval_result["cot_aligned"],
        "答案": eval_result["clean_answer"],
        "答案对否": eval_result["is_correct"],
        "准确答案": gold_answer,
        "context_len": len(combined_context)
    }


def _worker_wrapper(args_tuple: Tuple) -> Dict[str, Any]:
    """多线程包装器：负责序号注入与布尔类型持久化适配"""
    row_idx, row, config, normalized_mapping, company_list, year_list = args_tuple
    result = process_single_row(row, config, normalized_mapping, company_list, year_list)
    result["问题序号"] = row_idx

    for key, value in result.items():
        if isinstance(value, bool):
            result[key] = "true" if value else "false"
    return result


# ==============================================================================
# 🎬 7. 主控评估引擎调度器
# ==============================================================================

def evaluate_rag_ablation_parallel(config: RAGPipelineConfig) -> List[int]:
    """主控入口：驱动多线程执行评测，聚合指标并导出标准化结果"""
    logger.info(f"⚡ [消融实验启动] 模式: 【{config.mode.upper()}】 | 环境: 【{config.env.upper()}】")

    eval_rows, normalized_mapping, company_list, year_list = prepare_dataset(config)

    strict_json_path = Path(config.report_path.replace(".json", f"_{config.env}_{config.mode}_strict_details.json"))
    strict_json_path.parent.mkdir(parents=True, exist_ok=True)

    logs, all_lens = [], []
    stats = {"retrieval": 0, "generation": 0, "cot_align": 0, "total": 0}

    task_args = [
        (idx, row, config, normalized_mapping, company_list, year_list)
        for idx, row in enumerate(eval_rows, start=1)
    ]

    logger.info(f"并发流水线启动 (线程数: {config.max_workers})，总样本量: {len(task_args)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {executor.submit(_worker_wrapper, arg): arg for arg in task_args}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="消融实验评测中"):
            try:
                row_res = future.result()
                stats["total"] += 1
                if row_res.get("recall中了没有") == "true":
                    stats["retrieval"] += 1
                if row_res.get("答案对否") == "true":
                    stats["generation"] += 1
                if row_res.get("思维链对齐") == "true":
                    stats["cot_align"] += 1

                all_lens.append(row_res.pop("context_len", 0))
                logs.append(row_res)
            except Exception as e:
                logger.error(f"样本处理异常: {e}", exc_info=True)

    total_valid = stats["total"] or 1
    logs.sort(key=lambda x: x["问题序号"])

    final_report = {
        "总体消融看板": {
            "运行模式": config.mode,
            "评测环境": config.env,
            "总评测题数": stats["total"],
            "证据匹配召回率(%)": round((stats["retrieval"] / total_valid) * 100, 2),
            "答案生成准确率(%)": round((stats["generation"] / total_valid) * 100, 2),
            "思维链推理对齐率(%)": round((stats["cot_align"] / total_valid) * 100, 2),
            "检索Top_K上限": config.top_k
        },
        "明细数据": logs
    }

    with open(strict_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    logger.info(f"评测完成，看板汇总:\n"
                f"  • 评测题数: {stats['total']}\n"
                f"  • 证据匹配召回率: {final_report['总体消融看板']['证据匹配召回率(%)']}%\n"
                f"  • 答案生成准确率: {final_report['总体消融看板']['答案生成准确率(%)']}%\n"
                f"  • 严格审计报告落盘至: {strict_json_path}")

    return all_lens


# ==============================================================================
# 🚀 8. 执行入口
# ==============================================================================

if __name__ == "__main__":
    run_cfg = RAGPipelineConfig(
        mode="linear",
        env="sharedStore",
        top_k=3,
        k_hop=3,
        window_size=10,
        max_workers=16,
        sample_fraction=None
    )

    context_lengths = evaluate_rag_ablation_parallel(run_cfg)
    if context_lengths:
        print(f"\n📊 平均召回上下文长度 (Tokens/Chars): {np.mean(context_lengths):.2f}")