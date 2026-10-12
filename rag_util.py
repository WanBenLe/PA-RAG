#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块说明:
该脚本实现了面向金融复杂长文档问答的 Dense-Sparse 结合检索、
拓扑平滑重排 (SERAG 局部窗口高斯平滑) 与多模型多阶段判决评测流。
优化重心：配置解耦、网络异常健壮性、NumPy 批量计算提速与边界处理。
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from openai import OpenAI
from datasketch import MinHash

# ==================== 🛠️ 日志与基础运行时配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FinanceRAG")

# 隔离与环境配置 (可通过宿主机外部变量无缝覆盖)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ==================== ⚙️ 全局配置载入 (消除硬编码) ====================
class Config:
    """系统核心参数与服务端口配置"""
    # 路径配置
    PDF_CACHE_DIR = Path(os.getenv("PDF_CACHE_DIR", "/root/autodl-tmp/financebench/pdfs"))

    # 服务 Base URL 与 API 密钥
    SGLANG_VLLM_URL = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    EMBED_VLLM_URL = os.getenv("EMBED_VLLM_URL", "http://127.0.0.1:8001/v1")
    RERANK_QWEN_URL = os.getenv("RERANK_QWEN_URL", "http://127.0.0.1:8002/v1/rerank")
    RERANK_JINA_URL = os.getenv("RERANK_JINA_URL", "http://127.0.0.1:8003/v1/rerank")
    JINA_EMBED_URL = os.getenv("JINA_EMBED_URL", "http://127.0.0.1:8004/v1")

    # 外部第三方 API
    DS_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DS_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    # 模型名称
    MODEL_GEN = os.getenv("MODEL_GEN", "Qwen3.6-35B-A3B-AWQ")
    MODEL_QWEN_EMBED = os.getenv("MODEL_QWEN_EMBED", "Qwen3-Embedding-0.6B")
    MODEL_QWEN_RERANK = os.getenv("MODEL_QWEN_RERANK", "Qwen3-Reranker-0.6B")
    MODEL_JINA_EMBED = os.getenv("MODEL_JINA_EMBED", "jina-embeddings-v5")
    MODEL_JINA_RERANK = os.getenv("MODEL_JINA_RERANK", "jina-reranker-v3.5")
    MODEL_JUDGE = os.getenv("MODEL_JUDGE", "deepseek-v4-flash")

    # HTTP 超时阈值 (秒)
    TIMEOUT_SHORT = 15.0
    TIMEOUT_MID = 30.0
    TIMEOUT_LONG = 60.0


# ==================== 🚀 连接池与客户端单例初始化 ====================
# HTTP/API Keep-Alive 连接池
RERANK_SESSION = requests.Session()

# 基础模型客户端实例化
EMBED_CLIENT = OpenAI(api_key="EMPTY", base_url=Config.EMBED_VLLM_URL, timeout=Config.TIMEOUT_MID)
SGLANG_CLIENT = OpenAI(api_key="EMPTY", base_url=Config.SGLANG_VLLM_URL, timeout=Config.TIMEOUT_LONG)
JINA_EMBED_CLIENT = OpenAI(api_key="EMPTY", base_url=Config.JINA_EMBED_URL, timeout=Config.TIMEOUT_MID)
DS_CLIENT = (
    OpenAI(api_key=Config.DS_API_KEY, base_url=Config.DS_BASE_URL, timeout=Config.TIMEOUT_MID)
    if Config.DS_API_KEY else None
)


# ==================== ⚡ 正则表达式全局预编译 ====================
WORD_PATTERN = re.compile(r'[\w\u4e00-\u9fa5]+')
YEAR_PATTERN = re.compile(r'_(\d{4})')
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
YEAR_SCRUB_PATTERN = re.compile(r'(?i)(fy|year\s+)?\b(201[0-9]|202[0-9])\b')


# ==================== 📦 核心算法与检索函数 ====================

def fetch_and_normalized_serag_filter_chunks(
    cluster_dict: Dict[str, List[Tuple[int, int]]],
    batch_query_vectors: List[List[float]],
    anchor_nids: Set[str],
    milvus_client: Any,
    lambda_1: float = 0.7,
    lambda_2: float = 0.1,
    lambda_3: float = 0.2,
    sigma: float = 3.0,
    threshold: float = 0.6
) -> str:
    """
    SERAG (Structure-Enhanced RAG) 拓扑感知过滤算法:
    结合语义匹配度、锚点精准命中、物理行号高斯邻近距离三个维度综合加权筛选 Chunk。
    """
    # 提取目标主键 ID
    target_ids = [
        f"{doc_name}_LN_{i:04d}"
        for doc_name, intervals in cluster_dict.items()
        for start_idx, end_idx in intervals
        for i in range(start_idx, end_idx)
    ]
    if not target_ids:
        return ""

    # 从 Milvus 批量拉取数据
    id_list_str = ",".join(f'"{nid}"' for nid in target_ids)
    raw_chunks_res = milvus_client.query(
        collection_name="linear_rag_chunks",
        filter=f"node_id in [{id_list_str}]",
        output_fields=["node_id", "text", "vector"]
    )
    chunk_map = {r["node_id"]: r for r in raw_chunks_res}
    q_vecs = np.asarray(batch_query_vectors, dtype=np.float32)
    final_context_blocks = []

    for doc_name, intervals in cluster_dict.items():
        doc_nids = []
        doc_vecs = []

        # 整理单文档命中切片
        for start_idx, end_idx in intervals:
            for i in range(start_idx, end_idx + 1):
                nid = f"{doc_name}_LN_{i:04d}"
                if nid in chunk_map:
                    doc_nids.append(nid)
                    doc_vecs.append(chunk_map[nid]["vector"])

        if not doc_nids:
            continue

        c_vecs = np.asarray(doc_vecs, dtype=np.float32)
        # 语义向量内积 (已归一化，直接矩阵点积代表余弦相似度)
        raw_sems = np.max(np.dot(c_vecs, q_vecs.T), axis=1)

        # 基于物理行号的高斯衰减拓扑建模
        curr_lines = np.array([int(nid[-4:]) for nid in doc_nids], dtype=np.int32)
        anchor_indices = [int(a[-4:]) for a in anchor_nids if a.startswith(doc_name)]

        if anchor_indices:
            diffs = np.abs(curr_lines[:, None] - np.array(anchor_indices, dtype=np.int32))
            min_dists = np.min(diffs, axis=1)
            w_dists = np.exp(-(min_dists ** 2) / (2.0 * (sigma ** 2)))
        else:
            w_dists = np.zeros(len(doc_nids), dtype=np.float32)

        # 锚点命中逻辑增益与非负语义激活
        w_logs = np.array([1.0 if nid in anchor_nids else 0.0 for nid in doc_nids], dtype=np.float32)
        w_sems = np.maximum(0.0, raw_sems)

        # 综合评分矩阵 (加权叠加)
        final_scores_arr = lambda_1 * w_sems + lambda_2 * w_logs + lambda_3 * w_dists
        doc_scores = dict(zip(doc_nids, final_scores_arr))

        # 兜底机制：若全部未达阈值，回退保留语义最强点所在的区间
        max_final_score = np.max(final_scores_arr) if len(final_scores_arr) > 0 else 0.0
        fallback_interval = None
        if max_final_score < threshold:
            best_idx = np.argmax(raw_sems)
            best_line_num = curr_lines[best_idx]
            for start_idx, end_idx in intervals:
                if start_idx <= best_line_num <= end_idx:
                    fallback_interval = (start_idx, end_idx)
                    break

        final_context_blocks.append(f"\n================ [ {doc_name} ] ================")
        for start_idx, end_idx in intervals:
            if fallback_interval and (start_idx, end_idx) != fallback_interval:
                continue

            interval_texts = []
            for i in range(start_idx, end_idx + 1):
                nid = f"{doc_name}_LN_{i:04d}"
                if nid not in doc_scores:
                    continue

                is_passed = (doc_scores[nid] >= threshold) or (fallback_interval is not None)
                if is_passed:
                    interval_texts.append(f"--- [LN_{i:04d}] ---\n{chunk_map[nid]['text']}")
                else:
                    if not interval_texts or interval_texts[-1] != "[...] (Irrelevant chunks omitted)":
                        interval_texts.append("[...] (Irrelevant chunks omitted)")

            if interval_texts:
                final_context_blocks.append("\n".join(interval_texts))

    return "\n\n".join(final_context_blocks)


def year_filter(initial_sorted_ids: List[str], raw_years: List[Any]) -> List[str]:
    """根据年份正则过滤目标切片 ID 列表"""
    allowed_years = {str(y) for y in raw_years}
    return [
        nid for nid in initial_sorted_ids
        if (match := YEAR_PATTERN.search(nid)) and match.group(1) in allowed_years
    ]


def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """计算文本的 MinHash 签名，用于快速局部敏感哈希 (LSH) 去重与查重"""
    m = MinHash(num_perm=num_perm)
    words = set(WORD_PATTERN.findall(text.lower()))
    for word in words:
        m.update(word.encode("utf-8"))
    return m


def bridge_hit_chunks(
    hit_indices: List[int],
    max_gap: int = 4,
    padding: int = 1,
    scatter_threshold: float = 0.8
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    单遍扫描状态机：
    连通距离 <= max_gap 的行切片，执行填充并在离散度高于阈值时标记 needs_rerank。
    """
    if not hit_indices:
        return [], False

    sorted_hits = sorted(set(hit_indices))
    n_hits = len(sorted_hits)
    if n_hits == 1:
        return [(max(0, sorted_hits[0] - padding), sorted_hits[0] + padding)], False

    merged_intervals = []
    clusters_count = 0
    isolated_count = 0

    start_hit = sorted_hits[0]
    prev_hit = sorted_hits[0]
    curr_len = 1

    for i in range(1, n_hits):
        curr_hit = sorted_hits[i]
        if curr_hit - prev_hit <= max_gap:
            curr_len += 1
        else:
            clusters_count += 1
            if curr_len == 1:
                isolated_count += 1

            int_start = max(0, start_hit - padding)
            int_end = prev_hit + padding
            if merged_intervals and int_start <= merged_intervals[-1][1] + 1:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], int_end))
            else:
                merged_intervals.append((int_start, int_end))

            start_hit = curr_hit
            curr_len = 1
        prev_hit = curr_hit

    # 收尾最后一个聚类块
    clusters_count += 1
    if curr_len == 1:
        isolated_count += 1
    int_start = max(0, start_hit - padding)
    int_end = prev_hit + padding
    if merged_intervals and int_start <= merged_intervals[-1][1] + 1:
        merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], int_end))
    else:
        merged_intervals.append((int_start, int_end))

    scatter_ratio = isolated_count / n_hits
    needs_rerank = (clusters_count == n_hits) or (scatter_ratio >= scatter_threshold)

    return merged_intervals, needs_rerank


def parse_hit_nodes(node_list: List[str]) -> Dict[str, List[int]]:
    """解析以 '_LN_' 拼接的 node_id，返回每个文档按行号升序排列的字典"""
    parsed_dict = defaultdict(list)
    for node_id in node_list:
        doc_name, sep, line_str = node_id.rpartition("_LN_")
        if sep and line_str.isdigit():
            parsed_dict[doc_name].append(int(line_str))

    for doc in parsed_dict:
        parsed_dict[doc].sort()
    return dict(parsed_dict)


# ==================== 🎯 重排与向量化客户端接口 ====================

def call_vllm_rerank_api(query: str, documents: List[str], top_k: int, instruction: str = "") -> List[int]:
    """基于 Qwen3-Reranker-0.6B 的重排序接口"""
    if not documents:
        return []

    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    formatted_query = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    formatted_documents = [f"<Document>: {doc}{suffix}" for doc in documents]

    payload = {
        "model": Config.MODEL_QWEN_RERANK,
        "query": formatted_query,
        "documents": formatted_documents
    }

    try:
        response = RERANK_SESSION.post(Config.RERANK_QWEN_URL, json=payload, timeout=Config.TIMEOUT_SHORT)
        if response.status_code == 200:
            rerank_res = response.json().get("results", [])
            rerank_res.sort(key=lambda x: x["relevance_score"], reverse=True)
            return [hit["index"] for hit in rerank_res][:top_k]
        logger.warning(f"Qwen Rerank 异常响应 [{response.status_code}]: {response.text}")
    except Exception as e:
        logger.error(f"Qwen Rerank 网络异常: {e}")

    return list(range(min(top_k, len(documents))))


def call_vllm_rerank_jina_api(query: str, documents: List[str], top_k: int, instruction: str = "") -> List[int]:
    """基于 Jina-Reranker-v3.5 的重排序接口"""
    if not documents:
        return []

    combined_query = f"<instruct>\n{instruction}\n</instruct>\n{query}" if instruction else query
    payload = {
        "model": Config.MODEL_JINA_RERANK,
        "query": combined_query,
        "texts": documents,
        "documents": documents
    }

    try:
        response = RERANK_SESSION.post(Config.RERANK_JINA_URL, json=payload, timeout=Config.TIMEOUT_SHORT)
        if response.status_code == 200:
            rerank_res = response.json().get("results", [])
            rerank_res.sort(key=lambda x: x["relevance_score"], reverse=True)
            return [hit["index"] for hit in rerank_res][:top_k]
        logger.warning(f"Jina Rerank 异常响应 [{response.status_code}]: {response.text}")
    except Exception as e:
        logger.error(f"Jina Rerank 网络异常: {e}")

    return list(range(min(top_k, len(documents))))


def call_jina_embedding_api(text: str, is_query: bool = True, dim: int = 1024) -> List[float]:
    """Jina-v5 单文本向量化 (支持 MRL 维度截断与 L2 重归一化)"""
    if not text:
        return []
    prefix = "Query: " if is_query else "Document: "
    try:
        response = JINA_EMBED_CLIENT.embeddings.create(
            model=Config.MODEL_JINA_EMBED,
            input=[f"{prefix}{text}"]
        )
        raw_emb = np.array(response.data[0].embedding[:dim], dtype=np.float32)
        norm = np.linalg.norm(raw_emb)
        return (raw_emb / norm).tolist() if norm > 0 else raw_emb.tolist()
    except Exception as e:
        logger.error(f"Jina Embedding 编码失败: {e}")
        return []


def call_jina_embedding_batch_api(texts: List[str], is_query: bool = True, dim: int = 1024) -> List[List[float]]:
    """Jina-v5 批量文本编码器"""
    if not texts:
        return []
    prefix = "Query: " if is_query else "Document: "
    formatted_texts = [f"{prefix}{t}" for t in texts]

    try:
        response = JINA_EMBED_CLIENT.embeddings.create(
            model=Config.MODEL_JINA_EMBED,
            input=formatted_texts
        )
        raw_embs = np.array([d.embedding[:dim] for d in response.data], dtype=np.float32)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (raw_embs / norms).tolist()
    except Exception as e:
        logger.error(f"Jina Batch Embedding 批量编码失败: {e}")
        return []


def call_vllm_embedding_api(text: str, dim: int) -> List[float]:
    """Qwen-Embedding 单文本向量化"""
    try:
        response = EMBED_CLIENT.embeddings.create(
            model=Config.MODEL_QWEN_EMBED,
            input=[text]
        )
        raw_emb = np.array(response.data[0].embedding[:dim], dtype=np.float32)
        norm = np.linalg.norm(raw_emb)
        return (raw_emb / norm).tolist() if norm > 0 else raw_emb.tolist()
    except Exception as e:
        logger.error(f"Qwen Embedding 编码失败: {e}")
        return []


def call_vllm_embedding_batch_api(texts: List[str], dim: int) -> List[List[float]]:
    """Qwen-Embedding 批量文本编码器"""
    if not texts:
        return []
    try:
        response = EMBED_CLIENT.embeddings.create(
            model=Config.MODEL_QWEN_EMBED,
            input=texts
        )
        raw_embs = np.array([d.embedding[:dim] for d in response.data], dtype=np.float32)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (raw_embs / norms).tolist()
    except Exception as e:
        logger.error(f"Qwen Batch Embedding 批量编码失败: {e}")
        return []


# ==================== 🧠 生成模型与审计评估 ====================

def call_sglang_generation(prompt: str, system_message: str, thinking: bool = False) -> Dict[str, str]:
    """SGLang 大语言模型文本生成 (支持思考过程抽取)"""
    try:
        response = SGLANG_CLIENT.chat.completions.create(
            model=Config.MODEL_GEN,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=(0.6 if thinking else 0.7),
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}}
        )
        message = response.choices[0].message
        agent_thinking = getattr(message, "reasoning_content", "")
        if not agent_thinking and hasattr(message, "model_extra") and message.model_extra:
            agent_thinking = message.model_extra.get("reasoning_content", "")

        return {
            "thinking": agent_thinking.strip() if agent_thinking else "",
            "content": (message.content.strip() if message.content else "")
        }
    except Exception as e:
        logger.error(f"SGLang 生成调用异常: {e}")
        return {"thinking": "", "content": f"Generation Failure Fallback: {str(e)}"}


def realtime_eval_and_fix(raw_text: str) -> Dict[str, Any]:
    """解析或修复 LLM 产生的带中文引号或转义破损的非标准 JSON 字符串"""
    if not raw_text or not isinstance(raw_text, str):
        return {}

    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    json_content = match.group(0) if match else raw_text.strip()
    json_content = json_content.replace('\\n', '')

    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        pass

    # 尝试修复由于英文双引号被文本引用嵌套破坏的场景
    fixed_lines = []
    for line in json_content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"') or stripped.endswith('"') or stripped.endswith('",'):
            quote_indices = [m.start() for m in re.finditer(r'"', line)]
            if len(quote_indices) > 2:
                line_chars = list(line)
                for mid_idx in quote_indices[1:-1]:
                    left_part = line_chars[:mid_idx]
                    line_chars[mid_idx] = "“" if left_part.count("“") == left_part.count("”") else "”"
                line = "".join(line_chars)
        fixed_lines.append(line)

    try:
        return json.loads("\n".join(fixed_lines))
    except Exception:
        return {}


def extract_clean_financial_answer(text: str) -> str:
    """过滤思考链标签，返回最终回答"""
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def verify_numeric_tolerance(agent_ans: str, gold_ans: str, tolerance: float = 0.05) -> bool:
    """提取回答中的核心浮点数值，进行容差范围内的数值正确性检验"""
    def extract_first_float(text: str) -> Optional[float]:
        text_scrubbed = YEAR_SCRUB_PATTERN.sub('', text)
        cleaned = text_scrubbed.replace(",", "").replace("$", "").replace("%", "")
        match = FLOAT_PATTERN.search(cleaned)
        if not match:
            match = FLOAT_PATTERN.search(text.replace(",", "").replace("$", "").replace("%", ""))
        return float(match.group()) if match else None

    try:
        a_val = extract_first_float(agent_ans)
        g_val = extract_first_float(gold_ans)
        if a_val is not None and g_val is not None:
            if g_val == 0:
                return a_val == 0
            return abs(a_val - g_val) / abs(g_val) <= tolerance
    except Exception:
        pass
    return False


def run_accuracy_judge(question: str, agent_ans: str, gold_ans: str) -> bool:
    """基于 DeepSeek 审计裁判模型的准确率判定"""
    if not DS_CLIENT:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过 Accuracy 评审并兜底返回 False")
        return False

    query_prompt = f"- Query: {question}" if question else ""
    judge_prompt = f"""
You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers the query based on the golden answer provided by a human expert.

Numerical Accuracy: 
- Rounding differences should be **ignored** if they do not meaningfully change the conclusion.
- You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23.
- Fractions, percentage, and numerics could be considered similar (e.g., "11 of 14" is equivalent to "79%" and "0.79").

Evaluation Criteria:
- If the golden answer or any of its equivalence can be inferred from the AI answer, mark True.
- If the AI answer conveys the same rationale, conclusions, or is a superset of the golden answer, mark True.
- Otherwise, mark False.

Inputs:
{query_prompt}
- AI-Generated Answer: {agent_ans}
- Golden Answer: {gold_ans}

Your output should be ONLY a boolean value: `True` or `False`.
"""

    try:
        response = DS_CLIENT.chat.completions.create(
            model=Config.MODEL_JUDGE,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=2560,
            extra_body={"thinking": {"type": "enabled"}}
        )
        raw_res = response.choices[0].message.content.strip().lower()
        if "true" in raw_res:
            return True
        if "false" in raw_res:
            return False
        logger.warning(f"Judge 判决异常解析输出: {raw_res}")
    except Exception as e:
        logger.error(f"Judge 评审 API 异常: {e}")

    return False


def run_recall_judge(question: str, evidence: str, combined_context: str) -> Dict[str, Any]:
    """基于 DeepSeek 审计裁判模型的上下文召回率判定"""
    if not DS_CLIENT:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过 Recall 评审")
        return {}

    safe_context_snippet = str(combined_context).strip()
    if len(safe_context_snippet) > 6000:
        safe_context_snippet = safe_context_snippet[:6000] + "\n[Truncated for context size safety]..."

    judge_system = (
        "You are an expert multi-dimensional financial auditing judge. Your task is to evaluate the correctness, "
        "retrieval recall compliance, and thinking process quality of a financial RAG system.\n"
    )
    judge_prompt = (
        f"Determine if the Retrieved Context contains the necessary information to address the Question based on the Gold Evidence.\n\n"
        f"Question: {question}\n"
        f"Gold Evidence: {str(evidence).strip()}\n"
        f"Retrieved Context: {safe_context_snippet}\n\n"
        f"Rule: Mark true if the Retrieved Context contains or covers the core facts/numbers required to solve the query; otherwise false.\n\n"
        f"Output the result as a JSON object: {{\"evidence_matched\": true/false}}"
    )

    try:
        response = DS_CLIENT.chat.completions.create(
            model=Config.MODEL_JUDGE,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.1,
            extra_body={"thinking": {"type": "enabled"}},
            max_tokens=5120,
            response_format={"type": "json_object"}
        )
        return realtime_eval_and_fix(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Recall Judge API 异常: {e}")
        return {}