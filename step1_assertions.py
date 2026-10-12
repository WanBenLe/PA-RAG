import os
import sys
import re
import pickle
import asyncio
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pymupdf
import openai
from pymilvus import MilvusClient, DataType
import nest_asyncio
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm as async_tqdm

nest_asyncio.apply()

# ==============================================================================
# ⚙️ 全局工程配置 (解耦硬编码，支持环境变量覆盖)
# ==============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """流水线全局超参数与存储配置."""
    # 路径与硬件配置
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "1")
    pdf_dir: Path = Path(os.getenv("PDF_CACHE_DIR", "/root/autodl-tmp/financebench/pdfs"))
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR", "/root/autodl-tmp/nanobot/workspace/memory"))
    
    # 缓存中间产物文件名
    stage1_cache_file: str = "stage1_parsed_nodes.pkl"
    stage2_cache_file: str = "stage2_embedded_nodes.pkl"

    # Milvus 向量库配置
    db_filename: str = "PA_rag_decoupled.db"
    chunks_collection: str = "PA_rag_chunks"
    assertions_collection: str = "PA_rag_assertions"
    vector_dim: int = 512
    metric_type: str = "IP"  # 内积度量 (归一化向量即等价于 Cosine 相似度)

    # vLLM / Embedding 服务配置
    embed_base_url: str = os.getenv("EMBED_VLLM_URL", "http://127.0.0.1:8004/v1")
    embed_model_name: str = os.getenv("SERVED_MODEL_NAME", "jina-embeddings-v5")
    embed_batch_size: int = 64
    api_concurrency: int = 16
    request_timeout: float = 60.0

    # 文本切分与清洗超参数
    max_chunk_words: int = 180
    overlap_words: int = 35

    @property
    def db_path(self) -> str:
        """解析数据库绝对路径并确保目录存在."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return str(self.workspace_dir / self.db_filename)

    @property
    def stage1_cache_path(self) -> Path:
        return self.workspace_dir / self.stage1_cache_file

    @property
    def stage2_cache_path(self) -> Path:
        return self.workspace_dir / self.stage2_cache_file


CONFIG = PipelineConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.cuda_visible_devices


# ==============================================================================
# 🗄️ 阶段零：Milvus 模式初始化与集合重建
# ==============================================================================
def init_milvus_collections(cfg: PipelineConfig) -> None:
    """
    重建 Milvus 集合架构，构建对应的 IVF_FLAT / HNSW 索引以提升检索性能。
    """
    print(f"\n⚡ 连接 Milvus: {cfg.db_path}，初始化集合维度 [{cfg.vector_dim}]...")
    client = MilvusClient(cfg.db_path)

    # 清理遗留历史表
    for col in [cfg.chunks_collection, cfg.assertions_collection]:
        if client.has_collection(col):
            client.drop_collection(col)
            print(f"🗑️ 已清理历史集合: {col}")

    # 1. 重建 Chunk 表 Schema
    schema_chunks = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema_chunks.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema_chunks.add_field(field_name="node_id", datatype=DataType.VARCHAR, max_length=128)
    schema_chunks.add_field(field_name="source_doc", datatype=DataType.VARCHAR, max_length=255)
    schema_chunks.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema_chunks.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=cfg.vector_dim)
    
    # 2. 重建 Assertion 表 Schema
    schema_assertions = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema_assertions.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema_assertions.add_field(field_name="node_id", datatype=DataType.VARCHAR, max_length=128)
    schema_assertions.add_field(field_name="tree_like", datatype=DataType.VARCHAR, max_length=4096)
    schema_assertions.add_field(field_name="is_fact", datatype=DataType.BOOL)
    schema_assertions.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=cfg.vector_dim)

    # 3. 准备索引参数并创建集合 (提前声明索引加快后续装载与检索)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type=cfg.metric_type,
        index_type="AUTOINDEX"
    )

    client.create_collection(
        collection_name=cfg.chunks_collection, 
        schema=schema_chunks, 
        index_params=index_params
    )
    client.create_collection(
        collection_name=cfg.assertions_collection, 
        schema=schema_assertions, 
        index_params=index_params
    )

    client.close()
    print(f"🏁 库表与索引模式重建完毕！")


# ==============================================================================
# 🛠️ 基础文本处理与清洗工具
# ==============================================================================
def clean_extracted_layout(text: str) -> str:
    """清理 PDF 排版脏字符及不规则制表空白."""
    if not text:
        return ""
    text = text.replace('\u00a0', ' ')
    cleaned_lines = [
        re.sub(r'[ \t]+', ' ', stripped)
        for line in text.split('\n')
        if (stripped := line.strip())
    ]
    return "\n".join(cleaned_lines)


def sliding_word_chunker(text: str, max_words: int, overlap: int) -> List[str]:
    """单词级滑动窗口切词算法，保障回退策略下的段落语义连贯."""
    words = text.split()
    chunks = []
    idx = 0
    step = max(1, max_words - overlap)
    while idx < len(words):
        chunk_text = " ".join(words[idx : idx + max_words]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        idx += step
    return chunks


# ==============================================================================
# 🟢 阶段一：纯 CPU 多进程 PDF 解析
# ==============================================================================
def worker_parse_pdf(file_path: Path, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """
    独立进程 Worker：解析单个 PDF 文档并抽取层级元数据。
    """
    if file_path.suffix.lower() != ".pdf":
        return []

    structured_entries: List[Dict[str, Any]] = []
    node_counter = 1
    
    try:
        doc = pymupdf.open(file_path)
        part_pattern = re.compile(r'(PART\s+[IVXLCDM]+)', re.IGNORECASE)
        item_pattern = re.compile(r'(Item\s+\d+[A-Z]?\.)', re.IGNORECASE)
        current_part, current_item = "Front Matter", "Overview"
        full_pdf_text_backup = []

        for page_idx, page in enumerate(doc):
            page_text = clean_extracted_layout(page.get_text("text", sort=True))
            if not page_text:
                continue
            full_pdf_text_backup.append(page_text)

            for line in page_text.split("\n"):
                if part_match := part_pattern.search(line):
                    current_part = part_match.group(1).upper()
                if item_match := item_pattern.search(line):
                    current_item = line.strip()[:50]

            hierarchy_path = f"{current_part} -> {current_item}"
            structured_entries.append({
                "text": f"[File: {file_path.name} | {hierarchy_path} | Page: {page_idx + 1}]\n{page_text}",
                "node_id": f"{file_path.stem}_C{node_counter:04d}",
                "parent_id": f"P_{current_part[:4].strip()}",
                "node_title": current_item,
                "hierarchy_path": hierarchy_path,
                "source_doc": file_path.name
            })
            node_counter += 1
        doc.close()

        # 兜底降级策略：若未识别出标准结构，启用滑动窗口兜底
        if not structured_entries and full_pdf_text_backup:
            chunks = sliding_word_chunker(
                "\n".join(full_pdf_text_backup), 
                max_words=cfg.max_chunk_words, 
                overlap=cfg.overlap_words
            )
            for idx, chunk in enumerate(chunks):
                structured_entries.append({
                    "text": f"[Doc: {file_path.name} | Generic Backup] {chunk}",
                    "node_id": f"{file_path.stem}_BK{idx + 1:04d}",
                    "parent_id": "P_BK_ROOT",
                    "node_title": "Document Body Backup",
                    "hierarchy_path": "Root -> Body",
                    "source_doc": file_path.name
                })
        return structured_entries

    except Exception as e:
        print(f"⚠️ 文件解析失败 {file_path.name}: {e}")
        return []


def run_stage_1_parsing(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """阶段一调度入口：支持缓存断点续传及多进程并行解析."""
    print("\n" + "=" * 50)
    print("🟢 阶段一：启动纯 CPU 多进程 PDF 解析")
    print("=" * 50)

    if cfg.stage1_cache_path.exists():
        print(f"✅ 发现阶段一缓存 {cfg.stage1_cache_path}，直接加载！")
        with open(cfg.stage1_cache_path, "rb") as f:
            return pickle.load(f)

    pdf_files = sorted(list(cfg.pdf_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"⚠️ 在路径 {cfg.pdf_dir} 下未发现 PDF 文件。")
        return []

    all_parsed_nodes: List[Dict[str, Any]] = []
    optimal_workers = max(1, (os.cpu_count() or 4) - 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        futures = {executor.submit(worker_parse_pdf, fp, cfg): fp for fp in pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="解析 PDF"):
            if nodes := future.result():
                all_parsed_nodes.extend(nodes)

    print(f"💾 解析完成，提取 {len(all_parsed_nodes)} 个文本块。持久化存储...")
    with open(cfg.stage1_cache_path, "wb") as f:
        pickle.dump(all_parsed_nodes, f)

    return all_parsed_nodes


# ==============================================================================
# 🟡 阶段二：纯网络 I/O 异步特征提取 (Embedding)
# ==============================================================================
async def fetch_batch_embeddings(
    batch_texts: List[str],
    async_client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    cfg: PipelineConfig,
    is_query: bool = False,
    max_retries: int = 3
) -> List[List[float]]:
    """
    单批次向量推理：引入指数退避重试，禁止静默零向量以规避静默检索失效。
    """
    prefix = "Query: " if is_query else "Document: "
    formatted_texts = [f"{prefix}{t}" for t in batch_texts]

    async with sem:
        for attempt in range(max_retries):
            try:
                response = await async_client.embeddings.create(
                    model=cfg.embed_model_name,
                    input=formatted_texts,
                    timeout=cfg.request_timeout
                )
                
                # 截断维度并做 L2 归一化 (适配 Matryoshka 结构)
                raw_embs = np.array([item.embedding[:cfg.vector_dim] for item in response.data], dtype=np.float32)
                norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12  # 数值稳定性
                normalized_embs = raw_embs / norms
                return normalized_embs.tolist()

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n❌ 重试耗尽，批次向量提取失败: {e}")
                    raise e
                await asyncio.sleep(2 ** attempt)


async def run_stage_2_embedding(
    parsed_nodes: List[Dict[str, Any]], 
    cfg: PipelineConfig, 
    is_query: bool = False
) -> List[Dict[str, Any]]:
    """阶段二调度入口：基于信号量的高并发异步批量 Embedding 提取."""
    print("\n" + "=" * 50)
    print(f"🟡 阶段二：启动纯网络 I/O 异步特征提取 ({cfg.embed_model_name})")
    print("=" * 50)

    if cfg.stage2_cache_path.exists():
        print(f"✅ 发现阶段二缓存 {cfg.stage2_cache_path}，直接加载！")
        with open(cfg.stage2_cache_path, "rb") as f:
            return pickle.load(f)

    async_client = openai.AsyncOpenAI(
        api_key="EMPTY", 
        base_url=cfg.embed_base_url, 
        timeout=cfg.request_timeout * 2
    )
    sem = asyncio.Semaphore(cfg.api_concurrency)

    all_texts = [node["text"] for node in parsed_nodes]
    batches = [all_texts[i : i + cfg.embed_batch_size] for i in range(0, len(all_texts), cfg.embed_batch_size)]

    print(f"🚀 待提取样本: {len(all_texts)} 条，分批: {len(batches)} 批，并发通道: {cfg.api_concurrency}")

    tasks = [
        fetch_batch_embeddings(batch, async_client, sem, cfg, is_query=is_query)
        for batch in batches
    ]
    all_vectors_nested = await async_tqdm.gather(*tasks, desc="嵌入特征提取")
    all_vectors = [vec for batch in all_vectors_nested for vec in batch]

    if len(all_vectors) != len(parsed_nodes):
        raise RuntimeError(f"🚨 维度坍缩/数据不齐：特征数 ({len(all_vectors)}) != 节点数 ({len(parsed_nodes)})")

    for i, node in enumerate(parsed_nodes):
        node["vector"] = all_vectors[i]

    print(f"💾 特征向量合并完毕，持久化至 {cfg.stage2_cache_path}...")
    with open(cfg.stage2_cache_path, "wb") as f:
        pickle.dump(parsed_nodes, f)

    return parsed_nodes


# ==============================================================================
# 🚀 主执行入口
# ==============================================================================
def main():
    # 0. 重新构建向量库表结构
    init_milvus_collections(CONFIG)
    
    # 1. 运行文档解析
    nodes = run_stage_1_parsing(CONFIG)
    
    # 2. 运行嵌入提取
    if nodes:
        embedded_nodes = asyncio.run(run_stage_2_embedding(nodes, CONFIG))
        print(f"✨ 流水线前两阶段已全部执行成功，就绪节点数: {len(embedded_nodes)} (｡•̀ᴗ-)✧")


if __name__ == "__main__":
    main()