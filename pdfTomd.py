import asyncio
import os
import re
import aiofiles
import logging
import warnings
from pathlib import Path
from typing import MutableMapping

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# --- 屏蔽干扰日志 ---
warnings.filterwarnings('ignore', message='.*Token indices.*')
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- 1. 全局单例初始化 (关键：只加载一次) ---
EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"

# print(f"🚀 Loading Embedding Model: {EMBED_MODEL_ID}...")
# 全局模型实例
_model = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)
_model.max_seq_length = 512

# 全局 Tokenizer 实例
_tokenizer_hf = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
_tokenizer_docling = HuggingFaceTokenizer(tokenizer=_tokenizer_hf, max_tokens=512)

class BGEEmbeddingFunction:
    def ndims(self):
        return _model.get_sentence_embedding_dimension()
    
    def compute_source_embeddings(self, texts):
        return _model.encode(texts, convert_to_numpy=True)

embedding_func = BGEEmbeddingFunction()

class FinancialReports(LanceModel):
    vector: Vector(embedding_func.ndims())
    text: str
    source: str
    filename: str
    stock_code: str
    annual: str
    report_type: str
    chunk_index: int
    pages: str
    section: str

# --- 2. 核心处理逻辑 ---

async def process_pdf_to_vector_task(task_id: str, pdf_path: str, tasks_registry: MutableMapping, base_output_dir: str):
    """
    base_output_dir: 从 main.py 传入的绝对路径，确保路径一致
    """
    try:
        # 配置 Docling (关闭不必要的 OCR 以提升速度)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True  # 财报建议保留表格，若追求极致速度可改为 False

        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        tasks_registry[task_id]["progress"] = "1/3 Converting PDF to Markdown..."
        
        # 转换 PDF
        result = await asyncio.to_thread(converter.convert, pdf_path)
        markdown_content = result.document.export_to_markdown()

        # 路径处理
        pdf_p = Path(pdf_path)
        md_dir = Path(base_output_dir).joinpath("md")
        md_dir.mkdir(parents=True, exist_ok=True)
        md_name = pdf_p.name.replace(".pdf", ".md")
        md_abs_path = md_dir.joinpath(md_name)

        async with aiofiles.open(md_abs_path, "w", encoding="utf-8") as f:
            await f.write(markdown_content)

        tasks_registry[task_id]["progress"] = "2/3 Embedding and Indexing..."

        # 向量化与入库
        def run_indexing():
            lancedb_dir = Path(base_output_dir).joinpath("lancedb")
            lancedb_dir.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(lancedb_dir))
            
            table_name = "financial_reports_bge"

            # 解析元数据
            name_parts = re.findall(r'\d+', md_name)
            s_code = name_parts[0] if len(name_parts) > 0 else "unknown"
            s_year = name_parts[1] if len(name_parts) > 1 else "unknown"
            
            r_type = "unknown"
            if any(k in md_name for k in ["半年度", "半年报"]): r_type = "半年度报"
            elif any(k in md_name for k in ["年度报告", "年报"]): r_type = "年报"
            elif "一季度" in md_name: r_type = "一季度报"
            elif "三季度" in md_name: r_type = "三季度报"

            # 使用全局定义的 chunker
            chunker = HybridChunker(tokenizer=_tokenizer_docling, merge_peers=True)
            chunks = list(chunker.chunk(result.document))

            data = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunker.contextualize(chunk)
                page_numbers = list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov))

                data.append({
                    "text": chunk_text,
                    "vector": embedding_func.compute_source_embeddings([chunk_text])[0],
                    "source": str(md_abs_path),
                    "filename": md_name,
                    "stock_code": s_code,
                    "annual": s_year,
                    "report_type": r_type,
                    "chunk_index": i,
                    "pages": str(page_numbers),
                    "section": " > ".join(chunk.meta.headings) if chunk.meta.headings else "正文"
                })

            # 写入数据库
            table_names = db.list_tables().tables if hasattr(db.list_tables(), 'tables') else db.list_tables()
            if table_name in table_names:
                table = db.open_table(table_name)
                # 覆盖式写入：先删后加
                table.delete(f"stock_code = '{s_code}' AND annual = '{s_year}' AND report_type = '{r_type}'")
                table.add(data)
            else:
                table = db.create_table(table_name, schema=FinancialReports, data=data)
                # 创建全文搜索索引（解决 "Cannot perform full text search" 错误）
                table.create_fts_index("text", replace=True)
            
            return len(data)

        doc_count = await asyncio.to_thread(run_indexing)

        tasks_registry[task_id].update({
            "status": "completed",
            "progress": "3/3 100%",
            "result": f"Success: {doc_count} chunks indexed to {md_name}",
            "md_path": str(md_abs_path)
        })

    except Exception as e:
        logging.error(f"Task {task_id} failed: {str(e)}")
        tasks_registry[task_id].update({
            "status": "failed",
            "progress": "Error",
            "result": f"Failed: {str(e)}"
        })