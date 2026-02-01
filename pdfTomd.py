import asyncio
import os
import aiofiles
from pathlib import Path
from typing import MutableMapping
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import chromadb
from chromadb.utils import embedding_functions


async def process_pdf_to_vector_task(task_id: str, pdf_path: str, tasks_registry: MutableMapping):
    """
    核心转换逻辑：PDF -> Markdown -> ChromaDB
    """
    try:
        tasks_registry[task_id]["progress"] = "Converting PDF (Docling)..."

        # 1. 转换 PDF (耗时操作，放入线程池)
        pdf_p = Path(pdf_path)
        converter = DocumentConverter()
        # 使用 to_thread 防止阻塞事件循环
        result = await asyncio.to_thread(converter.convert, pdf_path)

        # 2. 导出并保存 Markdown
        tasks_registry[task_id]["progress"] = "Saving Markdown..."
        markdown_content = result.document.export_to_markdown()

        md_path = pdf_p.parent.parent.joinpath("md")
        md_path.mkdir(parents=True, exist_ok=True)
        md_name = pdf_p.name.replace(".pdf", ".md")
        md_abs_path = md_path.joinpath(md_name)

        async with aiofiles.open(md_abs_path, "w", encoding="utf-8") as f:
            await f.write(markdown_content)

        # 3. 向量化入库 (耗时操作)
        tasks_registry[task_id]["progress"] = "Embedding and Indexing..."

        def run_indexing():
            output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))
            chrome_client = chromadb.PersistentClient(path=output_dir)

            # 使用指定的 BGE 模型
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
            collection = chrome_client.get_or_create_collection(name="financial_reports", embedding_function=ef)

            # 从文件名中尝试提取元数据 (假设文件名格式如: 000001_2023_年报.pdf)
            # 这样查询时就可以直接根据 stock_code 过滤了
            name_parts = md_name.replace(".md", "").split("_")
            s_code = name_parts[0] if len(name_parts) > 0 else "unknown"
            s_year = name_parts[1] if len(name_parts) > 1 else "unknown"
            r_type = name_parts[2] if len(name_parts) > 2 else "unknown"

            chunker = HybridChunker()
            chunks = list(chunker.chunk(result.document))

            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(chunks):
                chunk_text = chunker.contextualize(chunk)
                page_numbers = list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov))

                # 核心优化：增加精准过滤字段
                meta = {
                    "source": str(md_abs_path),
                    "filename": md_name,
                    "stock_code": s_code,  # 用于精准 where 过滤
                    "annual": s_year,  # 用于精准 where 过滤
                    "report_type": r_type,  # 用于精准 where 过滤
                    "chunk_index": i,
                    "pages": str(page_numbers),
                    "section": " > ".join(chunk.meta.headings) if chunk.meta.headings else "正文"
                }
                documents.append(chunk_text)
                metadatas.append(meta)
                ids.append(f"{md_name}_chunk_{i}")

            # 分批处理防止由于 documents 过大导致 RPC 报错（可选优化）
            collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            return len(documents)

        # 在线程池中执行重度计算
        doc_count = await asyncio.to_thread(run_indexing)

        # 4. 成功回调
        tasks_registry[task_id].update({
            "status": "completed",
            "progress": "100%",
            "result": f"Success: {md_name} processed. {doc_count} chunks indexed.",
            "md_path": str(md_abs_path)
        })

    except Exception as e:
        tasks_registry[task_id].update({
            "status": "failed",
            "progress": "Error",
            "result": f"Failed: {str(e)}"
        })