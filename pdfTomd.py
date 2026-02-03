import asyncio
import os
import re
import aiofiles
from pathlib import Path
from typing import MutableMapping
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# 导入必要的配置类
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pyarrow as pa

# ✅ 核心：过滤 transformers 的警告（增强版）
import warnings
import logging

# 过滤所有与 token length 相关的警告
warnings.filterwarnings('ignore', message='Token indices sequence length')
warnings.filterwarnings('ignore', message='.*Token indices.*')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# 同时降低 transformers 的日志级别
logging.getLogger("transformers").setLevel(logging.ERROR)


# BGE-small 嵌入模型类（轻量高效，512 tokens 完全够用）
class BGEEmbeddingFunction:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        # ✅ 设置最大序列长度，超过会自动截断
        self.model.max_seq_length = 512
        
    def ndims(self):
        return self.model.get_sentence_embedding_dimension()
    
    def compute_source_embeddings(self, texts):
        """计算文本嵌入 - 超过 512 tokens 会自动截断"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def compute_query_embeddings(self, query):
        """计算查询嵌入 - 超过 512 tokens 会自动截断"""
        return self.model.encode([query], convert_to_numpy=True)[0]

# 1. 初始化嵌入模型（改用 BGE-small）
EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"
embedding_func = BGEEmbeddingFunction(EMBED_MODEL_ID)

class FinancialReports(LanceModel):
    vector: Vector(embedding_func.ndims())
    text: str
    source: str
    filename: str
    stock_code: str      # 股票代码，如：002927
    annual: str          # 年份，如：2024
    report_type: str     # 报告类型：一季度报、半年报、三季度报、年报
    chunk_index: int     # 文本块索引
    pages: str           # 页码列表
    section: str         # 章节路径

async def process_pdf_to_vector_task(task_id: str, pdf_path: str, tasks_registry: MutableMapping):

    """
    核心转换逻辑：PDF -> Markdown -> LanceDB

    """
    try:
        # 1. 精细化配置参数
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # 禁用 OCR
        pipeline_options.do_formula_enrichment = False  # 禁用公式识别
        
        # 针对财报，保留表格结构识别
        pipeline_options.do_table_structure = True 

        # 2. 初始化转换器，注入配置
        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # 1. 转换 PDF (耗时操作，放入线程池)
        # converter = DocumentConverter()
        
        tasks_registry[task_id]["progress"] = "Converting PDF (Docling)..."
        
        # 使用 to_thread 防止阻塞事件循环
        result = await asyncio.to_thread(converter.convert, pdf_path)

        # 2. 导出并保存 Markdown
        tasks_registry[task_id]["progress"] = "Saving Markdown..."
        markdown_content = result.document.export_to_markdown()

        pdf_p = Path(pdf_path)

        md_path = pdf_p.parent.parent.joinpath("md")
        md_path.mkdir(parents=True, exist_ok=True)
        md_name = pdf_p.name.replace(".pdf", ".md")
        md_abs_path = md_path.joinpath(md_name)

        async with aiofiles.open(md_abs_path, "w", encoding="utf-8") as f:
            await f.write(markdown_content)

        # 3. 向量化入库 (耗时操作)
        tasks_registry[task_id]["progress"] = "Embedding and Indexing..."

        def run_indexing():
            # ✅ 统一保存到 output/lancedb 目录
            lancedb_dir = Path(__file__).parent.joinpath("output", "lancedb")
            lancedb_dir.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(lancedb_dir))
            
            # ✅ 表名明确标识使用 BGE 模型
            table_name = "financial_reports_bge"

            # ✅ 从文件名中提取元数据（股票代码_年份_报告类型）
            # 示例文件名：002927_泰永长征_2024年三季度报告.pdf
            name_parts = re.findall(r'\d+', md_name)
            s_code = name_parts[0] if len(name_parts) > 0 else "unknown"
            s_year = name_parts[1] if len(name_parts) > 1 else "unknown"
            
            # ✅ 统一报告类型命名：一季度报、半年报、三季度报、年报
            if "半年度" in md_name or "半年度报" in md_name:
                r_type = "半年度报"
            elif "年度报告" in md_name or "年报" in md_name:
                r_type = "年报"
            elif "一季度" in md_name or "第一季度" in md_name:
                r_type = "一季度报"
            elif "三季度" in md_name or "第三季度" in md_name:
                r_type = "三季度报"
            else:
                r_type = "unknown"
            
            print(f"📊 向量化入库：{s_code}_{s_year}_{r_type}")

            # ✅ 使用 BGE-small，max_tokens=512 完全够用
            # 显式指定 tokenizer，避免默认 tokenizer 的误报警告
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
                max_tokens=512,
            )
            chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
            )

            chunks = list(chunker.chunk(result.document))

            # ✅ 辅助函数：智能截断长文本（可选，如果警告仍然出现）
            def smart_truncate(text: str, max_tokens: int = 512) -> str:
                """
                智能截断文本到指定 token 数
                如果文本不超长，直接返回；否则截断到 max_tokens
                """
                tokenizer_local = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
                tokens = tokenizer_local.encode(text, add_special_tokens=False)
                
                if len(tokens) <= max_tokens:
                    return text  # 不需要截断
                
                # 截断并解码
                truncated_tokens = tokens[:max_tokens]
                return tokenizer_local.decode(truncated_tokens, skip_special_tokens=True)

            data = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunker.contextualize(chunk)
                
                # ✅ 可选：如果还是有警告，取消下面的注释
                # chunk_text = smart_truncate(chunk_text, max_tokens=480)  # 留一点余量
                
                page_numbers = list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov))

                # 核心优化：增加精准过滤字段
                data.append({
                    "text": chunk_text,
                    "vector": embedding_func.compute_source_embeddings([chunk_text])[0],  # 手动计算嵌入
                    "source": str(md_abs_path),
                    "filename": md_name,
                    "stock_code": s_code,
                    "annual": s_year,
                    "report_type": r_type,
                    "chunk_index": i,
                    "pages": str(page_numbers),
                    "section": " > ".join(chunk.meta.headings) if chunk.meta.headings else "正文"
                })

            # Note: list_tables() 返回 ListTablesResponse 对象，需要访问 .tables 属性
            table_names = db.list_tables().tables if hasattr(db.list_tables(), 'tables') else []
            
            if table_name in table_names:
                table = db.open_table(table_name)
                
                # ✅ 智能去重：检查是否已存在该报告数据
                try:
                    existing = table.search() \
                        .where(f"stock_code = '{s_code}'") \
                        .where(f"annual = '{s_year}'") \
                        .where(f"report_type = '{r_type}'") \
                        .limit(1) \
                        .to_list()
                    
                    if existing:
                        # 存在旧数据，先删除
                        print(f"⚠️ 发现已存在数据：{s_code}_{s_year}_{r_type}，删除旧数据...")
                        table.delete(f"stock_code = '{s_code}' AND annual = '{s_year}' AND report_type = '{r_type}'")
                        print(f"✅ 追加新数据：{s_code}_{s_year}_{r_type} ({len(data)} chunks)")
                        table.add(data)
                    else:
                        # 不存在，直接追加
                        print(f"✅ 追加新数据：{s_code}_{s_year}_{r_type} ({len(data)} chunks)")
                        table.add(data)
                except Exception as e:
                    # 查询失败（可能是表结构变化），直接追加
                    print(f"⚠️ 查询失败，直接追加数据：{e}")
                    table.add(data)
            else:
                # 首次创建表
                print(f"✅ 创建新表：{table_name}")
                db.create_table(table_name, schema=FinancialReports, data=data)
            
            return len(data)


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