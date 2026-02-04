


import time, asyncio, os, re
from pathlib import Path
import aiofiles
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# 导入必要的配置类
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
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


# 替换为你本地的一份财报路径
source = r"D:\Temp\mycode\CorpIntel\output\pdf\(002927)泰永长征：2025年三季度报告.pdf"

save_path = r'D:\Temp\mycode\CorpIntel\output\md\(002927)泰永长征：2025年三季度报告.md'

async def test():

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
    
    result = converter.convert(source)

    print("PDF TO MD 开始时间：", time.time())

    markdown_content = result.document.export_to_markdown()
    print("PDF TO MD 结束时间：", time.time())
    # async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
    #         await f.write(markdown_content)
    #         await f.close()
    
    # 3. 向量化入库 (耗时操作)
    print("向量化入库时间：", time.time())

    md_name = Path(save_path).name

    # ✅ 统一保存到 output/lancedb 目录
    lancedb_dir = Path(__file__).parent.joinpath("output")
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_dir))
    table_name = "financial_reports_bge"

    # BGE-small 嵌入模型类（轻量高效，512 tokens 完全够用）
    class BGEEmbeddingFunction:
        def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
            print(f"{'='*60}")
            print(f"正在加载模型: {model_name}")
            print(f"【模型说明】BGE-small 轻量高效，max_seq_length=512")
            print(f"{'='*60}")
            
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            
            # ✅ 核心配置：设置最大序列长度为 512
            # 这样 encode 时会自动截断，不会报警告
            self.model.max_seq_length = 512
            
            print(f"【模型配置】")
            print(f"  model.max_seq_length: {self.model.max_seq_length}")
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                print(f"  tokenizer.model_max_length: {self.model.tokenizer.model_max_length}")
            print(f"{'='*60}")
            
        def ndims(self):
            return self.model.get_sentence_embedding_dimension()
        
        def compute_source_embeddings(self, texts):
            """计算文本嵌入 - 超过 512 tokens 会自动截断"""
            # 注意：max_seq_length=512 已在 __init__ 中设置，这里会自动截断
            return self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1
            )
        
        def compute_query_embeddings(self, query):
            """计算查询嵌入 - 超过 512 tokens 会自动截断"""
            return self.model.encode(
                [query], 
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]

    # 初始化嵌入函数（改用 BGE-small）
    EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"
    embedding_func = BGEEmbeddingFunction(EMBED_MODEL_ID)

    class FinancialReportsBGE(LanceModel):
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

    # ✅ 从文件名中提取元数据（股票代码_年份_报告类型）
    # 示例文件名：002927_泰永长征_2025年三季度报告.pdf
    name_parts = re.findall(r'\d+', md_name)

    s_code = name_parts[0] if len(name_parts) > 0 else "unknown"
    s_year = name_parts[1] if len(name_parts) > 1 else "unknown"
    
    # ✅ 统一报告类型命名：一季度报、半年报、三季度报、年报
    if "半年度" in md_name or "半年报" in md_name:
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
    print("初始化 HybridChunker...")
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

        # ✅ 精简元数据：支持多股票、多年份、多报告类型查询
        data.append({
            "text": chunk_text,
            "vector": embedding_func.compute_source_embeddings([chunk_text])[0],
            "source": save_path,
            "filename": md_name,
            "stock_code": s_code,      # 精准过滤：WHERE stock_code = '002927'
            "annual": s_year,          # 精准过滤：WHERE annual = '2024'
            "report_type": r_type,     # 精准过滤：WHERE report_type = '三季度报'
            "chunk_index": i,
            "pages": str(page_numbers),
            "section": " > ".join(chunk.meta.headings) if chunk.meta.headings else "正文"
        })

    # ✅ 智能合并策略：检查是否已存在同一份报告的数据
    # Note: list_tables() 返回 ListTablesResponse 对象，需要访问 .tables 属性获取表名列表
    table_names = db.list_tables().tables if hasattr(db.list_tables(), 'tables') else []
    
    if table_name in table_names:
        table = db.open_table(table_name)
        
        # 检查是否已存在该报告（通过 stock_code + annual + report_type 唯一标识）
        try:
            existing = table.search() \
                .where(f"stock_code = '{s_code}'") \
                .where(f"annual = '{s_year}'") \
                .where(f"report_type = '{r_type}'") \
                .limit(1) \
                .to_list()
            
            if existing:
                print(f"⚠️ 发现已存在数据：{s_code}_{s_year}_{r_type}，追加新数据...")
                table.add(data)
            else:
                print(f"✅ 追加新数据：{s_code}_{s_year}_{r_type} ({len(data)} chunks)")
                table.add(data)
        except Exception as e:
            # 查询失败（可能是表结构变化），直接追加
            print(f"⚠️ 查询失败，直接追加数据：{e}")
            table.add(data)
    else:
        # 首次创建表
        print(f"✅ 创建新表：{table_name}")
        db.create_table(table_name, schema=FinancialReportsBGE, data=data)

    print("向量化入库结束时间：", time.time())
    print(len(data))



if __name__ == "__main__":
    asyncio.run(test())