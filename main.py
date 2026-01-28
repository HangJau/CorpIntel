import os
from pathlib import Path
from typing import Literal
from fastmcp import FastMCP
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

import aiofiles
from sse import SSE
from szse import SZSE
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter





corp_intel_mcp = FastMCP("CorpIntel")
sse = SSE()
szse = SZSE()


@corp_intel_mcp.tool()
def find_md_report_list(stock_code,  annual, report_type: Literal["年报", "一季度报", "半年度报", "三季度报"],file_format: Literal["md", "pdf"] = "md"):
    """
    获取指定股票的财务报表数据，分为PDF和MD格式，PDF格式需要通过download_financial_report进行下载，MD格式已转换
    :param stock_code: 股票代码
    :param report_type: 报告类型 一季度报，半年度报，三季度报，年报
    :param annual: 年份
    :param file_format: 查询的报告文件类型，选 "md" 或 "pdf"
    :return: {"code": 0, "data": "报告路径"}
    """
    output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))
    output_path = Path(output_dir)

    if file_format == "md":
        output_path = output_path.joinpath("md")

    else:
        output_path = output_path.joinpath("pdf")

    report_resp = output_path.glob(f"*{stock_code}*{annual}*{report_type}*.{file_format}")
    report_resp = list(report_resp)

    if not report_resp:
        return {"code": 1, "msg": "未找到报告，请检查是否有进行转换或通过get_financial_report_list方法查询报告是否存在并通过download_financial_report进行下载"}

    return {"code": 0, "data": str(report_resp[0])}

@corp_intel_mcp.tool()
def get_financial_report_list(stock_code: str, annual: str, report_type: Literal["年报", "一季度报", "半年度报", "三季度报", "全部"]):
    """
    获取指定股票的财务报表列表
    :param stock_code: 股票代码
    :param report_type: 报告类型 第一季度报，半年报，第三季度报，年报
    :param annual: 年份
    :return:
    """
    sse_type = ("600", "601", "603", "605", "688")
    szse_type = ("000", "001", "002", "003", "004", "300")
    if stock_code.startswith(sse_type):
        return sse.get_corp_lintel_list(stock_code, report_type, annual)

    elif stock_code.startswith(szse_type):
        return szse.get_corp_lintel_list(stock_code, report_type, annual)

    else:
        return {"code": 1, "msg": "获取信息有误，请检查股票代码是否存在上交所或深交所"}


@corp_intel_mcp.tool()
async def download_financial_report(url: str, stock_code: str, title: str):
    """
    下载财务报表到指定的目录
    :param url: 下载地址
    :param stock_code: 股票代码
    :param title: 报告名称

    :return: {"code": 0, "data": f"{pdf_name}.pdf Save Success. save path {path}"}
    """
    output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))

    output_path = Path(output_dir).joinpath("pdf")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if "sse" in url:
        return await sse.download_pdf(url, stock_code, title, output_dir)

    elif "szse" in url:
        return await szse.download_pdf(url, stock_code, title, output_dir)

    else:
        return {"code": 1, "data": "下载失败，请重试"}


@corp_intel_mcp.tool(task=True)
async def pdf_to_md(pdf_path: str):
    """
    将财务报告pdf转为markdown
    :param pdf_path: pdf路径
    :return: {"code": 0, f"{md_name} Save Success. save path {md_abs_path}"}
    """
    if not Path(pdf_path).exists():
        return {"code": 1, "msg": "文件不存在，请检查文件路径是否正确"}

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # 导出 Markdown
    markdown_content = result.document.export_to_markdown()
    pdf_path = Path(pdf_path)

    md_path = pdf_path.parent.parent.joinpath("md")

    if not md_path.exists():
        md_path.mkdir(parents=True, exist_ok=True)

    md_name = pdf_path.name.replace(".pdf", ".md")

    md_abs_path = md_path.joinpath(md_name)

    # 保存结果
    async with aiofiles.open(md_abs_path, "w", encoding="utf-8") as f:
        await f.write(markdown_content)
        await f.close()

    # 写入chromadb
    output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))
    chroma_client = chromadb.PersistentClient(path=output_dir)
    
    # 显示指定模型为 bge-small-zh-v1.5
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
    collection = chroma_client.get_or_create_collection(name="financial_reports", embedding_function=ef)


    # 使用 docling 原生 HybridChunker 进行语义切片，能更好地处理表格
    chunker = HybridChunker()
    chunk_iter = chunker.chunk(result.document)
    chunks = list(chunk_iter)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunker.serialize(chunk)
        
        # 提取 docling 提供的丰富元数据
        # 比如页码 (prov 记录了来源页)
        page_numbers = list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov))
        # 比如层级标题
        heading_hierarchy = " > ".join(chunk.meta.headings) if chunk.meta.headings else "正文"

        meta = {
            "source": str(md_abs_path),
            "filename": md_name,
            "chunk_index": i,
            "pages": str(page_numbers), # 存入页码，方便 AI 回答“在第几页”
            "section": heading_hierarchy # 存入章节名
        }
        documents.append(chunk_text)
        metadatas.append(meta)
        ids.append(f"{md_name}_chunk_{i}")

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    return {"code": 0, "msg": f"{md_name} Save Success and {len(documents)} semantic chunks added to ChromaDB. save path {md_abs_path}"}


@corp_intel_mcp.tool()
async def query_financial_knowledge(question: str, stock_code: str = None, annual: str = None, report_type: str = None, n_results: int = 5):
    """
    根据问题在已解析的财报库中进行语义检索。
    建议提供 stock_code, annual, report_type 以确保检索目标已入库。
    """
    output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))
    
    # 1. 存在性检查逻辑
    if stock_code and annual and report_type:
        md_dir = Path(output_dir).joinpath("md")
        pdf_dir = Path(output_dir).joinpath("pdf")
        
        # 检查是否已存在 MD (代表已解析)
        md_pattern = f"*{stock_code}*{annual}*{report_type}*.md"
        md_files = list(md_dir.glob(md_pattern)) if md_dir.exists() else []
        
        if not md_files:
            # MD 不存在，检查 PDF 是否存在
            pdf_pattern = f"*{stock_code}*{annual}*{report_type}*.pdf"
            pdf_files = list(pdf_dir.glob(pdf_pattern)) if pdf_dir.exists() else []
            
            if pdf_files:
                return {
                    "code": 1, 
                    "msg": f"【数据缺失】数据库中暂无该财报信息。检测到本地已存在 PDF 文件：{pdf_files[0].name}，请先调用 pdf_to_md(pdf_path='{pdf_files[0]}') 进行解析入库。"
                }
            else:
                return {
                    "code": 1, 
                    "msg": f"【未找到报告】数据库及本地 PDF 库中均未找到该财报（{stock_code}/{annual}/{report_type}）。\n建议操作：\n1. 调用 get_financial_report_list 获取报告链接\n2. 调用 download_financial_report 下载 PDF\n3. 调用 pdf_to_md 解析入库。"
                }

    # 2. 数据库连接与查询
    chroma_client = chromadb.PersistentClient(path=output_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
    collection = chroma_client.get_or_create_collection(name="financial_reports", embedding_function=ef)
    
    # 如果提供了 stock_code，可以增加元数据过滤（如果 pdf_to_md 存了对应元数据的话）
    # 这里先使用通用的语义检索
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    # 3. 拼装返回结果
    context = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context += f"--- 来源: {meta['filename']} (章节: {meta['section']}, 页码: {meta['pages']}) ---\n"
            context += f"{doc}\n\n"
    else:
        return {"code": 1, "msg": "未在数据库中检索到相关内容，请确认财报是否已正确入库。"}
        
    return context





if __name__ == '__main__':
    corp_intel_mcp.run(transport="stdio")
