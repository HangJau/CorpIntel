import os
from pathlib import Path
from typing import Literal
from fastmcp import FastMCP
from docling.document_converter import DocumentConverter
import aiofiles
from sse import SSE
from szse import SZSE


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
def get_financial_report_list(stock_code: str, annual: str, report_type: Literal["年报", "一季度报", "半年度报", "三季度报"]):
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

    return {"code": 0, "msg": f"{md_name} Save Success. save path {md_abs_path}"}


if __name__ == '__main__':
    corp_intel_mcp.run(transport="stdio")
