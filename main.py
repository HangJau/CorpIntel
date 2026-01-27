# from enum import Enum
from typing import Literal
from fastmcp import FastMCP
from sse import SSE
from szse import SZSE

#
# class REPORT_TYPE(str, Enum):
#     YEARLY = "年报"
#     QUATER1 = "第一季度报"
#     QUATER2 = "半年报"
#     QUATER3 = "第三季度报"


corp_intel_mcp = FastMCP("CorpIntel")
sse = SSE()
szse = SZSE()



@corp_intel_mcp.tool()
def get_financial_report_list(stock_code: str, report_type: Literal["年报", "第一季度报", "半年报", "第三季度报"], annual: str):
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
async def download_financial_report(url: str, title: str, path: str):
    """
    下载财务报表到指定的目录
    :param url: 下载地址
    :param title: 报告名称
    :param path: 保存路径
    :return: {"code": 0, "data": f"{pdf_name}.pdf Save Success. save path {path}"}
    """
    if "sse" in url:
        return await sse.download_pdf(url, title, path)

    elif "szse" in url:
        return await szse.download_pdf(url, title, path)

    else:
        return {"code": 1, "data": "download error"}


if __name__ == '__main__':
    corp_intel_mcp.run(transport="stdio")
