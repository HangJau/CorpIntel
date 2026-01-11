from fastmcp import FastMCP
from sse import SSE


mcp = FastMCP("CorpIntel")


def get_financial_report(stock_code, report_type, annual):
    """

    :param stock_code: 股票代码
    :param report_type: 报告类型
    :param annual: 年份
    :return:
    """
    # TODO 待补充mcp tools
    if stock_code in []