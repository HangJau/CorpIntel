import asyncio
import os
import uuid
from pathlib import Path
from typing import Literal

import httpx
from cachetools import TTLCache
from fastmcp import FastMCP
import lancedb
from sentence_transformers import SentenceTransformer


from pdfTomd import process_pdf_to_vector_task
from sse import SSE
from szse import SZSE


# 统一获取绝对路径
BASE_DIR = Path(__file__).parent.absolute()
DEFAULT_OUTPUT_DIR = str(BASE_DIR.joinpath("output"))

# 全局任务注册表：最多存储 1000 个任务，每个任务存活 1 小时 (3600秒)
# 这样即使 AI 忘了查，内存也会自动回收
task_status_center = TTLCache(maxsize=1000, ttl=3600)

# 全局 Embedding 模型（用于查询时向量化）
EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"
_embed_model = None

def get_embed_model():
    """懒加载 Embedding 模型（仅在查询时初始化）"""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)
        _embed_model.max_seq_length = 512
    return _embed_model

corp_intel_mcp = FastMCP("CorpIntel")
sse = SSE()
szse = SZSE()


def get_real_output_dir():
    return os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


@corp_intel_mcp.tool()
def get_now_date():
    """
    获取当前日期，格式为{"status": 1, "date": "2026-02-04", "info": "工作日", "week": "周三", "is_workingday": 1}
    """
    header = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"}

    date_rsp = httpx.get("https://www.iamwawa.cn/workingday/api", headers=header)    
    return date_rsp.json()


@corp_intel_mcp.tool()
def find_report_list(stock_code: str, annual: str, report_type: Literal["年报", "一季度报", "半年度报", "三季度报"],
                        file_format: Literal["md", "pdf"] = "md"):
    """
    获取指定股票的财务报表数据，分为PDF和MD格式，PDF格式需要通过download_financial_report进行下载，MD格式已转换
    :param stock_code: 股票代码
    :param report_type: 报告类型 一季度报，半年度报，三季度报，年报
    :param annual: 年份
    :param file_format: 查询的报告文件类型，选 "md" 或 "pdf"
    :return: {"code": 0, "data": "报告路径"}
    """
    output_dir = get_real_output_dir()
    output_path = Path(output_dir)

    if file_format == "md":
        output_path = output_path.joinpath("md")

    else:
        output_path = output_path.joinpath("pdf")

    report_resp = output_path.glob(f"*{stock_code}*{annual}*{report_type}*.{file_format}")
    report_resp = list(report_resp)

    if not report_resp:
        return {"code": 1,
                "msg": "未找到报告，请检查是否有进行转换或通过get_financial_report_list方法查询报告是否存在并通过download_financial_report进行下载"}

    return {"code": 0, "data": str(report_resp[0])}


@corp_intel_mcp.tool()
def get_financial_report_list(stock_code: str, annual: str,
                              report_type: Literal["年报", "一季度报", "半年度报", "三季度报", "全部"]):
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
    output_dir = get_real_output_dir()

    output_path = Path(output_dir).joinpath("pdf")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if "sse" in url:
        return await sse.download_pdf(url, stock_code, title, str(output_path))

    elif "szse" in url:
        return await szse.download_pdf(url, stock_code, title, str(output_path))

    else:
        return {"code": 1, "data": "下载失败，请重试"}


@corp_intel_mcp.tool()
async def pdf_to_md_async(pdf_path: str) -> dict:
    """
    【耗时任务】启动异步解析流程，将 PDF 财报转为 Markdown 并存入向量数据库。

    逻辑：
    1. 立即返回 task_id。
    2. 后台执行：Docling解析 -> 语义切片 -> BGE向量化 -> LanceDB入库。


    注意：
    - 处理过程通常需要20分钟不等，根据当前设备配置大小需要更长时间。
    - 调用后，你必须使用 check_task_status(task_id) 轮询进度。
    - 在状态变为 'completed' 之前，query_financial_knowledge 将无法查到该报告数据。
    """
    if not Path(pdf_path).exists():
        return {"code": 1, "msg": "文件路径不存在"}

    task_id = f"task_{uuid.uuid4().hex[:8]}"

    # TTLCache 的使用方式和普通 dict 完全一致
    task_status_center[task_id] = {
        "status": "running",
        "progress": "Task initialized",
        "result": None
    }
    output_dir = get_real_output_dir()
    _ = asyncio.create_task(process_pdf_to_vector_task(task_id, pdf_path, task_status_center, output_dir))

    return {
        "code": 0,
        "task_id": task_id,
        "msg": "PDF处理任务已在后台启动。请使用 check_task_status 轮询进度，处理完成后结果将保留 1 小时。"
    }


@corp_intel_mcp.tool()
async def check_task_status(task_id: str) -> dict:
    """
    查询异步解析任务的状态。

    返回值说明：
    - status='running': 正在处理，请间隔 20 秒后再次查询。
    - status='completed': 处理成功，现在可以调用 query_financial_knowledge 进行检索。
    - status='failed': 处理失败，请检查 result 字段中的错误原因。
    """
    status = task_status_center.get(task_id)
    if not status:
        return {"code": 1, "msg": "无效的 Task ID 或任务已过期（任务结果仅保留 1 小时）"}

    # 获取当前状态
    current_status = status["status"]

    response = {
        "status": current_status,
        "progress": status.get("progress"),
        "result": status.get("result")
    }

    # 提示信息：告知用户结果保留时长
    if current_status in ["completed", "failed"]:
        response["msg"] = "任务已结束。结果将在 1 小时后自动从缓存中清除。"

    return response


@corp_intel_mcp.tool()
async def query_financial_knowledge(question: str, stock_code: str = None, annual: str = None, report_type: str = None,
                                    n_results: int = 5):
    """
    在向量数据库中进行语义搜索，回答有关财报的具体问题。

    参数建议：
    - 务必提供 stock_code (如 '600519') 和 annual (如 '2023') 以进行精准过滤。

    工作流引导：
    - 如果返回 "【数据缺失】"，表示 PDF 存在但未入库，请调用 pdf_to_md_async。
    - 如果返回 "【未找到报告】"，表示本地无文件，请调用 download_financial_report。
    """
    output_dir = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.joinpath("output")))

    # 2. 数据库连接（必须与 pdfTomd.py 一致）
    output_dir = get_real_output_dir()
    lancedb_dir = Path(output_dir).joinpath("lancedb")

    if not lancedb_dir.exists():
        return {"code": 1, "msg": "数据库目录不存在，请先转换文档。"}

    db = lancedb.connect(str(lancedb_dir))
    table_name = "financial_reports_bge"  # 必须与 pdfTomd.py 的表名一致
    
    # Note: list_tables() 返回 ListTablesResponse 对象，需要访问 .tables 属性
    table_names = db.list_tables().tables if hasattr(db.list_tables(), 'tables') else []
    
    if table_name not in table_names:
        return {"code": 1, "msg": "向量数据库尚未初始化，请先调用 pdf_to_md_async 处理文档。"}

    table = db.open_table(table_name)

    # --- 构建过滤条件 (SQL 语法) ---
    filters = []
    if stock_code: filters.append(f"stock_code = '{stock_code}'")
    if annual: filters.append(f"annual = '{annual}'")
    if report_type: filters.append(f"report_type = '{report_type}'")

    where_clause = " AND ".join(filters) if filters else None

    # 3. 将问题转换为向量（使用全局模型）
    model = get_embed_model()
    question_vector = model.encode([question], convert_to_numpy=True)[0]
    
    # 4. 执行向量搜索
    query = table.search(question_vector).limit(n_results)
    if where_clause:
        query = query.where(where_clause)
    
    results = query.to_list()

    # 5. 拼装结果
    if not results:
        return {"code": 1, "msg": "未检索到相关内容。可能是由于过滤条件过严或文档尚未入库。"}

    context_list = []
    for row in results:
        source = row.get('filename', '未知来源')
        section = row.get('section', '正文')
        pages = row.get('pages', '-')
        doc = row.get('text', '')

        entry = f"--- 来源: {source} (章节: {section}, 页码: {pages}) ---\n{doc}\n"
        context_list.append(entry)

    return "\n".join(context_list)



if __name__ == '__main__':
    corp_intel_mcp.run(transport="stdio")
