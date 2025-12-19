# -*- coding: utf-8 -*-
import time
import asyncio
import httpx


class SSE:
    # --- 封装配置数据 ---
    PLATE_MAP = {
        "全部": "0101,120100,020100,020200,120200",
        "主板": "0101",
        "科创板": "120100,020100,020200,120200",
    }

    REPORT_TYPE_MAP = {
        "全部": "ALL",
        "年报": "YEARLY",
        "第一季度报": "QUATER1",
        "半年报": "QUATER2",
        "第三季度报": "QUATER3",
    }

    PUBLICATION_TIME_MAP = {
        "年报": {"start_date": "01-01", "end_date": "04-30"},
        "第一季度报": {"start_date": "04-01", "end_date": "04-30"},
        "半年报": {"start_date": "07-01", "end_date": "08-31"},
        "第三季度报": {"start_date": "10-01", "end_date": "10-31"}
    }

    def __init__(self):
        headers = {
            'Referer': 'https://www.sse.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        }
        self.client = httpx.AsyncClient(base_url='https://query.sse.com.cn', headers=headers)

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_corp_lintel_list(self, stock_code: str, report_type: str, annual: str) -> dict:
        """
        获取公司某段时间内定期报告列表
        定期报告：年报、第一季度报、半年报、第三季度报
        :param stock_code: 股票代码
        :param report_type: 报告类型，全部、年报、第一季度报、半年报、第三季度报
        :param annual: 报告年份，格式为4位数字字符串，如"2024"
        :return:
        """
        # 参数校验
        if report_type not in self.REPORT_TYPE_MAP:
            return {"code": 1, "msg": f"无效的报告类型: {report_type}，可选值: {list(self.REPORT_TYPE_MAP.keys())}"}

        if not annual.isdigit() or len(annual) != 4:
            return {"code": 1, "msg": f"无效的年份格式: {annual}，应为4位数字字符串，如'2024'"}

        params = {
            'isPagination': 'true',
            'pageHelp.pageSize': '25',
            'pageHelp.pageNo': '1',
            'pageHelp.beginPage': '1',
            'pageHelp.cacheSize': '1',
            'pageHelp.endPage': '1',
            'productId': stock_code,
            'securityType': "0101,120100,020100,020200,120200",
            'reportType2': 'DQBG',
            'reportType': self.REPORT_TYPE_MAP[report_type],
            '_': int(time.time() * 1000)
        }

        # 计算日期范围
        if report_type in ("第一季度报", "半年报", "第三季度报"):
            year = annual
            params['beginDate'] = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['start_date']}"
            params['endDate'] = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['end_date']}"

        elif report_type == "年报":
            year = str(int(annual) + 1)
            params['beginDate'] = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['start_date']}"
            params['endDate'] = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['end_date']}"

        else:
            # 全部：整年度所有报告 + 次年4个月（包含年报披露期）
            params['beginDate'] = f"{annual}-01-01"
            params['endDate'] = f"{int(annual) + 1}-04-30"

        # 定义你需要的 key 列表
        target_keys = ["SECURITY_CODE", "SECURITY_NAME", "TITLE", "URL", "BULLETIN_YEAR", "BULLETIN_TYPE", "SSEDATE"]

        try:
            response = await self.client.get('/security/stock/queryCompanyBulletin.do', params=params)
            response.raise_for_status()
            dict_rsp = response.json()
            report_info_list = dict_rsp.get('result') or []

            result = []
            for report_info in report_info_list:
                if annual in report_info.get("TITLE"):
                    # 字典推导式：只取需要的 key
                    result.append({k: report_info[k] for k in target_keys if k in report_info})

            return {"code": 0, "data": result}

        except httpx.HTTPStatusError as e:
            return {"code": 1, "data": f"HTTP error occurred: {e}"}

        except Exception as e:
            return {"code": 1, "data": f"An error occurred: {e}"}

    def read_corp_lintel_content(self, pdf_url: str) -> dict:
        """
        读取公司定期报告内容
        :param pdf_url: pdf链接
        :return: 公司定期报告内容
        """
        pass


if __name__ == '__main__':
    async def test():
        async with SSE() as sse:

            # 测试3: 查询年报
            print("\n" + "=" * 50)
            print("测试3: 查询600036的2023年年报")
            r = await sse.get_corp_lintel_list("600036", "年报", "2023")
            print(f"结果: {r}")

            # 测试4: 查询全部报告
            print("\n" + "=" * 50)
            print("测试4: 查询600036的2024年全部报告")
            r = await sse.get_corp_lintel_list("600036", "全部", "2024")
            print(f"结果: {r}")

            # 测试5: 查询半年报
            print("\n" + "=" * 50)
            print("测试5: 查询600036的2024年半年报")
            r = await sse.get_corp_lintel_list("600036", "半年报", "2024")
            print(f"结果: {r}")

    asyncio.run(test())
