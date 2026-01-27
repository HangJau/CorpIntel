# -*- coding: utf-8 -*-
import re
import random
import asyncio
from pathlib import Path

import aiofiles
import httpx


class SZSE:
    DOWNLOAD_URL = "https://disc.static.szse.cn/download"
    # --- 封装配置数据 ---
    PLATE_MAP = {
        "主板": "11",
        "创业板": "16",
    }

    REPORT_TYPE_MAP = {
        "年报": "010301",
        "第一季度报": "010305",
        "半年报": "010303",
        "第三季度报": "010307",
        "全部": ""
    }

    PUBLICATION_TIME_MAP = {
        "年报": {"start_date": "01-01", "end_date": "04-30"},
        "第一季度报": {"start_date": "04-01", "end_date": "04-30"},
        "半年报": {"start_date": "07-01", "end_date": "08-31"},
        "第三季度报": {"start_date": "10-01", "end_date": "10-31"}
    }

    def __init__(self):
        headers = {
            'Host': 'www.szse.cn',
            'origin': 'https://www.szse.cn',
            'Referer': 'https://www.szse.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        }
        self.client = httpx.AsyncClient(base_url='https://www.szse.cn', headers=headers)

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

        data = {
            # "seDate": ["", ""],
            "stock": [stock_code],
            "channelCode": ["fixed_disc"],
            "bigCategoryId": [],
            "pageSize": 50,
            "pageNum": 1
        }

        # 计算日期范围
        if report_type in ("第一季度报", "半年报", "第三季度报"):
            year = annual
            start_date = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['start_date']}"
            end_date = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['end_date']}"

        elif report_type == "年报":
            year = str(int(annual) + 1)
            start_date = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['start_date']}"
            end_date = f"{year}-{self.PUBLICATION_TIME_MAP[report_type]['end_date']}"

        else:
            # 全部：整年度所有报告 + 次年4个月（包含年报披露期）
            start_date = f"{annual}-01-01"
            end_date = f"{int(annual) + 1}-04-30"

        param = {"random": random.random()}
        data["seDate"] = [start_date, end_date]

        # 定义你需要的 key 列表
        # target_keys = ["SECURITY_CODE", "SECURITY_NAME", "TITLE", "URL", "BULLETIN_YEAR", "BULLETIN_TYPE", "SSEDATE"]

        try:
            response = await self.client.post('/api/disc/announcement/annList', params=param, json=data)
            response.raise_for_status()
            dict_rsp = response.json()
            report_info_list = dict_rsp.get('data') or []

            results = []
            if report_info_list:
                for report_info in report_info_list:
                    title = report_info["title"]
                    year = re.search(r'\d{4}', title).group()
                    if str(year) == annual:
                        results.append(
                            {
                                "SECURITY_CODE": report_info['secCode'][0],
                                "SECURITY_NAME": report_info['secName'][0],
                                "TITLE": title,
                                "URL": self.DOWNLOAD_URL + report_info["attachPath"],
                                "BULLETIN_YEAR": year,
                                "BULLETIN_TYPE": report_type,
                                "SZSEDATE": report_info["publishTime"]
                            }
                        )

            return {"code": 0, "data": results}

        except httpx.HTTPStatusError as e:
            return {"code": 1, "data": f"HTTP error occurred: {e}"}

        except Exception as e:
            return {"code": 1, "data": f"An error occurred: {e}"}

    async def download_pdf(self, pdf_url, pdf_name: str, path: str):
        """
        下载财报
        :param pdf_url: 下载地址
        :param pdf_name: 财报名
        :param path: 保存路径
        :return:
        """
        try:
            self.client.headers.update({"Host": "disc.static.szse.cn"})
            pdf_resp = await self.client.get(pdf_url, params={"n": pdf_name + ".pdf"})
            pdf_resp.raise_for_status()
            file_path = Path(path).joinpath(pdf_name)

            # print(pdf_resp.text)

            async with aiofiles.open(f'{file_path}.pdf', mode='wb') as file:
                await file.write(pdf_resp.content)
                await file.close()

            return {"code": 0, "data": f"{pdf_name}.pdf Save Success. save path {path}"}

        except IOError as e:
            return {"code": 1, "data": f"IO error occurred: {e}"}

    def read_corp_lintel_content(self, pdf_url: str) -> dict:
        """
        读取公司定期报告内容
        :param pdf_url: pdf链接
        :return: 公司定期报告内容
        """
        pass


if __name__ == '__main__':
    async def test():
        async with SZSE() as szse:
            # 测试3: 查询年报
            print("\n" + "=" * 50)
            # print("测试3: 查询600036的2023年年报")
            # r = await sse.get_corp_lintel_list("002083", "年报", "2023")
            # print(f"结果: {r}")
            #
            # # 测试4: 查询全部报告
            # print("\n" + "=" * 50)
            # print("测试4: 查询600036的2024年全部报告")
            # r = await sse.get_corp_lintel_list("002083", "全部", "2024")
            # print(f"结果: {r}")
            #
            # # 测试5: 查询半年报
            # print("\n" + "=" * 50)
            # print("测试5: 查询600036的2024年半年报")
            # r = await sse.get_corp_lintel_list("002083", "半年报", "2024")
            # print(f"结果: {r}")

            # r = await szse.get_corp_lintel_list("000524", "第三季度报", "2025")
            # print(f"结果: {r}")
            await szse.download_pdf(
                "https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-10-31/df8c95e9-e004-4c5d-9429-21623efa5259.PDF",
                "岭南控股：2025年三季度报告", "D:\Temp\mycode\CorpIntel")


    asyncio.run(test())
