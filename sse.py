# -*- coding: utf-8 -*-
import time
import re
import asyncio
from pathlib import Path

import httpx
import aiofiles


class SSE:
    STATIC_URL = "https://static.sse.com.cn"
    # --- 封装配置数据 ---
    PLATE_MAP = {
        "全部": "0101,120100,020100,020200,120200",
        "主板": "0101",
        "科创板": "120100,020100,020200,120200",
    }

    REPORT_TYPE_MAP = {
        "全部": "ALL",
        "年报": "YEARLY",
        "一季度报": "QUATER1",
        "半年度报": "QUATER2",
        "三季度报": "QUATER3",
    }

    PUBLICATION_TIME_MAP = {
        "年报": {"start_date": "01-01", "end_date": "04-30"},
        "一季度报": {"start_date": "04-01", "end_date": "04-30"},
        "半年度报": {"start_date": "07-01", "end_date": "08-31"},
        "三季度报": {"start_date": "10-01", "end_date": "10-31"}
    }

    def __init__(self):
        self.headers = {
            'Referer': 'https://www.sse.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        }
        self.client = httpx.AsyncClient(headers=self.headers)

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
        if report_type in ("一季度报", "半年度报", "三季度报"):
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
            response = await self.client.get('https://query.sse.com.cn/security/stock/queryCompanyBulletin.do',
                                             params=params)
            response.raise_for_status()
            dict_rsp = response.json()
            report_info_list = dict_rsp.get('result') or []

            final_data = []
            for report_info in report_info_list:
                # 1. 过滤逻辑：只处理包含关键字的报告
                if annual in report_info.get("TITLE", ""):
                    # 2. 提取并构造新字典
                    item = {k: report_info[k] for k in target_keys if k in report_info}

                    # 3. 直接在提取时拼接 URL，减少后续循环
                    if "URL" in item:
                        item["URL"] = f"{self.STATIC_URL}{item['URL']}"

                    final_data.append(item)

            return {"code": 0, "data": final_data}

        except httpx.HTTPStatusError as e:
            return {"code": 1, "data": f"HTTP error occurred: {e}"}

        except Exception as e:
            return {"code": 1, "data": f"An error occurred: {e}"}

    @staticmethod
    def get_acw_sc__v2(arg1):
        """
        Python 还原 acw_sc__v2 加密逻辑
        :param arg1: 网页源代码中提取到的 arg1 变量
        :return: 算出的 cookie 值
        """
        # 映射表 (JS 中的 posList)
        pos_list = [
            15, 35, 29, 24, 33, 16, 1, 38, 10, 9, 19, 31, 40, 27, 22, 23,
            25, 13, 6, 11, 39, 18, 20, 8, 14, 21, 32, 26, 2, 30, 7, 4,
            17, 5, 3, 28, 34, 37, 12, 36
        ]

        # 掩码 (JS 中的 mask)
        mask = "3000176000856006061501533003690027800375"

        # 步骤 1: 字符串重排
        # 初始化一个长度为 40 的列表
        output_list = [''] * len(pos_list)
        for i in range(len(arg1)):
            char = arg1[i]
            # 找到 i+1 在 pos_list 中的索引位置
            for j in range(len(pos_list)):
                if pos_list[j] == i + 1:
                    output_list[j] = char

        arg2 = "".join(output_list)

        # 步骤 2: 十六进制异或
        arg3 = ""
        # 步长为 2 遍历
        for i in range(0, len(arg2), 2):
            if i >= len(mask):
                break

            # 取两个字符转为 16 进制整数
            str_char_val = int(arg2[i:i + 2], 16)
            mask_char_val = int(mask[i:i + 2], 16)

            # 异或并转回 16 进制，补足两位
            xor_val = hex(str_char_val ^ mask_char_val)[2:]
            if len(xor_val) == 1:
                xor_val = '0' + xor_val
            arg3 += xor_val

        return "acw_sc__v2=" + arg3

    async def download_pdf(self, pdf_url, code: str, pdf_name: str, path: str):
        """
        下载财报pdf
        :param pdf_url: 财报URL地址
        :param code: 股票代码
        :param pdf_name: 财报名称
        :param path: 保存财报地址
        :return:
        """
        pdf_resp = await self.client.get(pdf_url)

        if re_result := re.search(r"var arg1='(.+?)'", pdf_resp.text):
            arg1 = re_result.group(1)
            self.set_cookie(arg1)
        file_path = Path(path).joinpath(f"({code})" + pdf_name)
        pdf_resp = await self.client.get(pdf_url)

        async with aiofiles.open(f'{file_path}.pdf', mode='wb') as file:
            await file.write(pdf_resp.content)
            await file.close()

        return {"code": 0, "data": f"{pdf_name}.pdf Save Success. save path {path}"}

    def set_cookie(self, arg):
        """
        设置cookie
        :param arg: 加密值
        :return:
        """
        ck_content = self.get_acw_sc__v2(arg)
        self.client.headers.update({'cookie': ck_content})

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
            # print("\n" + "=" * 50)
            # print("测试3: 查询600036的2025年三季度报")
            # r = await sse.get_corp_lintel_list("600036", "三季度报", "2025")
            # print(f"结果: {r}")

            res = await sse.download_pdf(
                'https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-10-30/600036_20251030_29FE.pdf',
                "600036", "招商银行股份有限公司2025年第三季度报告", "D:/Temp/mycode/CorpIntel/output")
            print(f"结果: {res}")
            # # 测试4: 查询全部报告
            # print("\n" + "=" * 50)
            # print("测试4: 查询600036的2024年全部报告")
            # r = await sse.get_corp_lintel_list("600036", "全部", "2024")
            # print(f"结果: {r}")
            #
            # # 测试5: 查询半年报
            # print("\n" + "=" * 50)
            # print("测试5: 查询600036的2024年半年报")
            # r = await sse.get_corp_lintel_list("600036", "半年报", "2024")
            # print(f"结果: {r}")


    asyncio.run(test())
