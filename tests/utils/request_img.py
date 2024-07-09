# -*- coding: utf-8 -*-
__author__ = "zhu.hh"

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

url = 'https://retailp2.bellecdn.com/2015/MDM/HP/CWRH1N21MU1AM3.jpg'

session = requests.Session()
res = session.get(url)
res.raise_for_status()
print(res.content)
if not res.content:  # 检查是否有响应内容
    print(f"图片 {url} 无内容返回。")
else:
    print(f"图片 {url} 获取成功。")


aa = 'bi-mdm/bi-mdm/mdm1.jpg'
print(aa.split('bi-mdm')[1])