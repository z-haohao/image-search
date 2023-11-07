# -*- coding: utf-8 -*-
import os
import logging.handlers

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
os.makedirs(log_dir, exist_ok=True)

# 创建日志对象
logger = logging.getLogger('image_search')
logger.setLevel(logging.INFO)

# 创建一个每天切分的日志处理器
log_file = os.path.join(log_dir, "image_search.log")
handler = logging.handlers.TimedRotatingFileHandler(
    log_file,
    when='midnight',  # 每天切分
    interval=1,  # 每天
    backupCount=30,  # 最多保留30个旧日志文件
    encoding='utf-8'  # 设置编码为UTF-8
)


# 定义日志格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)

# 添加处理器到日志对象
logger.addHandler(handler)

# 使用案例：在.py中引入日志对象
# from image_search.config.logging_config import logger
#
# logger.info("This is an info message")
# logger.error("This is an error message")
