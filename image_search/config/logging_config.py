# -*- coding: utf-8 -*-
from loguru import logger
import os

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
os.makedirs(log_dir, exist_ok=True)

# 配置日志文件路径
log_file_path = os.path.join(log_dir, "image_search.log")

# 配置 Loguru 的 logger
logger.add(
    log_file_path,
    rotation="00:00",  # 在午夜时切割日志文件
    retention="30 days",  # 保留 30 天的日志
    encoding="utf-8",  # 设置编码为 UTF-8
    level="INFO"  # 日志级别为 INFO
)


# 现在你可以使用 logger 来记录日志
# logger.info("This is an info message")
# logger.error("This is an error message")


# 使用案例：在.py中引入日志对象
# from image_search.config.logging_config import logger
#
# logger.info("This is an info message")
# logger.error("This is an error message")
