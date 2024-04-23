# # 实例化Minio工具类
# # 注意：请用你的实际的endpoint, access_key和secret_key替换下面的字符串
# minio_client = MinioClient("10.251.37.248:7778",
#             "dlink_write",
#             "qzw4Mh3tH4RSYpMo")
#
# # 测试获取对象
# try:
#     bucket_name = "ods-ps"
#     # 对象面前要去掉同名称
#     object_name = "bi-mdm/2023/MDM/AD/FZ5710.jpg"
#     image_data = minio_client.get_object(bucket_name, object_name)
#     # 假设你想要将图像数据保存到文件中，你可以使用下面的代码
#     with open('output_image.jpg', 'wb') as file:
#         file.write(image_data)
#     print("Image data retrieved and written to 'output_image.jpg'")
# except S3Error as e:
#     print(f"MinIO S3 error: {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")


# # 示例使用
# if __name__ == "__main__":
#     fetcher = ImageFetcher()
#     image_suffix = "2021/MDM/TT/A5PPAK01DP1CM1.jpg"  # 实际的后缀路径
#     image_data = fetcher.fetch_image(image_suffix)
#     if image_data is not None:
#         # 可以进一步处理图片数据
#         with open("o.jpg", "wb") as file:
#             file.write(image_data)
#         pass


# https://pic.belle.net.cn//2015/MDM/BL/BICX1198XU2DX1.jpg



# milvus_client = MilvusClient(host='10.10.214.172', port='19530',user='root',passwd='root123',database='zhu_hh',collection='photo_search')
#
# # 测试插入数据
# try:
#     id = 1  # 示例主键 ID
#     photo_vector = [0.1] * 2048  # 示例 2048 长度的向量
#     partment = "Marketing"  # 示例部门
#     photo_url = "http://example.com/image.jpg"  # 示例图片 URL
#     primary_keys = milvus_client.upsert_data(id, photo_vector, partment, photo_url)
#     print(f"Upserted data with primary keys: {primary_keys}")
# except Exception as e:
#     logger.error(f"An error occurred during the test: {e}")


# if __name__ == '__main__':
#     # 实例化Minio工具类
#     # 注意：请用你的实际的endpoint, access_key和secret_key替换下面的字符串
#     minio_client = MinioClient("10.251.37.248:7778",
#                                "dlink_write",
#                                "qzw4Mh3tH4RSYpMo")
#
#     bucket_name = "ods-ps"
#     object_name = "bi-mdm/2023/MDM/AD/FZ5710.jpg"
#     image_data = minio_client.get_object(bucket_name, object_name)
#     # 假设你想要将图像数据保存到文件中，你可以使用下面的代码
#     img_net = Net()
#     img_vectory = img_net.image_to_netvector(image_data)
#     logger.info('图片向量为:' , img_vectory)



# import os
# import sys
# # 获取根目录
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # 将根目录添加到path中
# sys.path.append(BASE_DIR)
#
# from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
# from image_search.config.logging_config import logger
# from image_search.config.config_util import ConfigManager
# from image_search.utils.kafka_util import KafkaConsumerImgUrl
# from image_search.utils.image_fetcher import ImageFetcher, MinioClient, ImageRequest
# from image_search.utils.milvus_util import MilvusClient
# from image_search.utils.image_to_net_util import Net
# from retry import retry
# import threading
# import time
#
# # 记录程序启动的时间
# start_time = time.time()
#
#
# # 加载配置
# config_manager = ConfigManager(r'D:\code\pythonCode\image-search\config\config.yaml')
# logger.info('获取配置成功')
# kafka_config = config_manager.get_kafka_config()
# minio_config = config_manager.get_minio_config()
# milvus_config = config_manager.get_milvus_config()
# # 初始化模块
# minio_client = MinioClient(endpoint=minio_config['endpoint'], access_key=minio_config['access_key'],
#                            secret_key=minio_config['secret_key'], bucket_name=minio_config['bucket_name'])
# img_req = ImageRequest()
# image_fetcher = ImageFetcher(minio_client, img_req)
#
# data = image_fetcher.fetch_image('/retail/2023/mdm/173155/BR/5d6cdf5c845d4561a191b29bc992f7ac.JPG')
# print(data)
