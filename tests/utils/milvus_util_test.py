from image_search.utils.milvus_util import  MilvusClient
from image_search.config.logging_config import logger
from image_search.config.config_util import ConfigManager

from image_search.utils.image_fetcher import ImageFetcher,MinioClient,ImageRequest
from image_search.utils.milvus_util import MilvusClient
from image_search.utils.image_to_net_util import Net

config_manager = ConfigManager('../../config/config.yaml')
kafka_config = config_manager.get_kafka_config()
minio_config = config_manager.get_minio_config()
milvus_config = config_manager.get_milvus_config()
milvus_client = MilvusClient(host=milvus_config['host'], port=milvus_config['port'], user=milvus_config['user'],
                             passwd=milvus_config['password'], database=milvus_config['database'],
                             collection=milvus_config['collection'])







# 测试插入数据
try:
    id = 1  # 示例主键 ID
    photo_vector = [0.1] * 2048  # 示例 2048 长度的向量
    partment = "Marketing"  # 示例部门
    photo_url = "test/image.jpg"  # 示例图片 URL
    primary_keys = milvus_client.upsert_data(id,  partment,  photo_vector, photo_url)
    print(f"Upserted data with primary keys: {primary_keys}")
except Exception as e:
    logger.error(f"An error occurred during the test: {e}")
