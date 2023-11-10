# import os
# import sys
# # 获取根目录
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # 将根目录添加到path中
# sys.path.append(BASE_DIR)

from image_search.config.logging_config import logger
from image_search.config.config_util import ConfigManager
from image_search.utils.kafka_util import KafkaConsumer
from image_search.utils.image_fetcher import ImageFetcher, MinioClient, ImageRequest
from image_search.utils.milvus_util import MilvusClient
from image_search.utils.image_to_net_util import Net

from retry import retry


@retry(tries=3, delay=2)
def main():
    # 加载配置
    config_manager = ConfigManager('../config/config.yaml')
    logger.info('获取配置成功')
    kafka_config = config_manager.get_kafka_config()
    minio_config = config_manager.get_minio_config()
    milvus_config = config_manager.get_milvus_config()
    # 初始化模块
    minio_client = MinioClient(endpoint=minio_config['endpoint'], access_key=minio_config['access_key'],
                               secret_key=minio_config['secret_key'], bucket_name=minio_config['bucket_name'])
    img_req = ImageRequest()
    image_fetcher = ImageFetcher(minio_client, img_req)
    milvus_client = MilvusClient(host=milvus_config['host'], port=milvus_config['port'], user=milvus_config['user'],
                                 passwd=milvus_config['password'], database=milvus_config['database'],
                                 collection=milvus_config['collection'])
    emb_net = Net()
    # 初始化Kafka消费者
    consumer_client = KafkaConsumer(bootstrap_servers=kafka_config['servers'], topics=kafka_config['topic'],
                                    image_fetcher=image_fetcher, milvus_client=milvus_client, image_emb=emb_net)
    # 开始处理消息
    consumer_client.consume_messages()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to run the main program: {e}", exc_info=True)
