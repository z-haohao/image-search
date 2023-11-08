根据项目模块整理，我现在整理出一份完整项目代码，代码如下，请让我分批次输入完整项目代码，进行核验：
1. main程序代码：
    from image_search.config.logging_config import logger
    from image_search.config.config_util import ConfigManager
    from image_search.utils.kafka_util import KafkaConsumer
    from image_search.utils.image_fetcher import ImageFetcher,MinioClient,ImageRequest
    from image_search.utils.milvus_util import MilvusClient
    from image_search.utils.image_to_net_util import Net

    from retry import retry

    @retry(tries=3, delay=2)
    def main():
        # 加载配置
        config_manager = ConfigManager('../config/config.yaml')
        kafka_config = config_manager.get_kafka_config()
        minio_config = config_manager.get_minio_config()
        milvus_config = config_manager.get_milvus_config()
        # 初始化模块
        minio_client = MinioClient(endpoint=minio_config['endpoint'],access_key=minio_config['access_key'],secret_key=minio_config['secret_key'],bucket_name=minio_config['bucket_name'])
        img_req = ImageRequest()
        image_fetcher = ImageFetcher(minio_client, img_req)
        milvus_client = MilvusClient(host=milvus_config['host'],port=milvus_config['port'],user=milvus_config['user'],passwd=milvus_config['password'],database=milvus_config['database'],collection=milvus_config['collection'])
        emb_net = Net()
        # 初始化Kafka消费者
        consumer_client = KafkaConsumer(bootstrap_servers=kafka_config['servers'],topics=kafka_config['topic'], image_fetcher=image_fetcher, milvus_client= milvus_client, image_emb= emb_net)
        # 开始处理消息
        consumer_client.consume_messages()

    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            logger.error(f"Failed to run the main program: {e}", exc_info=True)
2:  Kafka消息队列监听模块（Kafka消费者） 模块代码
    from image_search.config.logging_config import logger
    from confluent_kafka import Consumer, KafkaException, KafkaError
    import json
    class KafkaConsumer:
        def __init__(self, topics, bootstrap_servers, image_fetcher, milvus_client,image_emb, group_id='image-search', auto_offset_reset='earliest'):
            self.consumer = Consumer({
                'bootstrap.servers': bootstrap_servers,
                'group.id': group_id,
                'auto.offset.reset': auto_offset_reset,
                'enable.auto.commit': True
            })
            self.topics = [topics]
            self.image_fetcher = image_fetcher
            self.milvus_client = milvus_client
            self.image_emb = image_emb
            logger.info("KafkaConsumer initialized.")

        def consume_messages(self):
            try:
                self.consumer.subscribe(self.topics)

                while True:
                    msg = self.consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition event
                            logger.info('%% %s [%d] reached end at offset %d\n' %
                                         (msg.topic(), msg.partition(), msg.offset()))
                        elif msg.error():
                            raise KafkaException(msg.error())
                    else:
                        self.process_message(msg)

            except KafkaException as e:
                logger.error(f"Kafka error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            finally:
                self.consumer.close()

        def process_message(self, msg):
            # 在这里处理消息

            msg_value = msg.value().decode('utf-8')
            msg = json.loads(msg_value)
            picture_url = msg['picture_url']
            product_no = msg['product_no']
            brand_no = msg['brand_no']
            img_data = self.image_fetcher.fetch_image(picture_url)

            if img_data is None :
                logger.error(f'当前照片获取失败: {msg}')
            else:
                img_emb = self.image_emb.image_to_netvector(image=img_data)
                self.milvus_client.upsert_data(product_no=product_no,brand_no=brand_no,img_emb=img_emb,picture_url=picture_url)





3. 图片获取模块
    from image_search.config.logging_config import logger
    from io import BytesIO
    from minio import Minio
    from minio.error import S3Error
    from retry import retry
    from image_search.config.logging_config import logger
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    class MinioClient:
        def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=False):
            # 初始化minio客户端
            self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
            self.bucket_name = bucket_name
            logger.info("Minio client 初始化完成")

        @retry(S3Error, tries=3, delay=2, backoff=2)
        def get_image(self, object_name):
            # 从Minio服务器获取对象
            object_name = 'bi-mdm' + object_name
            try:
                response = self.client.get_object(self.bucket_name, object_name)
                logger.info(f"Successfully retrieved {object_name} from bucket {self.bucket_name}")
                return response.data
            except S3Error as err:
                logger.error(f"An error occurred while trying to get object {object_name}: {err}")
                return None
            except Exception as err:
                logger.error(f"Unexpected error: {err}")
                return None


    class ImageRequest:
        SERVER_PREFIX = "https://pic.belle.net.cn/"
        def __init__(self):
            # 设置重试策略
            self.session = requests.Session()
            retries = Retry(total=5,
                            backoff_factor=0.1,
                            status_forcelist=[500, 502, 503, 504])
            self.session.mount('https://', HTTPAdapter(max_retries=retries))
            logger.info("request 初始化完成")

        def request_image(self, image_suffix):
            """
            根据图片URL的后缀获取图片数据。
            :param image_suffix: 图片URL的后缀
            :return: 一个图片转换为NumPy数组，如果有异常则返回None
            """
            image_url = f"{self.SERVER_PREFIX}{image_suffix}"
            try:
                response = self.session.get(image_url)
                response.raise_for_status()  # 如果响应状态码不是200，将引发HTTPError异常
                img_data = response.content
                logger.info(f"图片 {image_url} 获取成功。")
                return img_data
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTPS错误: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"连接错误: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"请求超时: {timeout_err}")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"请求错误: {req_err}")
            except Exception as e:
                logger.error(f"在获取图片时发生未知错误: {e}")
            return None


    class ImageFetcher:
        def __init__(self, minio_client,img_req):
            self.minio_client = minio_client
            self.img_req = img_req

        def fetch_image(self,  image_path):
            # 从MinIO中获取图片
            image = self.minio_client.get_image( image_path)
            if image:
                return image
            else:
                # 如果MinIO中没有找到图片，从备用服务器获取
                image = self.img_req.request_image(image_path)
                # image = None
                return image

4. 向量存储模块（Milvus客户端）
    from image_search.config.logging_config import logger
    from pymilvus import (
        connections, utility, FieldSchema, CollectionSchema,
        DataType, Collection, db
    )

    from retry import retry
    class MilvusClient:
        def __init__(self, host='localhost', port='19530', user='', passwd='',database='', collection=''):
            connections.connect(user=user, password=passwd, host=host, port=port)
            db.using_database(database)
            # 建立到 Milvus 服务器的连接
            self.collection_name = collection
            self.product_no = 'product_no'
            self.brand_no = 'brand_no'
            self.img_emb = 'img_emb'
            self.picture_url = 'picture_url'
            logger.info(f"当前milvus连接成功: {host} - database: {database} -- collection: {collection}")
            # 创建集合和分区，如果它们不存在
            if not utility.has_collection(collection):
                logger.info(f'当前collection : {collection} 尚未创建。')
                self.create_collection(collection)

        def create_collection(self, collection_name):
            # 定义集合的 schema
            fields = [
                FieldSchema(name=self.product_no, dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name=self.brand_no, dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name=self.img_emb, dtype=DataType.FLOAT_VECTOR, dim=2048),
                FieldSchema(name=self.picture_url, dtype=DataType.VARCHAR, max_length=256)
            ]
            schema = CollectionSchema(fields, auto_id=False,
                                      description="Commodity Systems Department, Image Vectorization")
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            collection = Collection(name=collection_name, schema=schema)  # Get an existing collection.

            #  创建索引
            collection.create_index(
                field_name=self.img_emb,
                index_params=index_params,
                index_name = "photo_vec_index"
            )
            collection.load()
            logger.info(f"Collection {collection.name} is created.")


        @retry(Exception, tries=3, delay=2, backoff=2)
        def upsert_data(self, product_no, brand_no, img_emb, picture_url):
            # 插入数据
            try:
                collection = Collection(self.collection_name)
                # 构建数据
                entities = [
                    [product_no],
                    [brand_no],
                    [img_emb],
                    [picture_url]
                ]

                if not collection.has_partition(brand_no):
                    collection.create_partition(brand_no)

                # 插入操作
                result = collection.upsert(entities)
                # 刷新集合以保证数据已写入
                collection.load()
                return result.primary_keys
            except Exception as e:
                logger.error(f"Error while upserting data with ID {id}: {e}")
                raise
5. 错误处理和日志记录模块
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
