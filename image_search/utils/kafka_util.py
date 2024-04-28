# -*- coding: utf-8 -*-
from image_search.config.logging_config import logger
# 消费消息的效率排名是：confluent-kafka > kafka-python > pykafka
from confluent_kafka import Consumer, KafkaException, KafkaError
import json

class KafkaConsumerImgUrl:
    def __init__(self, topics, bootstrap_servers, image_fetcher, milvus_client, image_emb, group_id='bdc.product_img_vector',
                 auto_offset_reset='earliest'):
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
        # {
        #   "ec_picture_id": 21353783,
        #   "product_no": "20240319000230",
        #   "brand_no": "ST",
        #   "picture_url": "/staccato/2024/20240319000230/20240319000230_02_l.jpg"
        # }
        picture_id = msg['picture_id']
        picture_url = msg['picture_url']
        product_no = msg['product_no']
        brand_no = msg['brand_no']
        picture_source = msg['picture_source']

        # 通过获取的数据，进行判断需要去哪个桶中获取数据，进行写入
        if picture_source == 'ecommerce_pro_picture':
            # 数据来自大拇指
            bucket_name = 'ods-cdm-image'
        elif picture_source == 'pro_picture':
            bucket_name = 'ods-ps'

        img_data = self.image_fetcher.fetch_image(bucket_name,picture_url)

        if img_data is None:
            logger.error(f'当前照片获取失败: {msg}')
        else:
            img_emb = self.image_emb.image_to_netvector(image=img_data)
            self.milvus_client.upsert_data(picture_id = picture_id,product_no=product_no, brand_no=brand_no, img_emb=img_emb,
                                           picture_url=picture_url,picture_source=picture_source)
            # ec_picture_id = ec_picture_id,