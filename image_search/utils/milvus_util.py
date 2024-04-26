# -*- coding: utf-8 -*-
from image_search.config.logging_config import logger
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema,
    DataType, Collection, db
)

from retry import retry
# `mdm_product_no`,`ec_pic_type`,`ec_order_no

class MilvusClient:
    def __init__(self, host='localhost', port='19530', user='', passwd='', database='', collection=''):
        connections.connect(user=user, password=passwd, host=host, port=port)
        db.using_database(database)
        # 建立到 Milvus 服务器的连接
        self.collection_name = collection
        self.picture_id = 'picture_id'
        self.product_no = 'product_no'
        self.brand_no = 'brand_no'
        self.img_emb = 'img_emb'
        self.picture_url = 'picture_url'
        self.picture_source = 'picture_source'
        logger.info(f"当前milvus连接成功: {host} - database: {database} -- collection: {collection}")
        # 创建集合和分区，如果它们不存在
        if not utility.has_collection(collection):
            logger.info(f'当前collection : {collection} 尚未创建。')
            self.create_collection(collection)

    def create_collection(self, collection_name):
        # 定义集合的 schema
        fields = [
            FieldSchema(name=self.picture_id, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=self.product_no, dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name=self.brand_no, dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name=self.img_emb, dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name=self.picture_url, dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name=self.picture_source, dtype=DataType.VARCHAR, max_length=2048)
        ]
        schema = CollectionSchema(fields, auto_id=False,
                                  description="Commodity Systems Department, Image Vectorization")
        index_params = {
            "index_type": "IVF_SQ8",
            "metric_type": "IP",
            "params": {"nlist": 1024}
        }

        collection = Collection(name=collection_name, schema=schema)  # Get an existing collection.

        #  创建索引
        collection.create_index(
            field_name=self.img_emb,
            index_params=index_params,
            index_name="photo_vec_index"
        )
        collection.load()
        logger.info(f"Collection {collection.name} is created.")

    @retry(Exception, tries=3, delay=2, backoff=2)
    def upsert_data(self, picture_id, product_no, brand_no, img_emb, picture_url,picture_source):
        # 插入数据
        try:
            collection = Collection(self.collection_name)
            # 构建数据
            entities = [
                [picture_id],
                [product_no],
                [brand_no],
                [img_emb],
                [picture_url],
                [picture_source]
            ]

            if not collection.has_partition(brand_no):
                logger.info(f"当前分区不存在创建分区{brand_no}")
                collection.create_partition(brand_no)

            # 插入操作
            result = collection.upsert(entities, partition_name=brand_no)
            # 刷新集合以保证数据已写入
            #collection.load()
            return result.primary_keys
        except Exception as e:
            logger.error(f"Error while upserting data with ID {id}: {e}")
            raise
