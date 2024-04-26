# -*- coding: utf-8 -*-
from image_search.config.logging_config import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from minio.error import S3Error
from minio import Minio
from retry import retry
import requests


class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=False):
        # 初始化minio客户端
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = bucket_name
        logger.info("Minio client 初始化完成")

    @retry(S3Error, tries=3, delay=2, backoff=2)
    def get_image(self, object_name):
        # 从Minio服务器获取对象
        if '?' in object_name:
            # 去除服务器对象访问中有? 的连接地址
            object_name = object_name.split('?')[0]
        object_name = 'pics' + object_name
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.data
        except S3Error as err:
            logger.error(f"An error occurred while trying to get object {object_name}: {err}")
            return None
        except Exception as err:
            logger.error(f"Unexpected error: {err} -- {object_name}")
            return None

class ImageRequest:
    # SERVER_PREFIX = "https://retailp2.bellecdn.com/"
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=False):
        # 初始化minio客户端
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = bucket_name
        logger.info("Minio client 初始化完成")

    @retry(S3Error, tries=3, delay=2, backoff=2)
    def get_image(self, object_name):
        # 从Minio服务器获取对象
        if '?' in object_name:
            # 去除服务器对象访问中有? 的连接地址
            object_name = object_name.split('?')[0]
        object_name = 'pics' + object_name
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.data
        except S3Error as err:
            logger.error(f"An error occurred while trying to get object {object_name}: {err}")
            return None
        except Exception as err:
            logger.error(f"Unexpected error: {err} -- {object_name}")
            return None


class ImageFetcher:
    def __init__(self, minio_client, img_req):
        self.minio_client = minio_client
        self.img_req = img_req

    def fetch_image(self, image_path):
        # 从MinIO中获取图片
        image = self.minio_client.get_image(image_path)
        if image:
            return image
        else:
            # 如果MinIO中没有找到图片，从备用服务器获取
            image = self.img_req.get_image(image_path)
            # image = None
            return image
