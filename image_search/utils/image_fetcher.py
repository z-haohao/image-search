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
        # bucket_name 通过数据中的来源进行判断
        # self.bucket_name = bucket_name
        logger.info("Minio client 初始化完成")

    @retry(S3Error, tries=3, delay=2, backoff=2)
    def get_image(self,bucket_name, object_name):
        # 从Minio服务器获取对象
        if '?' in object_name:
            # 去除服务器对象访问中有? 的连接地址
            object_name = object_name.split('?')[0]
        # object_name = 'pics' + object_name
        try:
            # 通过参数的方式将bucket_name进行传递进来
            response = self.client.get_object(bucket_name, object_name)
            return response.data
        except S3Error as err:
            logger.error(f"An error occurred while trying to get object {object_name}: {err}")
            return None
        except Exception as err:
            logger.error(f"Unexpected error: {err} -- {object_name}")
            return None

class ImageRequest:
    # SERVER_PREFIX = "https://retailp2.bellecdn.com/"
    def __init__(self, SERVER_PREFIX="https://retailp2.bellecdn.com/"):
        # 设置重试策略
        self.session = requests.Session()
        self.SERVER_PREFIX = SERVER_PREFIX
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
            logger.error(f"在获取图片时发生未知错误: {e},{image_url}")
        return None


class ImageFetcher:
    def __init__(self, minio_client, img_req , img_req_http):
        self.minio_client = minio_client
        self.img_req = img_req
        self.img_req_http = img_req_http
    def fetch_image(self, bucket_name , image_path):
        if bucket_name == 'ods-cdm-image':
            image_path = 'pics' + image_path
        elif bucket_name == 'ods-ps':
            image_path = 'bi-mdm' + image_path
        # 从MinIO中获取图片
        image = self.minio_client.get_image(bucket_name, image_path)
        if image:
            return image

        elif bucket_name == 'ods-ps':
            # 如果MinIO中没有找到图片，从备用服务器获取
            image_path = image_path.replace('bi-mdm','')
            image = self.img_req_http.request_image(image_path)
            # image = None
            return image
        elif bucket_name == 'ods-cdm-image':
            # 如果数据中台MinIO中没有找到图片，从技术中台拉去数据
            bucket_name = 'cdm-image'
            image = self.img_req.get_image(bucket_name,image_path)
            # image = None
            return image
