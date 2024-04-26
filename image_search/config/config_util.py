# -*- coding: utf-8 -*-
import yaml

# 定义全局变量来存储配置信息
config = None



class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_kafka_config(self):
        return self.config['kafka']

    def get_minio_config(self):
        return self.config['minio']

    def get_technology_minio_config(self):
        return self.config['technology_minio']

    def get_milvus_config(self):
        return self.config['milvus']

    def image_server_prefix_config(self):
        return self.config['image_server']




# def load_config(config_file_path):
#     """加载配置文件并设置全局配置"""
#     global config
#     with open(config_file_path, 'r') as yaml_file:
#         config = yaml.load(yaml_file, Loader=yaml.FullLoader)
#
#
# def get_kafka_config():
#     """获取Kafka配置信息"""
#     return config.get('kafka', {})
#
#
# def get_milvus_config():
#     """获取Milvus配置信息"""
#     return config.get('milvus', {})
#
#
# def get_minio_config():
#     """获取Minio配置信息"""
#     return config.get('minio', {})


# if __name__ == '__main__':
#     # 加载配置文件并设置全局配置
#     config_file_path = '../../config/config.yaml'
#     load_config(config_file_path)
#
#     # 示例用法，无需再次加载配置文件
#     kafka_config = get_kafka_config()
#     milvus_config = get_milvus_config()
#     minio_config = get_minio_config()
#
#     print(f'Kafka brokers: {kafka_config.get("brokers")}')
#
#     print(f'Milvus Host: {milvus_config.get("host")}')
#     print(f'Milvus Port: {milvus_config.get("port")}')
#
#     print(f'Minio Endpoint: {minio_config.get("endpoint")}')
#     print(f'Minio Access Key: {minio_config.get("access_key")}')
#     print(f'Minio Secret Key: {minio_config.get("secret_key")}')
