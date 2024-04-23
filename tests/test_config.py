# -*- coding: utf-8 -*-
import unittest
import os
from image_search.config.config_util import ConfigManager


class TestConfigUtil(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 在测试类开始前加载配置文件
        config_file_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
        ConfigManager.load_config(config_file_path)

    def test_get_kafka_config(self):
        kafka_config = ConfigManager.get_kafka_config()
        self.assertIsInstance(kafka_config, dict)
        self.assertEqual(kafka_config.get('brokers'), 'domain1:9092')

    def test_get_milvus_config(self):
        milvus_config = ConfigManager.get_milvus_config()
        self.assertIsInstance(milvus_config, dict)
        self.assertEqual(milvus_config.get('host'), 'test_host')
        self.assertEqual(milvus_config.get('port'), 5678)
        self.assertEqual(milvus_config.get('user'), 'abc123')
        self.assertEqual(milvus_config.get('password'), 'abc1234')

    def test_get_minio_config(self):
        minio_config = ConfigManager.get_minio_config()
        self.assertIsInstance(minio_config, dict)
        self.assertEqual(minio_config.get('endpoint'), 'http://test_minio.example.com')
        self.assertEqual(minio_config.get('access_key'), 'minio_abc1234')
        self.assertEqual(minio_config.get('secret_key'), 'minio_abc12345')


if __name__ == '__main__':
    unittest.main()
