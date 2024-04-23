import time
from datetime import datetime
from confluent_kafka import Consumer
from threadpool import ThreadPool, makeRequests


class KafkaConsumerTool:
    def __init__(self, broker, topic):
        config = {
            'bootstrap.servers': broker,
            'session.timeout.ms': 30000,
            'auto.offset.reset': 'earliest',
            'api.version.request': False,
            'broker.version.fallback': '2.6.0',
            'group.id': 'mini-spider'
        }
        self.consumer = Consumer(config)
        self.topic = topic

    def receive_msg(self, x):
        self.consumer.subscribe([self.topic])
        print(datetime.now())
        while True:
            msg = self.consumer.poll(timeout=30.0)
            print(msg.value())


if __name__ == '__main__':
    thread_num = 10
    broker = 'szsjhl-damai-mysql1-test-10-10-223-16-belle.lan:9092,szsjhl-damai-mysql-test-10-10-223-19-belle.lan:9092'
    topic = 'bi_mdm_product_picture'
    consumer = KafkaConsumerTool(broker, topic)
    pool = ThreadPool(thread_num)
    for r in makeRequests(consumer.receive_msg, [i for i in range(thread_num)]):
        pool.putRequest(r)
    pool.wait()
