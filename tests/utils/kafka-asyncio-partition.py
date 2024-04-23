import time
from threading import Thread
from datetime import datetime
from confluent_kafka import Consumer


class ChildThread(Thread):
    def __init__(self, name, broker, topic):
        Thread.__init__(self, name=name)
        self.con = KafkaConsumerTool(broker, topic)

    def run(self):
        self.con.receive_msg()


class KafkaConsumerTool:
    def __init__(self, broker, topic):
        config ={
            'bootstrap.servers': broker,
            'group.id': 'test11',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        }
        self.consumer = Consumer(config)
        self.topic = topic

    def receive_msg(self):
        self.consumer.subscribe([self.topic])
        print(datetime.now())
        while True:
            msg = self.consumer.poll(timeout=30.0)
            print(msg.value())


if __name__ == '__main__':
    thread_num = 3
    broker = 'szsjhl-damai-mysql1-test-10-10-223-16-belle.lan:9092,szsjhl-damai-mysql-test-10-10-223-19-belle.lan:9092'
    topic = 'bi_mdm_product_picture'
    threads = [ChildThread("thread_" + str(i + 1), broker, topic) for i in range(thread_num)]

    for i in range(thread_num):
        threads[i].setDaemon(True)
    for i in range(thread_num):
        threads[i].start()
