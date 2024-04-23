#!/usr/bin/python3

import threading
import time


def output():
  print("当前线程！！！！！！！！！！！！")
  time.sleep(1)


for i in range(5):
  t = threading.Thread(target=output())
  t.start()