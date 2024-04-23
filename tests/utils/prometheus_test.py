import threading
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway




# 记录程序启动的时间
start_time = time.time()

def send_heartbeat():
    registry = CollectorRegistry()
    g = Gauge('image_search_uptime', 'Heartbeat duration', registry=registry)
    # 计算运行时间
    run_time = time.time() - start_time
    print(int(run_time))
    g.set(int(run_time))

    push_to_gateway('http://10.251.35.156:19091', job='image_search_job', registry=registry)

    print("推送指标一次")
    heartbeat_timer = threading.Timer(15, send_heartbeat)
    heartbeat_timer.daemon = True
    heartbeat_timer.start()



if __name__ == '__main__':
    # 启动守护线程
    heartbeat_timer = threading.Timer(15, send_heartbeat)
    heartbeat_timer.daemon = True
    heartbeat_timer.start()

    n  = 1
    while(True):
        print('aa--aa')
        time.sleep(1)
        n = n+1
        if (n > 100):
            aa






