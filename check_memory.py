import psutil
import time


min_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024

#1分ごとにメモリ確認　最小メモリを記録する
while True:
    time.sleep(60)

    tmp_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
    if min_mem > tmp_mem:
        min_mem = tmp_mem
        print("available min memory: ", min_mem, "GB")