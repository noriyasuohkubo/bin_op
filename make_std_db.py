import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import warnings
import math


warnings.simplefilter('error')

"""
make_base_dbで作成したDBデータにもとづいて
closeの標準化を行い、close,timeの値を保存していく
"""

# 処理時間計測
t1 = time.time()

#抽出元のDB名
symbol_org = "GBPJPY_BASE"

#ウィンドウサイズ(DBデータ何本分の平均と標準偏差を求めるか)
std_shifts = (15*30,15*60,15*90,15*120,15*150,15*180,)

close_shift = 1

#新規に作成するDB名
symbol = "GBPJPY"

db_no = 3
host = "127.0.0.1"

start = datetime.datetime(2009, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2020, 1, 1)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
#result_data = redis_db.zrange(symbol_org, 20000000, 30000000, withscores=True)
print("result_data length:" + str(len(result_data)))

close_tmp, time_tmp, score_tmp = [], [], []

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    score_tmp.append(score)
    close_tmp.append(tmps.get("close"))
    time_tmp.append(tmps.get("time"))

# メモリ解放
del result_data
gc.collect()
close_np = np.array(close_tmp)
print("gc end")
print("close_np len:", len(close_np))
print(close_np[:10])

print(time_tmp[0:10])
print(time_tmp[-10:])

#変化率を作成
for i, v in enumerate(close_tmp):
    #変化元(std_shift前のデータ)がないのでとばす
    if i < (max(std_shifts) -1):
        continue
    divide = close_np[i] / close_np[i - close_shift]
    if close_np[i] == close_np[i - close_shift]:
        divide = 1
    divide = 10000 * math.log(divide)
    child = {'close': close_tmp[i],
             'close_divide': divide,
             'time': time_tmp[i]}

    for std_shift in std_shifts:
        try:
            mean = np.mean(close_np[i - std_shift + 1:i + 1])
            std = np.std(close_np[i - std_shift + 1:i + 1])
            if std != 0.0:
                val = (close_tmp[i] - mean) / std
            else:
                val = 0.0
            child['std' + str(std_shift)] = val
        except RuntimeWarning as warn:
            #print("mean", mean)
            #print("std",std)
            #print("time",time_tmp[i])
            print("RuntimeWarning", " score:",score_tmp[i])

    redis_db.zadd(symbol, json.dumps(child), score_tmp[i])

    if i % 10000000 == 0:
        dt_now = datetime.datetime.now()
        print(dt_now, " ", i)



t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))

#手動にて抽出元のDBを削除して永続化すること！
#print("now db saving")
#redis_db.save()
