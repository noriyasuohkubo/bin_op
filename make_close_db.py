import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal

"""
make_base_dbで作成したDBデータにもとづいて
closeの変化率とclose,timeの値を保存していく

※dukasのデータは日本時間の土曜朝7時から月曜朝7時までデータがない
"""


# 処理時間計測
t1 = time.time()

#抽出元のDB名
symbol_org = "GBPJPY_BASE"


close_shift = 1

#新規に作成するDB名
symbol = "GBPJPY_2_0"

#直前の1分足のスコアをデータに含めるか
#例えば12：03：44なら12:02
include_min = False

db_no = 3
host = "127.0.0.1"

start = datetime.datetime(2007, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2020, 1, 2)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

close_tmp, time_tmp, score_tmp = [], [], []

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    score_tmp.append(score)
    close_tmp.append( float( Decimal(str(tmps.get("close"))) * Decimal(str(100)))  ) #dukaのレートは本来のレートの100分の1なのでもとにもどす
    time_tmp.append(tmps.get("time"))

close_np = np.array(close_tmp)
print("gc end")
print("close_tmp len:", len(close_tmp))
print(close_tmp[:10])

print("start:", datetime.datetime.fromtimestamp(score_tmp[0]))
print("end:", datetime.datetime.fromtimestamp(score_tmp[-1]))

# メモリ解放
del result_data, close_tmp
#gc.collect()

#変化率のlogをとる
math_log = False

#変化率を作成
for i, v in enumerate(close_np):
    #変化元(close_shift前のデータ)がないのでとばす
    if i < close_shift:
        continue

    divide = close_np[i] / close_np[i - close_shift]
    if close_np[i] == close_np[i - close_shift]:
        divide = 1

    if math_log:
        divide = 10000 * math.log(divide)
    else:
        divide = 10000 * (divide - 1)

    child = {'c': close_np[i],
            'd': divide,
            't': time_tmp[i],
             }

    """
    if include_min:
        tmp_score = int(score_tmp[i]) - (int(time_tmp[i][-2:]) % 60) - 60
        child['min_score'] = tmp_score
    """

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
