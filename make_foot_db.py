import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from util import get_divide

"""
duaksから取得した分足データ(1分や5分)をもとにdivなどを計算して特徴量を作成していく
"""


# 処理時間計測
t1 = time.time()

#足の長さ sec
term = 3600

#抽出元のDB名
symbol_org = "USDJPY_H1_db2"

#新規に作成するDB名
symbol_new = "USDJPY_" + str(term) + "_FOOT"

#変化率のlogをとる
math_log = False

in_db_no = 3
out_db_no = 2
in_host = "127.0.0.1"
out_host = "127.0.0.1"

start = datetime.datetime(2024, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2026, 5, 2)
end_stp = int(time.mktime(end.timetuple()))

redis_db_in = redis.Redis(host=in_host, port=6379, db=in_db_no, decode_responses=True)
redis_db_out = redis.Redis(host=out_host, port=6379, db=out_db_no, decode_responses=True)

result_data = redis_db_in.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))


for i, line in enumerate(result_data):
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    child = {
        'eh':float(tmps.get("h")),
        'el':float(tmps.get("l")),
        'sc': score,
    }

    """
    #既存レコードがあるばあい、削除して追加    
    tmp_val = redis_db_out.zrangebyscore(symbol_new, score, score)
    if len(tmp_val) >= 1:
        rm_cnt = redis_db_out.zremrangebyscore(symbol_new, score, score)  # 削除した件数取得
        if rm_cnt != 1:
            # 削除できなかったらおかしいのでエラーとする
            print("cannot remove!!!", score)
            exit()
    """
    redis_db_out.zadd(symbol_new, json.dumps(child), score)

    if i % 10000000 == 0:
        dt_now = datetime.datetime.now()
        print(dt_now, " ", i)

t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))

