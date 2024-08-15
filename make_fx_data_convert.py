import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from chk_summertime import *
from util import *

"""
1秒ごとに画面から取得しているレートデータのopenレート価格をcloseに変換する
"""

#抽出元のDB名
db_name_org = "BTCUSDT_SPOT_S1"

#新規に作成するDB名
db_name_new = "BTCUSD"

host_org = "localhost"
host_new = "localhost"

db_no_org = 0
db_no_new = 0

#実際のスプレッドを移行する場合True
real_spread_flg = True
pips = 0.1

#ask, bid以外のデータ移行するカラム
cols = []

start_date = datetime(2024, 2, 1)
end_date = datetime(2024, 3, 7)

start_stp = int(time.mktime(start_date.timetuple()))
end_stp = int(time.mktime(end_date.timetuple())) -1 #含めないので1秒マイナス

redis_db_org = redis.Redis(host=host_org, port=6379, db=db_no_org, decode_responses=True)
redis_db_new = redis.Redis(host=host_new, port=6379, db=db_no_new, decode_responses=True)

# 処理時間計測
t1 = time.time()

result_data = redis_db_org.zrangebyscore(db_name_org, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

for i, line in enumerate(result_data):
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)

    regist_score = score -1 #open価格を１秒前のデータのclose価格とする

    ask = tmps.get("ask")
    bid = tmps.get("bid")
    close = float((Decimal(str(ask)) + Decimal(str(bid))) / Decimal("2"))

    new_child = {
        "c":close,
        "sc":regist_score,
    }

    if real_spread_flg:
        tmp_spread = get_decimal_sub(ask, bid)
        new_child["s"] = int(get_decimal_multi(get_decimal_divide(1, pips), tmp_spread))
    else:
        new_child["s"] = 0

    for col in cols:
        new_child[col] = tmps.get(col)

    """
    #既存レコードがあるばあい、削除して追加
    tmp_val = redis_db_new.zrangebyscore(db_name_new, regist_score, regist_score)
    if len(tmp_val) >= 1:
        rm_cnt = redis_db_new.zremrangebyscore(db_name_new, regist_score, regist_score)  # 削除した件数取得
        if rm_cnt != 1:
            # 削除できなかったらおかしいのでエラーとする
            print("cannot remove!!!", score)
            exit()
    """
    ret = redis_db_new.zadd(db_name_new, json.dumps(new_child), score)

t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))

