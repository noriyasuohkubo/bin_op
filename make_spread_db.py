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
dukasのデータをもとに
ハイローのデータからスプレッド情報だけを取得して新たにレコード作成する
"""

#抽出元のDB名
symbol_org = "GBPJPY_2_0_OLD"
#新規に作成するDB名
symbol = "GBPJPY_2_0"
#ハイローのDB名
symbol_hl = "GBPJPY_30_SPR"

#直前の1分足のスコアをデータに含めるか
#例えば12：03：44なら12:02
include_min = False

in_db_no = 1
out_db_no = 1
hl_db_no = 1

host = "127.0.0.1"

start = datetime.datetime(2020, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2021, 10, 1)
end_stp = int(time.mktime(end.timetuple()))

redis_db_in = redis.Redis(host=host, port=6379, db=in_db_no, decode_responses=True)
redis_db_out = redis.Redis(host=host, port=6379, db=out_db_no, decode_responses=True)
redis_db_hl = redis.Redis(host=host, port=6379, db=hl_db_no, decode_responses=True)

result_data = redis_db_in.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

close_tmp, time_tmp, score_tmp = [], [], []

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    # データが1秒間隔の場合
    #DataSequence2の仕様により奇数秒にのみスプレッドデータをもたせる
    #tmp_hl = redis_db_hl.zrangebyscore(symbol_hl, score -1, score -1)
    tmp_hl = redis_db_hl.zrangebyscore(symbol_hl, score , score ) #データが2秒間隔の場合
    spread_tmp = -1

    if len(tmp_hl) != 0:
        tmp_body = tmp_hl[0]
        tmp_val = json.loads(tmp_body)
        spread_tmp = -1
        if tmp_val.get("spread") != None:
            # 0.3などの形で入っているので実際の値にするため10倍にする
            spread_tmp = int(Decimal(str(tmp_val.get("spread"))) * Decimal("10"))

    child = {'c': tmps.get("c"),
             'd': tmps.get("d"),
             't': tmps.get("t"),
             's': spread_tmp,
             }
    redis_db_out.zadd(symbol, json.dumps(child), score)


print("FINISH")
