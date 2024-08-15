import os
import signal
import sys
import time
from datetime import datetime, timedelta, date
from chk_summertime import *
import redis
import json
import numpy as np

"""
thinkmarketsのトレード履歴から勝率と平均獲得pipsを算出する
"""

#DB情報
db_name = "USDJPY_40_TM_HISTORY"
db_no = 8
host = "192.168.1.14" #win3
#host = "win5"

#レコード削除対象期間
start = datetime(2024, 3, 1,)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2024, 4, 1,0)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)

trade_cnt = len(result_data)
print("trade_cnt:", trade_cnt)

pips_list = []
win_cnt = 0
lose_cnt = 0


for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    bet_type = tmps.get("bet_type")
    open_rate = float(tmps.get("open_rate"))
    close_rate = float(tmps.get("close_rate"))

    pips = close_rate - open_rate
    if bet_type == "sell":
        pips = pips * -1

    pips_list.append(pips)

    if pips>= 0:
        win_cnt += 1
    elif pips < 0:
        lose_cnt += 1


print("win_rate:", win_cnt/trade_cnt)
print("win_cnt:", win_cnt)
print("lose_cnt:", lose_cnt)

print("avg pips:", np.average(np.array(pips_list)))





