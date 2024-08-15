import os
import signal
import sys
import time
from datetime import datetime, timedelta, date
from chk_summertime import *
import redis
import json
import numpy as np
from util import *

"""
FXのDB履歴からループに要する秒の平均を算出する
"""

#DB情報
db_name = "USDJPY_4_GMO"
#db_name = "USDJPY_4_LION"

db_no = 8
host = "win9"

#レコード削除対象期間
start = datetime(2024, 3, 1, 0)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2025, 1, 1,0)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)

print("result_data length:", len(result_data))
loop_sec_list = []

for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    loop_sec = tmps.get("loop_take")
    if loop_sec != None:
        loop_sec_list.append(float(loop_sec))

loop_sec_list = np.array(loop_sec_list)
print("loop_sec_list length:", len(loop_sec_list))
print("AVG LOOP TAKE SEC:", np.average(loop_sec_list))
print("MAX LOOP TAKE SEC:", np.max(loop_sec_list))

start = 0
while True:
    end = get_decimal_add(start, 0.1)
    if end > 10:
        break
    length = len(np.where((loop_sec_list >= start) & (loop_sec_list < end))[0])
    if length != 0:
        print(str(start) + "~" + str(end) + ":", length)




    start = get_decimal_add(start, 0.1)