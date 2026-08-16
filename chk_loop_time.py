import json

import numpy as np
import os
import redis
from datetime import datetime
import time
import sys
from decimal import Decimal
from util import *
"""
ループ時間を調査
"""

start = datetime(2026, 2, 16,  )
start_stp = int(time.mktime(start.timetuple()))
end = datetime(2026, 2, 17, )
end_stp = int(time.mktime(end.timetuple()))

except_hour_list = [20,21,22,23]

db_no = 8
db_name = "USDJPY_PREDICT30_2009"
host = "win5"


redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:", len(result_data))

lists = []

cnt = 0
for line in result_data:
    body = line[0]
    score = line[1]
    tmp = json.loads(body)

    if len(except_hour_list) != 0:
        if datetime.fromtimestamp(score).hour in except_hour_list:
            continue

    loop_take = tmp.get("predict_take")
    lists.append(float(loop_take))

    if float(loop_take) > 0.1:
        print(score, tmp.get("time"),float(loop_take))

list_array = np.array(lists)

print("avg:", np.average(list_array))
print("max:", max(list_array))