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
スプレッドを調査
"""

start = datetime(2024, 12, 1,  )
start_stp = int(time.mktime(start.timetuple()))
end = datetime(2026, 1, 24, )
end_stp = int(time.mktime(end.timetuple()))

except_hour_list = [20,21,22,23]

db_no = 2
#db_name = "AXI_USDJPY.p_S1"
db_name = "USDJPY_1_0"
#host = "192.168.1.115"
host = "localhost"


redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:", len(result_data))

spreads = {}

cnt = 0
for line in result_data:
    body = line[0]
    score = line[1]
    tmp = json.loads(body)

    if len(except_hour_list) != 0:
        if datetime.fromtimestamp(score).hour in except_hour_list:
            continue

    #spread = tmp.get("spread")
    spread = tmp.get("s")

    if spread in spreads.keys():
        spreads[spread] +=1
    else:
        spreads[spread] = 1

    cnt += 1

for k, v in sorted(spreads.items()):
    print("spread:",k, " 件数:",v, " %", v/cnt*100)