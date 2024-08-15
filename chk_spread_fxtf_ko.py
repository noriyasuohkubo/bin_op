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
FXTFのノックアウトオプションのスプレッドを調査
"""

start = datetime(2023, 4, 1,  )
start_stp = int(time.mktime(start.timetuple()))
end = datetime(2024, 3, 7, )
end_stp = int(time.mktime(end.timetuple()))

db_no = 8
db_name = "USDJPY_60_FXTF_KO"
host = "win8"

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:", len(result_data))

spreads = {}

cnt = 0
for line in result_data:
    cnt += 1
    body = line[0]
    score = line[1]
    tmp = json.loads(body)

    spread = tmp.get("spread")

    if spread in spreads.keys():
        spreads[spread] +=1
    else:
        spreads[spread] = 1

    ret = redis_db.zadd(db_name, json.dumps(tmp), score)

for k,v in spreads.items():
    print(k,v)