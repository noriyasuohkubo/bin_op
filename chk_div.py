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
変化率を調査
"""

start = datetime(2016, 1, 1, )
start_stp = int(time.mktime(start.timetuple()))
end = datetime(2024, 12, 1, )
end_stp = int(time.mktime(end.timetuple()))

except_hour_list = [20,21,22,23]

db_no = 3

db_name = "USDJPY_1_0"
host = "localhost"

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:", len(result_data))

up = []
dw = []
even = []

all = []
all_abs = []

border = 0.1

#何秒後の変化率を見るか
after_sec = 30

min_div = 10
min_div_sec = 300

score_c = {}

for line in result_data:
    body = line[0]
    score = line[1]
    tmp = json.loads(body)
    c = float(tmp.get("c"))
    score_c[score] = c

cnt = 0
for score, c in score_c.items():

    if len(except_hour_list) != 0:
        if datetime.fromtimestamp(score).hour in except_hour_list:
            continue

    b = score_c.get(get_decimal_sub(score, min_div_sec))
    a = score_c.get(get_decimal_add(score, after_sec))
    if b == None or a == None:
        continue
    else:
        before_div = abs(get_divide(b, c))
        if before_div < min_div:
            continue
        after_div = get_divide(c, a)

    all.append(after_div)
    all_abs.append(abs(after_div))
    if after_div >= border:
        up.append(after_div)
    elif after_div <= border * -1:
        dw.append(after_div)
    else:
        even.append(after_div)

    cnt += 1

up = np.array(up)
dw = np.array(dw)
all = np.array(all)

print("all :",len(all), np.average(all), )
print("all_abs :",len(all_abs), np.average(all_abs), )
print("up :", len(up), len(up)/len(all), np.average(up))
print("dw :", len(dw), len(dw)/len(all), np.average(dw))