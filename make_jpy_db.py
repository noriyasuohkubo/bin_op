import json
import numpy as np
import os

import psutil
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from util import *
import talib
import csv
import pandas as pd

"""
1秒のUSDJPYレコードを参照してレートを取り込む
"""
db_no_old = 3
db_no = 2
db_no_jpy = 2

db_name_old = "EURUSD"
db_name = "EURUSD"
db_name_jpy = "USDJPY"

host = "127.0.0.1"
host_jpy = "127.0.0.1"

start_dt = datetime.datetime(2021, 1, 1)
end_dt = datetime.datetime(2023, 5, 1)
start_stp = int(time.mktime(start_dt.timetuple()))
end_stp = int(time.mktime(end_dt.timetuple())) -1

redis_db_old = redis.Redis(host=host, port=6379, db=db_no_old, decode_responses=True)
redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
redis_db_jpy = redis.Redis(host=host_jpy, port=6379, db=db_no_jpy, decode_responses=True)

result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    tmp_val = redis_db_jpy.zrangebyscore(db_name_jpy, score, score, withscores=True)
    #JPYデータがあるものだけレコードに残す
    if len(tmp_val) != 0:
        body_jpy = tmp_val[0][0]

        tmps_jpy = json.loads(body_jpy)
        tmps["jpy"] = tmps_jpy.get("close")
        redis_db.zadd(db_name, json.dumps(tmps), score)




