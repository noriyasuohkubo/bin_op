import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from util import *
import send_mail as mail
"""
スプレッドを0にする
"""

db_list = ["USDJPY_0.25_0"]
db_tick_list = ["USDJPY_0.25_0_TICK"]

db_no_org = 2

host = "127.0.0.1"

start = datetime.datetime(2023, 4, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2024, 8, 10)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no_org, decode_responses=True)

for db_name in db_list:
    db_name_new = db_name + "_new" #1件ごとに削除するのが遅いので別テーブルを作成してあとでリネームする

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp = [], [], []

    for cnt, line in enumerate(result_data):
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        tmps["s"] = 0

        redis_db.zadd(db_name_new, json.dumps(tmps), score)

        if cnt % 1000000 == 0:
            dt_now = datetime.datetime.now()
            print(dt_now, " ", cnt)

for db_name in db_tick_list:
    db_name_new = db_name + "_new"  # 1件ごとに削除するのが遅いので別テーブルを作成してあとでリネームする

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp = [], [], []

    for cnt, line in enumerate(result_data):
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        tmps["s"] = 0

        tk_list = tmps.get("tk").split(",")

        new_tk_str = ""
        for tmp_tk in tk_list:
            tmp_close, tmp_spread = tmp_tk.split(":")
            if new_tk_str == "":
                new_tk_str = tmp_close + ":" + str(0)
            else:
                new_tk_str = new_tk_str + "," + tmp_close + ":" + str(0)

        tmps["tk"] = new_tk_str

        redis_db.zadd(db_name_new, json.dumps(tmps), score)

        if cnt % 1000000 == 0:
            dt_now = datetime.datetime.now()
            print(dt_now, " ", cnt)

print("FINISH")
# 終わったらメールで知らせる
mail.send_message(host, ": make_spread_db_0 finished!!!")