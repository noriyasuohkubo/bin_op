from datetime import datetime
from datetime import timedelta
import time
import socket

import redis
import json
from decimal import Decimal
import traceback
import os
import gc
import numpy as np
import math
import sys
from util import *
import send_mail as mail

"""
自分が属する1分足での高値安値をDB登録する
"""


start_day = "2016/01/01 00:00:00" #この時間含む(以上)
end_day = "2026/05/02 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

#開始日時が終了日時より前であるかチェック
if start_day_dt >= end_day_dt:
    print("Error:開始日時が終了日時より前です！！！")
    exit()


db_no = 2

db_name_old = "GBPAUD_1_0"
db_name_new = "GBPAUD_1_0_NEW"

#取得元DB
#symbol = "USDJPY"
symbol = "GBPAUD"
db_name_tick = symbol + "_1_0_TICK"

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

def convert():
    # 処理時間計測
    t1 = time.time()


    old_data = redis_db.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("old_data length:" + str(len(old_data)))

    old_data_dict = {}

    for i, line in enumerate(old_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        old_data_dict[score] = tmps

    del old_data

    tick_data = redis_db.zrangebyscore(db_name_tick, start_stp, end_stp, withscores=True)
    print("tick_data length:" + str(len(tick_data)))

    lists = []

    for i, line in enumerate(tick_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        val_dict = {}
        val_dict["tk"] = tmps.get("tk")
        val_dict["sc"] = score

        lists.append(val_dict)

    del tick_data

    cnt = 0

    for j, val in enumerate(lists):
        cnt += 1

        score = val["sc"]
        old_data = old_data_dict.get(score)
        if old_data == None:
            continue

        # 自分より前の00秒までの必要なデータの長さ
        need_len = int(get_decimal_mod(score, 60))

        if j < need_len:
            #前のデータが足りない
            old_data["m1_h"] = None
            old_data["m1_l"] = None
        else:
            # 前のデータを集める
            bef_data = lists[j - need_len: j + 1]

            tmp_tk_list = []
            for aft in bef_data:
                tmp_tk = aft.get("tk").split(",")
                for tk in tmp_tk:
                    c = float(tk.split(":")[0])
                    tmp_tk_list.append(c)

            old_data["m1_h"] = max(tmp_tk_list)
            old_data["m1_l"] = min(tmp_tk_list)

        redis_db.zadd(db_name_new, json.dumps(old_data), score)

        if cnt % 10000000 == 0:
            dt_now = datetime.now()
            print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    # 終わったらメールで知らせる
    mail.send_message(socket.gethostname(), ": make_m1_hl finished!!!")

