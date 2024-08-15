import json
import numpy as np
import os
import redis
from datetime import datetime, timedelta
import time
import gc
import math
from decimal import Decimal
from send_mail import *
from util import tracebackPrint

"""
取引を行っているマシンが稼働しているかDBアクセスして確認
"""

machines = [
    'win2',
    'win3',
    'win4',
    'win5',
    'win6',
    'win7',
    #'192.168.1.17',
    'win8',
    'win9',
]

loop_term = 60 * 5 #確認する間隔秒

db_no = 0
db_name = 'heartbeat'

tmp_dt = datetime.now()
base_dt = datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                            hour=tmp_dt.hour, minute=tmp_dt.minute, second=tmp_dt.second, microsecond=0)

# conf.LOOP_TERM
base_dt = base_dt + timedelta(seconds=5)

base_t = time.mktime(base_dt.timetuple()) + 0.01
print("base_dt", base_dt)

while True:
    time.sleep(0.0001)
    if (base_t - time.time()) < 0.0005:  # time.timeの誤差を考慮して0.5ミリ秒早く起きる
        break

    # もし追い越してしまったらエラーとする
    if (base_t - time.time()) < -0.01:
        print("TIME START FAILED!!", base_t, time.time())
        send_message('TIME START FAILED!!', 'heart beat error!!!')

ERR_FLG = False

while (True):
    base_t_just = int(base_t - 0.01)  # base_tは0.01秒遅くなっているため
    base_t_just_score = int(base_t_just)
    base_t_just_dt = datetime.fromtimestamp(base_t_just_score)

    offset = base_t - time.time()  # 起動すべき時間と起動した時間の差
    tmp_offset = offset
    if tmp_offset < 0:
        tmp_offset = tmp_offset * -1
    # offsetが1000ミリ秒以上の場合メール送信 早くても遅くても駄目
    if tmp_offset > 1:
        print("offset over 1000milces", offset,)
        send_message('offset over 1000milces', 'heart beat error!!!')

    if ERR_FLG == False:
        #ハートビート実施
        for m in machines:
            #print(m)
            try:
                redis_db = redis.Redis(host=m, port=6379, db=db_no, decode_responses=True)
                result_data = redis_db.zrangebyscore(db_name, 1, 1, withscores=True)
            except Exception as e:
                print(tracebackPrint(e))
                send_message(m + ': heart beat ConnectionError!!!', m + ': heart beat ConnectionError!!!')
                ERR_FLG = True
                continue

            if len(result_data) == 0:
                #念の為5秒おいてリトライ
                time.sleep(5)
                result_data = redis_db.zrangebyscore(db_name, 1, 1, withscores=True)

                if len(result_data) == 0:
                    send_message(m +': heart beat error!!!', m +': heart beat error!!!')

                    ERR_FLG = True
            else:
                print(base_t_just_dt, "OK:", m)
    # 次に起動すべき時間
    base_t += loop_term

    # 次のターンまでスリープする
    sleep_time = base_t - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)