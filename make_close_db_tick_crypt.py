import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from chk_summertime import *

"""
make_tick_db_cryptやmake_tick_db_gmoで作成したDBデータにもとづいて
close,timeの値を保存していく

tickデータがない時間については補完する

スプレッドについては決め打ちで0を入れておいて、conf_class.pyのADJUST_PIPSで利益を調節する
"""

#抽出元のDB名
symbol_org = "BTCUSD_TICK_KRAKEN"
#秒間隔
term = 1
#新規に作成するDB名
symbol = "BTCUSD"

close_shift = 1

in_db_no = 2
out_db_no = 3

host = "127.0.0.1"

start_date = datetime(2014, 1, 1)
end_date = datetime(2023, 5, 1)

redis_db_in = redis.Redis(host=host, port=6379, db=in_db_no, decode_responses=True)
redis_db_out = redis.Redis(host=host, port=6379, db=out_db_no, decode_responses=True)

def regist_db(now_data, now_key):
    regist_score = now_key
    now_close = now_data[-1]["mid"]
    mids = []
    for tmp_data in now_data:
        mids.append(tmp_data["mid"])


    spr = now_data[-1]["spr"]  # spreadは最後のtickのspreadを入れる
    tick_str = ""  # tickデータをmid + ":" + sprでカンマ区切りで入れていく

    for i, data in enumerate(now_data):
        if i != 0:
            if now_data[i-1]["mid"] == data["mid"] and now_data[i-1]["spr"] == data["spr"]:
                #前のtickと同じmidでない場合だけ登録(メモリの無駄なので省略する)
                continue
            else:
                tick_str = tick_str + "," + str(data["mid"]) + ":" + str(data["spr"])
        else:
            tick_str = str(data["mid"]) + ":" + str(data["spr"])

    #tick_lenをもとめる
    #tick_num = redis_db_out.zrangebyscore(symbol_org, regist_score, float(Decimal(str(regist_score)) + Decimal("0.999")))
    #tick_len = len(tick_num)

    child = {'c': now_close,
             's': spr,
             'sc': regist_score,
             'tk': tick_str,
             #'v': tick_len,#volume
             }

    """
    tmp_val = redis_db_out.zrangebyscore(symbol, regist_score, regist_score)
    if len(tmp_val) == 0:
       redis_db_out.zadd(symbol, json.dumps(child), regist_score)
    """
    redis_db_out.zadd(symbol, json.dumps(child), regist_score)

def regist_data():
    global start_date

    while True:
        print(datetime.now(), " date:", start_date)
        #1日ごとにデータを集めてレコードを作っていく
        next_date = start_date + timedelta(days=1)
        next_stp = float(time.mktime(next_date.timetuple())) - 0.001

        start_stp = int(time.mktime(start_date.timetuple()))

        end = next_date + timedelta(seconds=int(Decimal("-1")*Decimal(str(term))))
        end_key = int(time.mktime(end.timetuple()))

        prev_key = None
        prev_data = []
        now_key = None
        now_data = []

        # 前日の23:59:59のレコードを取得しておく
        result_prev = redis_db_out.zrangebyscore(symbol, start_stp - term, start_stp - term, withscores=True)
        for line in result_prev:
            body = line[0]
            prev_key = int(line[1])
            tmps = json.loads(body)
            mid = float(tmps.get("c"))
            spr = int(tmps.get("s"))
            prev_data = [{
                "mid":mid,
                "spr": spr,
                          }]

        result_by_day = redis_db_in.zrangebyscore(symbol_org, start_stp, next_stp, withscores=True)
        """
        print(result_by_day[:10])
        tmp_reverse = result_by_day[-10:]
        tmp_reverse.reverse()
        print(tmp_reverse)
        """
        for j, line in enumerate(result_by_day):
            body = line[0]
            tmp_score = float(line[1])
            tmps = json.loads(body)
            tmp_mid = float(tmps.get("rate"))

            tmp_date = datetime.fromtimestamp(tmp_score)
            #マイクロ秒なしのdatetime
            date_no_microsec = datetime(tmp_date.year,tmp_date.month,tmp_date.day,tmp_date.hour,tmp_date.minute,tmp_date.second)
            score = int(time.mktime(date_no_microsec.timetuple()))

            tmp_child = {
                "mid":tmp_mid,
                "spr": 0,
            }

            key = score - int(Decimal(str(score)) % Decimal(str(term)))

            if (now_key != None and now_key != key):
                #keyが変わった場合、蓄積したnow_dataからrecord作成してkeyまで間があいた場合は埋めていく

                if prev_key != None:
                    regist_key = prev_key + term
                    while True:
                        if regist_key == now_key:
                            regist_db(now_data, regist_key)
                            prev_data = [now_data[-1]]
                            prev_key = regist_key
                            break
                        else:
                            #間があいた場合
                            regist_db(prev_data, regist_key)
                            regist_key = regist_key + term
                else:
                    #前日の23:59:58のレコードがない場合(日曜日の場合はない)
                    prev_data = [now_data[-1]]

                    prev_key = now_key
                    #print(prev_key, prev_data)

                now_key = key
                now_data = []

            now_data.append(tmp_child)
            if now_key == None:
                now_key = key

        if len(now_data) != 0:
            if prev_key != None:
                regist_key = prev_key + term
                while True:
                    if regist_key > end_key:
                        break

                    if regist_key == now_key:
                        regist_db(now_data, regist_key)
                        prev_data = [now_data[-1]]
                        regist_key = regist_key + term
                    elif regist_key < now_key:
                        #print("2",regist_key)
                        regist_db(prev_data, regist_key)
                        regist_key = regist_key + term
                    else:
                        #print("3", regist_key)
                        regist_db(prev_data, regist_key)
                        regist_key = regist_key + term

        start_date = next_date
        if start_date >= end_date:
            break

if __name__ == '__main__':
    # 処理時間計測
    t1 = time.time()
    regist_data()

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

