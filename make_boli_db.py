from datetime import datetime
from datetime import timedelta
import time
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


"""
ボリバンの値をDB登録する
"""


start_day = "2024/12/01 00:00:00" #この時間含む(以上)
end_day = "2026/06/27 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

#開始日時が終了日時より前であるかチェック
if start_day_dt >= end_day_dt:
    print("Error:開始日時が終了日時より前です！！！")
    exit()

#期間
terms = [1]

#DBのもとのレコード秒間隔
org_term = 1
db_no = 1

#取得元DBの足
foot_list = [
    {
        "foot":"M1",
        "boli_length":5,
        "make_div":[], #alphaを入れる
    },
    {
        "foot": "H1",
        "boli_length": 75,
        "make_div": [],  # alphaを入れる
    },
    {
        "foot": "H1",
        "boli_length": 300,
        "make_div": [],  # alphaを入れる
    },
]


symbol = "USDJPY"

db_name = "USDJPY_1_0"
db_name_new = "USDJPY_1_0_NEW"

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)


def convert():
    # 処理時間計測
    t1 = time.time()

    lists_list = []
    score_index_list = []

    for f in foot_list:
        foot = f["foot"]

        db_name_org = symbol + "_" + foot

        #一ヶ月前のデータから取得
        foot_data = redis_db.zrangebyscore(db_name_org, start_stp - (3600 *24 * 30), end_stp, withscores=True)
        print("foot_data length:" + str(len(foot_data)))

        lists = []
        score_index = {} #scoreとlistsでのindexのひもづけ

        for i, line in enumerate(foot_data):
            body = line[0]
            score = float(line[1])
            tmps = json.loads(body)

            lists.append(float(tmps.get("c")))
            score_index[score] = i

        del foot_data

        lists_list.append(lists)
        score_index_list.append(score_index)

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    cnt = 0

    for i, line in enumerate(result_data):
        body = line[0]
        score = float(line[1])
        tmps = json.loads(body)
        now_c = float(tmps.get("c"))

        for i, f in enumerate(foot_list):
            foot = f["foot"]
            boli_length = f["boli_length"]

            if foot == "H1":
                foot_sec = 3600
            elif foot == "M1":
                foot_sec = 60

            #自分が属する足のスコアを取得
            target_foot_score = get_decimal_sub(score, get_decimal_mod(score, foot_sec))

            list_idx = score_index_list[i].get(target_foot_score)

            if list_idx != None:

                #足データを集める
                try:
                    data_list = lists_list[i][list_idx - (boli_length - 1):list_idx]
                    data_list.append(now_c)

                    data_array = np.array(data_list)
                    mean = data_array.mean()
                    std = data_array.std()

                    tmps["BOLI-" + foot + "-" + str(boli_length) + "-MEAN"] = mean
                    tmps["BOLI-" + foot + "-" + str(boli_length) + "-STD"] = std

                    make_div = f["make_div"]
                    for alpha in make_div:
                        #alphaを入れていく
                        up_alpha = get_divide(get_decimal_add(mean, get_decimal_multi(std, alpha)), now_c)
                        dw_alpha = get_divide(get_decimal_sub(mean, get_decimal_multi(std, alpha)), now_c)
                        tmps["BOLI-" + foot + "-" + str(boli_length) + "-DIV" + "-UP" + str(alpha)] = up_alpha
                        tmps["BOLI-" + foot + "-" + str(boli_length) + "-DIV" + "-DW" + str(alpha)] = dw_alpha

                except Exception as e:
                    print(tracebackPrint(e))
                    print(score, target_foot_score)
                    #データない場合
                    pass

        ret = redis_db.zadd(db_name_new, json.dumps(tmps), score)

        cnt += 1

    if cnt % 10000000 == 0:
        dt_now = datetime.now()
        print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

