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
水平線をひけるような同じレートが多い値幅を
その数とともにDBに登録する

"""
start_day = "2010/1/1 00:00:00" #この時間含む(以上)
end_day = "2010/11/7 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない


start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

db_no_old = 3 #参照する足のDB
db_no_new = 3 #登録するDB

#足の長さ(秒)
org_term = 60

width = 0.01 #値幅
history_min = 60 * 24 #過去何分のデータを集計対象とするか
history_len = int(Decimal(str(history_min)) * Decimal(str(60)) / Decimal(str(org_term))) -1


symbol = "USDJPY"
db_name_old = "USDJPY_" + str(org_term) + "_FOOT"
db_name_new = "USDJPY_" + str(org_term) + "_" + str(history_min) + "_" + str(width) + "_HOR"

redis_db_old = redis.Redis(host='localhost', port=6379, db=db_no_old, decode_responses=True)
redis_db_new = redis.Redis(host='localhost', port=6379, db=db_no_new, decode_responses=True)


def convert():
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    lists = []

    for i, line in enumerate(result_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        val_dict = {}
        val_dict["c"] = float(tmps.get("c"))
        val_dict["eh"] = float(tmps.get("eh"))
        val_dict["el"] = float(tmps.get("el"))

        val_dict["idx"] = i
        val_dict["sc"] = score

        lists.append(val_dict)

    del result_data

    cnt = 0
    for j, val in enumerate(lists):
        cnt += 1
        score = val["sc"]
        regist_score = score + org_term #登録するスコアは予想時のスコアとする
        end_idx = val["idx"]
        start_idx = end_idx - history_len
        if start_idx < 0 :
            #開始データがなければスキップ
            continue

        #過去のデータを集める
        history_data = lists[start_idx:end_idx+1]

        rate_list = []

        for i, val_dict in enumerate(history_data):

            h = val_dict.get("eh")
            if h != None and np.isnan(h) == False:
                rate_list.append(float(h))

            l = val_dict.get("el")
            if l != None and np.isnan(l) == False:
                rate_list.append(float(l))

        min_rate = min(rate_list)
        max_rate = max(rate_list)
        now_rate = float(Decimal(str(min_rate)) - (Decimal(str(min_rate)) % Decimal(str(width)))) #端数を省いてきりのよい数字にする

        rate_arr = np.array(rate_list)

        rate_cnt_list = []

        while True:
            if now_rate> max_rate:
                break

            next_rate = float(Decimal(str(now_rate)) + Decimal(str(width)))

            #指定値幅に合致する件数をカウント
            hit_cnt = len(np.where((now_rate <= rate_arr) & (rate_arr < next_rate))[0])

            if hit_cnt > 1:
                rate_cnt_list.append(str(now_rate) + ":" + str(hit_cnt))

            now_rate = next_rate

        child ={
            "sc":regist_score,
            "data":list_to_str(rate_cnt_list, ",")
        }


        #既存レコードがあるばあい、削除して追加    
        tmp_val = redis_db_new.zrangebyscore(db_name_new, regist_score, regist_score)
        registed_cnt = len(tmp_val)
        if registed_cnt >= 1:
            rm_cnt = redis_db_new.zremrangebyscore(db_name_new, regist_score, regist_score)  # 削除した件数取得
            if rm_cnt  != registed_cnt:
                # 削除できなかったらおかしいのでエラーとする
                print("cannot remove!!!", regist_score)
                exit()

        ret = redis_db_new.zadd(db_name_new, json.dumps(child), regist_score)

        if cnt % 10000000 == 0:
            dt_now = datetime.now()
            print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

