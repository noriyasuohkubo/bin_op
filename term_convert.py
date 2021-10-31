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


"""
make_close_dbで作成した2秒間隔のレコードを任意の間隔に拡張しなおす。
例)秒から1分

変更間隔が1分なら00:00:00のレコードの場合00:00:55のレコードの終値を登録することになる

"""

start_day = "2019/12/31 00:00:00" #この時間含む(以上)
end_day = "2021/09/30 22:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

#開始日時が終了日時より前であるかチェック
if start_day_dt >= end_day_dt:

    print("Error:開始日時が終了日時より前です！！！")
    exit()

#変更する間隔(sec)

terms = [10,30,90,300,]

#DBのもとのレコード秒間隔
org_term = 1

bet_term = 2
#closeの値をdbレコードに含める
close_flg = False

div_flg = True

spread_flg = False

#変化率のlogをとる
math_log = False

db_no_old = 3
db_no_new = 3
#取得元DB
db_name_old = "GBPJPY_1_0"

redis_db_old = redis.Redis(host='localhost', port=6379, db=db_no_old, decode_responses=True)
redis_db_new = redis.Redis(host='localhost', port=6379, db=db_no_new, decode_responses=True)

def convert():
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    #key:score val:dict(cs,cm,etc)
    lists = {}
    for line in result_data:
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        #t = tmps.get("t")
        # 秒に変換
        #hour = int(t[11:13]) * 3600
        #min = int(t[14:16]) * 60
        #sec = int(t[17:19])
        #t_val = hour + min + sec

        val_dict = {}
        val_dict["c"] = tmps.get("c")
        if spread_flg == True:
            val_dict["s"] = tmps.get("s")
        #val_dict["t"] = tmps.get("t")
        #val_dict["t_val"] = t_val

        lists[score] = val_dict

    del result_data

    for term in terms:
        # 1分間隔でつくるとして15なら00:01:15 00:02:15 00:03:15と位相をずらす
        shift_list = []
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            shift_list.append(term - ((i + 1) * bet_term))

        print(shift_list)

        for shift in shift_list:
            cnt = 0
            db_name_new = "GBPJPY_" + str(term) + "_" + str(shift)
            print("shift:", shift)
            for score, val in lists.items():

                cnt += 1
                #t_val = val["t_val"]

                #間隔に当たっていたら2秒まえのレコードを取得
                # cdを計算するため
                #if t_val % term == shift:
                if score % term == shift:
                    #2秒前のレコード
                    if (score -org_term) in lists:
                        prev = lists[score -org_term]
                    else:
                        continue
                        #なければとばす

                    if (score + term -org_term) in lists:
                        after = lists[score + term -org_term]

                        divide = float(after["c"]) / float(prev["c"])
                        if after["c"] == prev["c"]:
                            divide = 1
                        if math_log:
                            divide = 10000 * math.log(divide, math.e * 0.1)
                        else:
                            divide = 10000 * (divide - 1)

                        child = {#'c': after["c"],
                                 'sc': score,
                                 }

                        if div_flg == True:
                            child["d"] = divide

                        if spread_flg == True:
                            child["s"] = val["s"]

                        if close_flg == True:
                            child["c"] = after["c"]


                        tmp_val = redis_db_new.zrangebyscore(db_name_new, score, score)
                        if len(tmp_val) == 0:
                            # レコードなければ登録
                            ret = redis_db_new.zadd(db_name_new, json.dumps(child), score)
                            #ret = redis_db_new.zadd(db_name_new, json.dumps(child), score)
                            # もし登録できなかった場合
                            if ret == 0:
                                print(child)
                                print(score)
                    else:
                        continue

                if cnt % 10000000 == 0:
                    dt_now = datetime.now()
                    print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

