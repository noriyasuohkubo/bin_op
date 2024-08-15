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


"""
ubuntuのハイロー取引で作成したDBデータ{"close":142.765,"time":"2020-01-06 19:56:06","spread":0.1}
をcentos用に変換

close→c
spread→s

"""

start_day = "2023/1/1 00:00:00" #この時間含む(以上)
end_day = "2023/6/1 00:00:00"  # この時間含む(以下)

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple()))

#変化率のlogをとる
math_log = False

db_no_old = 1
db_no_new = 1

#取得元DB
db_name_old = "GBPJPY_30_SPR"
db_name_new = "GBPJPY_2_0"

redis_db_old = redis.Redis(host='localhost', port=6379, db=db_no_old, decode_responses=True)
redis_db_new = redis.Redis(host='localhost', port=6379, db=db_no_new, decode_responses=True)


def convert():
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp, spread_tmp = [], [], [], []
    cnt0 = 0
    for line in result_data:
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        score_tmp.append(score)
        close_tmp.append(float(tmps.get("close")))
        time_tmp.append(tmps.get("time"))
        if tmps.get("spread") != None:
            # 0.3などの形で入っているので実際の値にするため10倍にする
            spread_tmp.append(int(Decimal(str(tmps.get("spread"))) * Decimal("10")))

        else:
            spread_tmp.append(-1)

    del result_data

    for i in range(len(close_tmp)):
        if i == 0 :
            continue
        if close_tmp[i -1] == 0 or close_tmp[i] == 0:
            print("zero", score_tmp[i -1])
            continue
        if close_tmp[i -1] == None or close_tmp[i] == None:
            print("None", score_tmp[i -1])
            continue

        divide = float(close_tmp[i] / close_tmp[i -1])
        if close_tmp[i] == close_tmp[i -1] :
            divide = 1
        if math_log:
            divide = 10000 * math.log(divide, math.e * 0.1)
        else:
            divide = 10000 * (divide - 1)

        child = {'c': close_tmp[i],
                 'd1': divide,
                 't': time_tmp[i],
                 's': spread_tmp[i],
                 }


        tmp_val = redis_db_new.zrangebyscore(db_name_new, score_tmp[i], score_tmp[i])
        if len(tmp_val) == 0:
            # レコードなければ登録

            ret = redis_db_new.zadd(db_name_new, json.dumps(child), score_tmp[i])
            # もし登録できなかった場合
            if ret == 0:
                print(child)
                print(score_tmp[i])


        #ret = redis_db_new.zadd(db_name_new, json.dumps(child), score_tmp[i])

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

