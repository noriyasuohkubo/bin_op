import json
import numpy as np
import os
import redis
import datetime
import time
from decimal import Decimal

#Oandaで取得したレートを新たに保存しなおす

#OANDA側のDB情報
db_name = "OANDA_USDJPY_S1"
db_no = 0
host = "win2"

#作り直すDB情報
new_db_name = "USDJPY_2_0"
new_db_oanda_no = 0
new_db_host = "localhost"

#つくりなおす秒間隔
term = 2
term_str = str(term)

#レコードをつくりなおす期間
start = datetime.datetime(2021, 11, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2022, 1, 1)

end_stp = int(time.mktime(end.timetuple()))

redis_fx_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

redis_new_oanda_db = redis.Redis(host=new_db_host, port=6379, db=new_db_oanda_no, decode_responses=True)

result_data = redis_fx_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

time_tmp, score_tmp, spread_tmp = [], [], []
close_oanda_tmp = []
close_oanda_tmp = []


for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    #取得したレコードのレートをterm秒前のレコードのcloseとして登録しなおす
    if Decimal(str(score)) % Decimal(term_str) == 0:
        regist_score = int(score - term)
        #print(regist_score)
        ask = tmps.get("ask")
        bid = tmps.get("bid")
        close = float((Decimal(str(ask)) + Decimal(str(bid))) / Decimal("2")) #仲値を計算
        spread = tmps.get("spread")

        score_tmp.append(regist_score)
        close_oanda_tmp.append(close)

        time_tmp.append(str(datetime.datetime.fromtimestamp(regist_score)))
        spread_tmp.append(spread)

del result_data

cnt = 0
for i in range(len(close_oanda_tmp)):
    if i == 0:
        continue

    divide = close_oanda_tmp[i] / close_oanda_tmp[i -1]
    if close_oanda_tmp[i] == close_oanda_tmp[i -1] :
        divide = 1

    divide = 10000 * (divide - 1)

    child = {'c': close_oanda_tmp[i],
             'd': divide,
             't': time_tmp[i],
             's': spread_tmp[i],
             }

    tmp_val = redis_new_oanda_db.zrangebyscore(new_db_name, score_tmp[i], score_tmp[i])
    if len(tmp_val) == 0:
        # レコードなければ登録

        ret = redis_new_oanda_db.zadd(new_db_name, json.dumps(child), score_tmp[i])
        # もし登録できなかった場合
        if ret == 0:
            print(child)
            print(score)

    cnt +=1
    if cnt % 1000000 == 0:
        print(datetime.datetime.now(), " cnt:" + str(cnt))

#redis_new_oanda_db.save()


print("finish!!!")

