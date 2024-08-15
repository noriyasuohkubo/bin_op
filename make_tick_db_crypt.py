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
import csv
from decimal import Decimal
import pandas as pd

"""
http://api.bitcoincharts.com/v1/csv/
からダウンロードできるビットコインのtickデータをもとに
DBを作成する。

例)
<DATETIME>	<PRICE>	<VOLUME>
1414752866,39310.000000000000,0.083900000000

"""
#抽出元のディレクトリ名
import_dir = "/db/redis/"

#抽出元のファイル名
import_files = [
    #"coincheckJPY.csv",
   "krakenUSD.csv",
]

#新規に作成するDB名
DB_NAME = "BTCUSD_TICK_KRAKEN"
DB_NO = 2
HOST = "127.0.0.1"

startDt = datetime(2014, 1, 1,)
endDt = datetime(2023, 5, 1)

start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

# 処理時間計測
t1 = time.time()
redis_db = redis.Redis(host=HOST, port=6379, db=DB_NO, decode_responses=True)

for file in import_files:
    print(file)

    df = pd.read_csv(import_dir + file, names=('score', 'rate', 'vol',))

    # 開始、終了期間で絞る
    df = df.query('@start_score <= score < @end_score')

    #header = next(f)
    #print(header)
    cnt = 0
    now_score = None
    score_num = 0
    for row in df.itertuples():
        cnt += 1
        #print(row)

        dtime = datetime.fromtimestamp(int(row.score))

        price = float(row.rate)
        #print(tmp_date,price)
        #dtime = datetime.strptime(tmp_date, "%Y.%m.%d %H:%M:%S.%f")
        tmp_score = dtime.timestamp()

        if now_score == None:
            now_score = tmp_score
            score_num = 0
        elif now_score != None and now_score != tmp_score:
            now_score = tmp_score
            score_num = 0
        elif now_score != None and now_score == tmp_score:
            score_num += 1

        if score_num >= 999:
            print("score_num over 999", tmp_score)
        tmp_score = float(Decimal(str(tmp_score)) + (Decimal(score_num) *  Decimal("0.001") ))

        dtime = datetime.fromtimestamp(tmp_score)

        child = {'rate': price,
                 'time': dtime.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                 }

        tmp_val = redis_db.zrangebyscore(DB_NAME, tmp_score, tmp_score)
        if len(tmp_val) == 0:
            redis_db.zadd(DB_NAME, json.dumps(child), tmp_score)

        if cnt % 10000000 == 0:
            dt_now = datetime.now()
            print(dt_now, " ", cnt)

t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))
