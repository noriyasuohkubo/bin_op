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
bitflyerのウェブ取引画面から取得したレートとスプレッドをDBに登録する

例)
<SCORE>	<PRICE>	<SPREAD>
1657929180,2872957,0.02%

"""
#抽出元のディレクトリ名
import_dir = "/db/redis/"

#抽出元のファイル名
import_files = [
    "BITFLYER_FX_BTC_JPY_S1_backup.csv",
]

#新規に作成するDB名
DB_NAME = "BTCJPY_RAW_BITFLYER"
DB_NO = 2
HOST = "127.0.0.1"

startDt = datetime(2014, 1,1)
endDt = datetime(2023, 4, 1)

start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

#bitflyerのメンテナンス時間。この間はスプレッドが0になるので登録しない
maintenance_hour = 19
maintenance_minutes = [0,1,2,3,4,5,6,7,8,9,10]

# 処理時間計測
t1 = time.time()
redis_db = redis.Redis(host=HOST, port=6379, db=DB_NO, decode_responses=True)

for file in import_files:
    print(file)

    df = pd.read_csv(import_dir + file, names=('score', 'rate', 'spr',),dtype = {'score':'int', 'rate':'object', 'spr':'object',})

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
        try:
            if ("%" in row.spr) == False:
                # %が文字列にないなら正しく画面上の値を取得できていないのでスキップ
                print(row.score, row.rate, row.spr)
                continue

            spr = row.spr[:-1]
            tmp_score = int(row.score) -1 #closeにするために1秒前のスコアにする
            price = int(row.rate)

            #メンテナンス中のレコードは登録しない
            minute = datetime.fromtimestamp(tmp_score).minute
            hour = datetime.fromtimestamp(tmp_score).hour
            if maintenance_hour == hour and minute in maintenance_minutes:
                continue
            
            #print(tmp_date,price)
            #dtime = datetime.strptime(tmp_date, "%Y.%m.%d %H:%M:%S.%f")

            dtime = datetime.fromtimestamp(tmp_score)

            child = {'c': price,
                     'spr': spr,
                     'time': dtime.strftime("%Y/%m/%d %H:%M:%S")
                     }

            #tmp_val = redis_db.zrangebyscore(DB_NAME, tmp_score, tmp_score)
            #if len(tmp_val) == 0:
            #    redis_db.zadd(DB_NAME, json.dumps(child), tmp_score)

            redis_db.zadd(DB_NAME, json.dumps(child), tmp_score)

        except Exception as e:
            # 変換できない値があるということは正しく画面上の値を取得できていないのでスキップ
            print(row.score, row.rate, row.spr)
        if cnt % 10000000 == 0:
            dt_now = datetime.now()
            print(dt_now, " ", cnt)

t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))
