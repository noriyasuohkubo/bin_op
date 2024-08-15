import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
import redis
import json
import conf_class

"""
highlowのCSVから取引履歴をDBに登録していく
"""

PAIR = "GBP/JPY"
startDt = datetime(2023, 1, 1)
endDt = datetime(2023, 4, 1)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

conf = conf_class.ConfClass()

FILE_DIR = "/app/bin_op/analyze/"

FILE_NAMES = [
    "みねさん_highlow.csv",

]
for FILE_NAME in FILE_NAMES:

    FILE_POSTFIX = FILE_NAME.split("_")[0]
    FILE = FILE_DIR + FILE_NAME

    if os.path.exists(FILE) == False:
        print("そのようなファイルはありません")
        exit(1)

    #GMTのマシンでダウンロードしたなら日付調整は必要なし
    df = pd.read_csv(FILE, encoding="SHIFT-JIS", parse_dates=[" 取引時間 ", " 終了時刻 "], date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S'))
    df["score"] = df[" 取引時間 "].apply(lambda t: int(t.timestamp())) #score追加
    df = df.set_index("score", drop=False) #INDEX追加
    df = df.sort_index(ascending=True)  # scoreで昇順 古い順にする
    df = df.query('@start_score <= score < @end_score')  # 開始、終了期間で絞る


    df = df.rename(columns={'取引原資産': 'pair', ' 取引内容 ': 'startVal', '判定レート ': 'endVal', 'ペイアウト ':'profit', " 取引時間 ":"time"},)

    #for row in df[:300].itertuples():
    #    print(row)

    redis_db = redis.Redis(host='localhost', port=6379, db=conf.DB_TRADE_NO, decode_responses=True)

    regist_cnt = 0
    for row in df.itertuples():
        if row.pair == PAIR:
            s = row.Index

            tmp_val = redis_db.zrangebyscore(conf.DB_TRADE_NAME, s, s)

            if len(tmp_val) == 0:
                result = "win"
                if row.profit == "---":
                    result = "lose"

                child = {
                    "startVal": row.startVal,
                    "endVal": row.endVal,
                    "result": result,
                    "time": row.time.strftime("%Y/%m/%d, %H:%M:%S"),
                }
                ret = redis_db.zadd(conf.DB_TRADE_NAME, json.dumps(child), s)
                regist_cnt += 1

    print("regist_cnt:",regist_cnt)

print("END!!!")
