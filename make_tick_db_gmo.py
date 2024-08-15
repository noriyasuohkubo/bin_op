import numpy as np
import pandas as pd
import requests
import itertools
import os
from bs4 import BeautifulSoup
import redis
import json
from chk_summertime import *
import csv
from decimal import Decimal
import pandas as pd

def trade_csv_list(tick_code, year, month):
    # 取得したいWEBサイトをurlで指定
    url = f'https://api.coin.z.com/data/trades/{tick_code}/{year}/{month}'
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    contents = soup.find(class_="data-list")
    get_a = contents.find_all("a")
    # 上の回数分hrefの中身を取得ループ
    csv_list = [url + '/' + alink.get("href") for alink in get_a]
    return csv_list

# 指定した月まで一括で取得
def lump_together_list(tick_code, year, month):
    csv_list = []
    csv_list += trade_csv_list(tick_code, str(year), str(month).zfill(2))
    return csv_list

# いくつかの銘柄をまとめてデータをとる
pair = ['BTC_JPY',] #レバレッジ取引のデータ取得
target = [
    (2018, 9),
    (2018, 10),
    (2018, 11),
    (2018, 12),
    (2019, 1),
    (2019, 2),
    (2019, 3),
    (2019, 4),
    (2019, 5),
    (2019, 6),
    (2019, 7),
    (2019, 8),
    (2019, 9),
    (2019, 10),
    (2019, 11),
    (2019, 12),
    (2020, 1),
    (2020, 2),
    (2020, 3),
    (2020, 4),
    (2020, 5),
    (2020, 6),
    (2020, 7),
    (2020, 8),
    (2020, 9),
    (2020, 10),
    (2020, 11),
    (2020, 12),
    (2021, 1),
    (2021, 2),
    (2021, 3),
    (2021, 4),
    (2021, 5),
    (2021, 6),
    (2021, 7),
    (2021, 8),
    (2021, 9),
    (2021, 10),
    (2021, 11),
    (2021, 12),
    (2022, 1),
    (2022, 2),
    (2022, 3),
    (2022, 4),
    (2022, 5),
    (2022, 6),
    (2022, 7),
    (2022, 8),
    (2022, 9),
    (2022, 10),
    (2022, 11),
    (2022, 12),
    (2023, 1),
    (2023, 2),
    (2023, 3),
]

csv_list = []
for tick, year_month in itertools.product(pair, target):
    y, m = year_month
    csv_list += lump_together_list(tick, y, m)

print(csv_list)

#新規に作成するDB名
DB_NAME = "BTCJPY_TICK"
DB_NO = 3
HOST = "127.0.0.1"

startDt = datetime(2018, 9, 1,)
endDt = datetime(2023, 3, 1)

start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

# 処理時間計測
t1 = time.time()
redis_db = redis.Redis(host=HOST, port=6379, db=DB_NO, decode_responses=True)

for file in csv_list:
    print(file)

    df = pd.read_csv(file, index_col="timestamp",  parse_dates=True, date_parser=lambda x: time.mktime(pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S.%f").timetuple()))
    df= pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S.%f")
    #df_t = pd.Series(time.mktime(df["timestamp"].timetuple()), name="score")
    #df = pd.concat([df, df_t], axis=1)
    df['score'] = df.timestamp.apply(lambda t: t.timestamp())
    #print(df['score'][:10])
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

        price = int(float(row.price))
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
