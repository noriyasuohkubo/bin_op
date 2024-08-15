import json
import numpy as np
import os

import psutil
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from util import *
import talib
import csv
import pandas as pd
import pickle
import socket

"""
lgbm用のpandasデータの型を変更する
"""
print("START")
start_time = time.perf_counter()

startDt = datetime(2017, 1, 1)
endDt = datetime(2023, 9, 3)
#startDt = datetime(2016, 1, 1)
#endDt = datetime(2023, 7, 29)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

end_tmp = endDt + timedelta(days=-1)

symbol = "USDJPY"
bet_term = 2
data_term = 2

#leftのファイルを基本とする
df_file = "CF37"
df_file_path = "/db2/lgbm/" + symbol + "/concat_file/" + df_file + ".pickle"

#ファイル読み込み
with open(df_file_path, 'rb') as f:
    df = pickle.load(f)
    print("df info")
    #for i in df['194-10-UP'].values:
    #    print(i)
    #exit()

    # 開始、終了期間で絞る
    df = df.query('@start_score <= score < @end_score')

#列名を変更("ダブルクォートがあるとLGBMではエラーとなるので除外する)
col_name_org = df.columns.tolist()
col_name_new = {}
for col in col_name_org:
    col_name_new[col] = col.replace('"', '')

df = df.rename(columns=col_name_new)

print(df.info())

print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

print(df[:100])
print(df[-100:])
csv_regist_cols = df.columns.tolist()


csv_regist_cols.sort()  # カラムを名前順にする
# o,scoreは最後にしたいので削除後、追加
if 'o' in csv_regist_cols:
    csv_regist_cols.remove('o')
    csv_regist_cols.append('o')
if 'score' in csv_regist_cols:
    csv_regist_cols.remove('score')
    csv_regist_cols.append('score')

print("cols", csv_regist_cols)

input_name = list_to_str(csv_regist_cols, "@")
drop_files = "_ORG-" + df_file
tmp_file_name = symbol + "_B" + str(bet_term) + "_D" + str(data_term) + "_IN-" + input_name + "_" + \
                date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,
                                                                          format='%Y%m%d') + drop_files + "_" + socket.gethostname()

db_name_file = "INPUT_FILE_NO_" + symbol
# win2のDBを参照してモデルのナンバリングを行う
r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET INPUT_FILE_NO")
    exit(1)
else:
    newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

    for line in result:
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)
        tmp_name = tmps.get("input_name")
        if tmp_name == tmp_file_name:
            # 同じファイルがないが確認
            print("The File Already Exists!!!")
            exit(1)

    # DBにモデルを登録
    child = {
        'input_name': tmp_file_name,
        'no': newest_no
    }
    r.zadd(db_name_file, json.dumps(child), newest_no)

tmp_path = "/db2/lgbm/" + symbol + "/input_file/" + "IF" + str(newest_no)
pickle_file_name = tmp_path + ".pickle"

print("newest_no", newest_no)
print("input_name", tmp_file_name)

df.to_pickle(pickle_file_name)
print("save pickle finished")
print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")



print(datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)