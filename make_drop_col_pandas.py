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
lgbm用のpandasデータの不要行、列を削除
"""
print("START")
start_time = time.perf_counter()

startDt = datetime.datetime(2010, 1, 1)
endDt = datetime.datetime(2024, 6, 30)
#startDt = datetime.datetime(2016, 1, 1)
#endDt = datetime.datetime(2023, 7, 29)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

end_tmp = endDt + timedelta(days=-1)

symbol = "USDJPY"

bet_term = 1

#学習・テストデータの行間隔(sec)
#score % data_term != 0 の行を削除する
data_term = 1

#不要なカラム
drop_cols = []
#drop_cols = ['311-30-DW-126', '311-30-DW-132', '311-30-DW-138', '311-30-DW-144', '311-30-DW-150', '311-30-DW-156', '311-30-DW-162', '311-30-DW-168', '311-30-DW-174', '311-30-DW-180', '311-30-DW-186', '311-30-DW-192', '311-30-DW-198', '311-30-DW-204', '311-30-DW-210', '311-30-DW-216', '311-30-DW-222', '311-30-DW-228', '311-30-DW-234', '311-30-DW-240', '311-30-DW-246', '311-30-DW-252', '311-30-DW-258', '311-30-DW-264', '311-30-DW-270', '311-30-DW-276', '311-30-DW-282', '311-30-DW-288', '311-30-DW-294', '311-30-DW-300', '311-30-DW-306', '311-30-DW-312', '311-30-DW-318', '311-30-DW-324', '311-30-DW-330', '311-30-DW-336', '311-30-DW-342', '311-30-DW-348', '311-30-DW-354', '311-30-DW-360', '311-30-DW-366', '311-30-DW-372', '311-30-DW-378', '311-30-DW-384', '311-30-DW-390', '311-30-DW-396', '311-30-DW-402', '311-30-DW-408', '311-30-DW-414', '311-30-DW-420', '311-30-DW-426', '311-30-DW-432', '311-30-DW-438', '311-30-DW-444', '311-30-DW-450', '311-30-DW-456', '311-30-DW-462', '311-30-DW-468', '311-30-DW-474', '311-30-DW-480', '311-30-DW-486', '311-30-DW-492', '311-30-DW-498', '311-30-DW-504', '311-30-DW-510', '311-30-DW-516', '311-30-DW-522', '311-30-DW-528', '311-30-DW-534', '311-30-DW-540', '311-30-DW-546', '311-30-DW-552', '311-30-DW-558', '311-30-DW-564', '311-30-DW-570', '311-30-DW-576', '311-30-DW-582', '311-30-DW-588', '311-30-DW-594', '311-30-DW-600', '311-30-UP-126', '311-30-UP-132', '311-30-UP-138', '311-30-UP-144', '311-30-UP-150', '311-30-UP-156', '311-30-UP-162', '311-30-UP-168', '311-30-UP-174', '311-30-UP-180', '311-30-UP-186', '311-30-UP-192', '311-30-UP-198', '311-30-UP-204', '311-30-UP-210', '311-30-UP-216', '311-30-UP-222', '311-30-UP-228', '311-30-UP-234', '311-30-UP-240', '311-30-UP-246', '311-30-UP-252', '311-30-UP-258', '311-30-UP-264', '311-30-UP-270', '311-30-UP-276', '311-30-UP-282', '311-30-UP-288', '311-30-UP-294', '311-30-UP-300', '311-30-UP-306', '311-30-UP-312', '311-30-UP-318', '311-30-UP-324', '311-30-UP-330', '311-30-UP-336', '311-30-UP-342', '311-30-UP-348', '311-30-UP-354', '311-30-UP-360', '311-30-UP-366', '311-30-UP-372', '311-30-UP-378', '311-30-UP-384', '311-30-UP-390', '311-30-UP-396', '311-30-UP-402', '311-30-UP-408', '311-30-UP-414', '311-30-UP-420', '311-30-UP-426', '311-30-UP-432', '311-30-UP-438', '311-30-UP-444', '311-30-UP-450', '311-30-UP-456', '311-30-UP-462', '311-30-UP-468', '311-30-UP-474', '311-30-UP-480', '311-30-UP-486', '311-30-UP-492', '311-30-UP-498', '311-30-UP-504', '311-30-UP-510', '311-30-UP-516', '311-30-UP-522', '311-30-UP-528', '311-30-UP-534', '311-30-UP-540', '311-30-UP-546', '311-30-UP-552', '311-30-UP-558', '311-30-UP-564', '311-30-UP-570', '311-30-UP-576', '311-30-UP-582', '311-30-UP-588', '311-30-UP-594', '311-30-UP-600']

#残すカラム
not_drop_cols = '887-39-DW@887-39-UP@887-39-SAME@885-6-REG@887-39-DW-4@887-39-UP-4@887-39-SAME-4@885-6-REG-4@887-39-DW-8@887-39-UP-8@887-39-SAME-8@885-6-REG-8@887-39-DW-12@887-39-UP-12@887-39-SAME-12@885-6-REG-12@o@score'.split('@')

#対象外とする時間
EXCEPT_LIST = []
EXCEPT_LIST_STR = "_EL" + list_to_str(EXCEPT_LIST, spl="-") if len(EXCEPT_LIST) != 0 else ""

df_file = "MF187"
df_file_type = df_file[:2]

if df_file_type == 'CF':
    df_file_type_name = 'CONCAT'
    df_file_path_base = "/concat_file/"
elif df_file_type == 'MF':
    df_file_type_name = 'MERGE'
    df_file_path_base = "/merge_file/"
elif df_file_type == 'IF':
    df_file_type_name = 'INPUT'
    df_file_path_base = "/input_file/"
elif df_file_type == 'PF':
    df_file_type_name = 'PREDICT'
    df_file_path_base = "/predict_file/"

df_file_path = "/db2/lgbm/" + symbol + df_file_path_base + df_file + ".pickle"

#ファイル読み込み
with open(df_file_path, 'rb') as f:
    df = pickle.load(f)
    print("df info")
    print(df.info())
    # 開始、終了期間で絞る
    df = df.query('@start_score <= score < @end_score')

#行削除
if bet_term != data_term:
    df = df[df.score % data_term == 0]

#列削除
if len(drop_cols) != 0:
    df.drop(columns=drop_cols, inplace=True)

if len(not_drop_cols) != 0:
    df = df.loc[:, not_drop_cols]

if len(EXCEPT_LIST) != 0:
    df.loc[:, "out"] = df.isnull().any(axis=1)

    df_t = pd.Series(pd.to_datetime(df['score'], utc=True, unit='s'),name="time")  # 日時追加 UTC, 単位(unit)をs(sec)にする
    df = pd.concat([df, df_t], axis=1)

    # 時データ追加
    df.loc[:, "hour"] = df['time'].dt.hour

    df.loc[df['hour'].isin(EXCEPT_LIST), "out"] = True

    # 対象外データは除外
    df = df[df["out"] == False]

    # 不要データ削除
    df = df.drop(['out', 'time', 'hour'], axis=1)


print("drop finished")
print("df info")
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
tmp_file_name = symbol + "_B" + str(bet_term)+ "_D" + str(data_term) + "_IN-" + input_name + "_" + EXCEPT_LIST_STR + \
                date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,format='%Y%m%d') + drop_files + "_" + socket.gethostname()

db_name_file = df_file_type_name + "_FILE_NO_" + symbol

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

tmp_path = "/db2/lgbm/" + symbol + df_file_path_base + df_file_type + str(newest_no)
pickle_file_name = tmp_path + ".pickle"

print("newest_no", newest_no)
print("input_name", tmp_file_name)

df.to_pickle(pickle_file_name)
print("save pickle finished")
print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")



print(datetime.datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)