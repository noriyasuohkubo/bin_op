import json
import numpy as np
import os

import psutil
import redis
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
from datetime import datetime
"""
lgbm用のpandasデータを縦方向に連結する
"""
print("START")
print("memory0", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

start_time = time.perf_counter()

symbol = "USDJPY"
bet_term = 2
data_term = 2

startDt = datetime(2023, 4, 1)
endDt = datetime(2024, 8, 10)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))


dfs =[
    "CF174",
    "MF219"
]

"""
for i in range(207,211+1):
    dfs.append('PF' + str(i))
"""

df_paths = []
for d in dfs:
    if d[:2] == "IF":
        df_paths.append("/db2/lgbm/" + symbol + "/input_file/")
    elif d[:2] == "CF":
        df_paths.append("/db2/lgbm/" + symbol + "/concat_file/")
    elif d[:2] == "MF":
        df_paths.append("/db2/lgbm/" + symbol + "/merge_file/")
    elif d[:2] == "PF":
        df_paths.append("/db2/lgbm/" + symbol + "/predict_file/")
    else:
        exit(1)

print(dfs)
print(df_paths)


df_list = []
for i,df_tmp in enumerate(dfs):

    df_file_path = df_paths[i] + df_tmp + ".pickle"
    # ファイル読み込み
    with open(df_file_path, 'rb') as f:
        df = pickle.load(f)

    print(df_file_path)
    print("info")
    print(df.info())
    df_list.append(df)
    print("memory0." + str(i), psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

# 結合実施
df_org = pd.concat(df_list)
print("memory1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
gc.collect()
print("memory1.1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
# 開始、終了期間で絞る
df_org.query('@start_score <= score < @end_score',inplace=True)
print("memory2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

df_org.set_index("score", drop=False, inplace=True)
print("memory3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

df_org.sort_index(ascending=True, inplace=True)  # scoreの昇順　古い順にする
print("memory4", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

df_index_list_tmp = list(df_org.index)
if len(df_index_list_tmp) != len(set(df_index_list_tmp)):
    # scoreに重複があるのでエラー
    print("score duplicate!!!")
    visited = set()
    dup = [x for x in df_index_list_tmp if x in visited or (visited.add(x))]
    print("duplicate:", dup)

    # 重複する行を削除
    df_org.drop(index=dup, inplace=True)

    # 再度チェックする
    df_index_list_tmp = list(df_org.index)
    if len(df_index_list_tmp) != len(set(df_index_list_tmp)):
        # scoreに重複があるのでエラー
        print("score duplicate!!!")
        visited = set()
        dup = [x for x in df_index_list_tmp if x in visited or (visited.add(x))]
        print("duplicate:", dup)
        exit(1)

print("concat finished")
print("info")
print(df_org.info())

print(df_org[:100])
csv_regist_cols = df_org.columns.tolist()

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
concat_files = "_CFS-" + list_to_str(dfs)
end_tmp = endDt + timedelta(days=-1)

tmp_file_name = symbol + "_B" + str(bet_term) + "_D" + str(data_term) + "_IN-" + input_name + "_" + \
                date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,format='%Y%m%d') + concat_files + "_" + socket.gethostname()

db_name_file = "CONCAT_FILE_NO_" + symbol
# win2のDBを参照してモデルのナンバリングを行う
r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET CONCAT_FILE_NO")
    exit(1)
else:
    newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

    for line in result:
        body = line[0]
        score = float(line[1])
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

csv_path = "/db2/lgbm/" + symbol + "/concat_file/" + "CF" + str(newest_no)
#csv_file_name = csv_path + ".csv"
pickle_file_name = csv_path + ".pickle"

print("dir_no", newest_no)
print("input_name", tmp_file_name)
# print("csv_regist_cols", csv_regist_cols)

#保存
df_org.to_pickle(pickle_file_name)

print(datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)