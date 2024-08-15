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
lgbm用のcsvデータをpickleに保存しなおす。
メモリ容量が足りない場合に一時的にCSVにデータ蓄積したものをpickleに保存する時に使用する。
"""
print("START")
start_time = time.perf_counter()

symbol = "USDJPY"
bet_term = 2

file_list = ["MF38","MF39"]#csvデータ作成に用いたファイル名

startDt = datetime(2007, 1, 1)
endDt = datetime(2023, 9, 3)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

end_tmp = endDt + timedelta(days=-1)

csv_file_name = "/db2/lgbm/" + symbol + "/csv_file/tmp.csv"

df = pd.read_csv(csv_file_name)

df = df.set_index("score", drop=False)
df = df.sort_index(ascending=True)  # scoreの昇順　古い順にする

print("csv read finished")
print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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
print(df.info())


input_name = list_to_str(csv_regist_cols, ",")
files = "_FS-" + list_to_str(file_list)
tmp_file_name = symbol + "_B" + str(bet_term) + "_IN-" + input_name + "_" + \
                date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,format='%Y%m%d') + files + "_" + socket.gethostname() + "_CSV_TO_PICKLE"

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


# pickleで書き直す
df.to_pickle(pickle_file_name)
print("save pickle finished")
print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")


print(datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)