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
lgbm_make_dataで作成されるtestまたはtrainデータpickleの不要列を削除
残るカラム、もしくは不要なカラムを指定する
"""
print("START")
start_time = time.perf_counter()

symbol = "USDJPY"

test_flg = False

pickle_no = 268

#lgbm_make_dataで必ず作成されるカラム
org_cols = ['target_index', 'hour', 'min', 'sec', 'week', 'weeknum']

#不要なカラム
drop_cols = []
#drop_cols = ['311-30-DW-126', '311-30-DW-132', '311-30-DW-138', '311-30-DW-144', '311-30-DW-150', '311-30-DW-156', '311-30-DW-162', '311-30-DW-168', '311-30-DW-174', '311-30-DW-180', '311-30-DW-186', '311-30-DW-192', '311-30-DW-198', '311-30-DW-204', '311-30-DW-210', '311-30-DW-216', '311-30-DW-222', '311-30-DW-228', '311-30-DW-234', '311-30-DW-240', '311-30-DW-246', '311-30-DW-252', '311-30-DW-258', '311-30-DW-264', '311-30-DW-270', '311-30-DW-276', '311-30-DW-282', '311-30-DW-288', '311-30-DW-294', '311-30-DW-300', '311-30-DW-306', '311-30-DW-312', '311-30-DW-318', '311-30-DW-324', '311-30-DW-330', '311-30-DW-336', '311-30-DW-342', '311-30-DW-348', '311-30-DW-354', '311-30-DW-360', '311-30-DW-366', '311-30-DW-372', '311-30-DW-378', '311-30-DW-384', '311-30-DW-390', '311-30-DW-396', '311-30-DW-402', '311-30-DW-408', '311-30-DW-414', '311-30-DW-420', '311-30-DW-426', '311-30-DW-432', '311-30-DW-438', '311-30-DW-444', '311-30-DW-450', '311-30-DW-456', '311-30-DW-462', '311-30-DW-468', '311-30-DW-474', '311-30-DW-480', '311-30-DW-486', '311-30-DW-492', '311-30-DW-498', '311-30-DW-504', '311-30-DW-510', '311-30-DW-516', '311-30-DW-522', '311-30-DW-528', '311-30-DW-534', '311-30-DW-540', '311-30-DW-546', '311-30-DW-552', '311-30-DW-558', '311-30-DW-564', '311-30-DW-570', '311-30-DW-576', '311-30-DW-582', '311-30-DW-588', '311-30-DW-594', '311-30-DW-600', '311-30-UP-126', '311-30-UP-132', '311-30-UP-138', '311-30-UP-144', '311-30-UP-150', '311-30-UP-156', '311-30-UP-162', '311-30-UP-168', '311-30-UP-174', '311-30-UP-180', '311-30-UP-186', '311-30-UP-192', '311-30-UP-198', '311-30-UP-204', '311-30-UP-210', '311-30-UP-216', '311-30-UP-222', '311-30-UP-228', '311-30-UP-234', '311-30-UP-240', '311-30-UP-246', '311-30-UP-252', '311-30-UP-258', '311-30-UP-264', '311-30-UP-270', '311-30-UP-276', '311-30-UP-282', '311-30-UP-288', '311-30-UP-294', '311-30-UP-300', '311-30-UP-306', '311-30-UP-312', '311-30-UP-318', '311-30-UP-324', '311-30-UP-330', '311-30-UP-336', '311-30-UP-342', '311-30-UP-348', '311-30-UP-354', '311-30-UP-360', '311-30-UP-366', '311-30-UP-372', '311-30-UP-378', '311-30-UP-384', '311-30-UP-390', '311-30-UP-396', '311-30-UP-402', '311-30-UP-408', '311-30-UP-414', '311-30-UP-420', '311-30-UP-426', '311-30-UP-432', '311-30-UP-438', '311-30-UP-444', '311-30-UP-450', '311-30-UP-456', '311-30-UP-462', '311-30-UP-468', '311-30-UP-474', '311-30-UP-480', '311-30-UP-486', '311-30-UP-492', '311-30-UP-498', '311-30-UP-504', '311-30-UP-510', '311-30-UP-516', '311-30-UP-522', '311-30-UP-528', '311-30-UP-534', '311-30-UP-540', '311-30-UP-546', '311-30-UP-552', '311-30-UP-558', '311-30-UP-564', '311-30-UP-570', '311-30-UP-576', '311-30-UP-582', '311-30-UP-588', '311-30-UP-594', '311-30-UP-600']

#残すカラム
not_drop_cols = '1-d-1@1-d-10@1-d-100@1-d-110@1-d-120@1-d-130@1-d-140@1-d-150@1-d-160@1-d-170@1-d-180@1-d-2@1-d-20@1-d-3@1-d-30@1-d-4@1-d-40@1-d-5@1-d-50@1-d-6@1-d-60@1-d-7@1-d-70@1-d-8@1-d-80@1-d-9@1-d-90@887-39-DW@887-39-UP@887-39-SAME@885-6-REG@887-39-DW-4@887-39-UP-4@887-39-SAME-4@885-6-REG-4@887-39-DW-8@887-39-UP-8@887-39-SAME-8@885-6-REG-8@887-39-DW-12@887-39-UP-12@887-39-SAME-12@885-6-REG-12'.split('@')

if test_flg:
    data_path = "/db2/lgbm/" + symbol + "/test_file/TESF" + str(pickle_no) + ".pickle"
    conf_path = "/db2/lgbm/" + symbol + "/test_file/TESF" + str(pickle_no) + "-conf.pickle"
else:
    data_path = "/db2/lgbm/" + symbol + "/train_file/TRAF" + str(pickle_no) + ".pickle"
    conf_path = "/db2/lgbm/" + symbol + "/train_file/TRAF" + str(pickle_no) + "-conf.pickle"

#ファイル読み込み
with open(data_path, 'rb') as f:
    lmd = pickle.load(f)
    df = lmd.get_x()
    #現在のカラム表示
    print("before drop cols:")
    print(df.columns.tolist())

with open(conf_path, 'rb') as f:
    conf = pickle.load(f)

#列削除
if len(drop_cols) != 0:
    df.drop(columns=drop_cols, inplace=True)

if len(not_drop_cols) != 0:
    not_drop_cols.extend(org_cols)
    df = df.loc[:, not_drop_cols]

lmd.x = df
lmd.columns = df.columns.tolist()  # 列リストを更新

#confのINPUT_DATAを更新
tmp_input_data = []
for col in lmd.columns:
    if (col in org_cols) == False:
        tmp_input_data.append(col)

conf.INPUT_DATA = tmp_input_data
if conf.USE_H:
    conf.INPUT_DATA.extend(["hour"])
if conf.USE_M:
    conf.INPUT_DATA.extend(["min"])
if conf.USE_S:
    conf.INPUT_DATA.extend(["sec"])
if conf.USE_W:
    conf.INPUT_DATA.extend(["week"])
if conf.USE_WN:
    conf.INPUT_DATA.extend(["weeknum"])

conf.INPUT_DATA_STR = list_to_str(conf.INPUT_DATA, "@")

# 修正後のカラム表示
print("after drop cols:")
print(df.columns.tolist())

print("drop finished")
print("df info")
print(df.info())

print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

if test_flg:
    db_name_file = "TEST_FILE_NO_" + conf.SYMBOL
else:
    db_name_file = "TRAIN_FILE_NO_" + conf.SYMBOL

r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)

#DB登録
csv_regist_cols = lmd.get_columns()
input_name = list_to_str(csv_regist_cols, "@")

# win2のDBを参照して元のファイルのinput_nameを取得する
result = r.zrangebyscore(db_name_file,pickle_no, pickle_no, withscores=True)
if len(result) == 0:
    print("CANNOT GET FILE_NO", db_name_file)
    exit(1)
else:
    line = result[0]
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)
    tmp_input_name = tmps.get("input_name")
    input_names = tmp_input_name.split("_")

    tmp_file_name = ""

    #input_nameだけすげ替える
    for i in input_names:
        if i[:3] == "IN-":
            i = input_name

        if tmp_file_name == "":
            tmp_file_name = i
        else:
            tmp_file_name = tmp_file_name + "_" + i

print("tmp_file_name:", tmp_file_name)

# win2のDBを参照してモデルのナンバリングを行う
result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET FILE_NO", db_name_file)
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
            print("The File Already Exists!!!", tmp_file_name)
            exit(1)

    # DBにモデルを登録
    child = {
        'input_name': tmp_file_name,
        'no': newest_no
    }
    r.zadd(db_name_file, json.dumps(child), newest_no)

if test_flg:
    data_save_path = "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF" + str(newest_no) + ".pickle"
    conf_save_path = "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF" + str(newest_no) + "-conf.pickle"
else:
    data_save_path = "/db2/lgbm/" + conf.SYMBOL + "/train_file/TRAF" + str(newest_no) + ".pickle"
    conf_save_path = "/db2/lgbm/" + conf.SYMBOL + "/train_file/TRAF" + str(newest_no) + "-conf.pickle"

print("newest_no", newest_no)
print("input_name", tmp_file_name)
# データをpickleで保存する
with open(data_save_path, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
    pickle.dump(lmd, f)

with open(conf_save_path, 'wb') as cf:  # confを保存しておく
    pickle.dump(conf, cf)

print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)