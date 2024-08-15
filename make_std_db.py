import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import warnings
import math
from decimal import Decimal
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle

"""
標準化または正規化を行い、値を保存していく
"""
type = "minmax" #std or minmax

symbol = "USDJPY"
bet_term = 5
term = 60 #抽出元のレコード間隔秒数

target_cols = ["d1", "smam1-1"] #標準化対象データ列
target_data_dict = {}
for target_col in target_cols:
    target_data_dict[target_col] = []

target_scaler_dict = {}

#元のデータから削除する値
delete_cols = [
    "smamr-1", "smamr-5", "smamr-15", "smamr-30", "smamr-60"
]

test_db_no = 2
train_db_no = 3
host = "127.0.0.1"

start_train = datetime.datetime(2004, 1, 1)
end_train = datetime.datetime(2020, 12, 31)

start_test = datetime.datetime(2021, 1, 1)
end_test = datetime.datetime(2022, 5, 1)

start_end_str = start_train.strftime("%Y%m%d") + "-" + end_train.strftime("%Y%m%d")

start_stp_train = int(time.mktime(start_train.timetuple()))
end_stp_train = int(time.mktime(end_train.timetuple())) -1

start_stp_test = int(time.mktime(start_test.timetuple()))
end_stp_test = int(time.mktime(end_test.timetuple())) -1

redis_db_train = redis.Redis(host=host, port=6379, db=train_db_no, decode_responses=True)
redis_db_test = redis.Redis(host=host, port=6379, db=test_db_no, decode_responses=True)

redis_list = [
    {"r":redis_db_train, "start": start_stp_train, "end":end_stp_train},
    {"r":redis_db_test, "start": start_stp_test, "end":end_stp_test},
]

db_list = []
if term >= bet_term:
    for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
        db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))
else:
    db_list.append(symbol + "_" + str(term) + "_0")


print(db_list)

for db in db_list:
    print(db)

    result_data = redis_db_train.zrangebyscore(db, start_stp_train, end_stp_train, withscores=True)
    #print("result_data length:" + str(len(result_data)))

    for line in result_data:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)
        for target_col in target_cols:
            tmp_val = tmps.get(target_col)
            if tmp_val != None and np.isnan(tmp_val) == False:
                target_data_dict[target_col].append(tmp_val)

print("make scaler")

if type == "std":
    for target_col in target_cols:
        target_data_list = np.array(target_data_dict[target_col]).reshape(-1, 1)  # サンプル数, 特徴量の二次元配列にしなければならない
        scaler = StandardScaler()
        scaler.fit(target_data_list)
        target_scaler_dict[target_col] = scaler

        tmp_file_name = symbol + ".bt" + str(bet_term) + ".s" + str(term) + "." + target_col + "." + type + "." + start_end_str

        with open('/app/scaler/' + tmp_file_name, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
            pickle.dump(scaler, f)

        print(tmp_file_name)

elif type == "minmax":
    for target_col in target_cols:
        #target_data_list = [-200,200] #2016-202012までのデータではd1は最大81.69193503247519 最小-180.82256726324576なので-200 ~ 200で正規化
        target_data_list = np.array(target_data_dict[target_col]).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(target_data_list)
        target_scaler_dict[target_col] = scaler

        tmp_file_name = symbol + ".bt" + str(bet_term) + ".s" + str(term) + "." + target_col + "." + type + "." + start_end_str
        with open('/app/scaler/' + tmp_file_name, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
            pickle.dump(scaler, f)

        print(tmp_file_name)

        #with open('/app/scaler.pickle', 'rb') as f:
        #    scaler2 = pickle.load(f)


#データを標準化して保存する
for redis_db in redis_list:
    r = redis_db["r"]

    for db in db_list:
        print(db)

        result_data = r.zrangebyscore(db, redis_db["start"], redis_db["end"], withscores=True)
        print("result_data length:" + str(len(result_data)))

        for line in result_data:
            body = line[0]
            score = line[1]
            tmps = json.loads(body)

            for target_col in target_cols:
                scaler = target_scaler_dict[target_col]
                tmp_val = tmps.get(target_col)

                if tmp_val != None and np.isnan(tmp_val) == False:
                    tmp_val = np.array([tmp_val]).reshape(-1, 1)
                    tmp_val = scaler.transform(tmp_val)
                    tmp_val = tmp_val.flatten()[0]

                if type == "minmax":
                    tmps["mm~" + target_col] = tmp_val
                elif type == "std":
                    tmps["std~" + target_col] = tmp_val

            for col in delete_cols:
                if col in tmps.keys():
                    del tmps[col]

            #削除して改めて追加
            rm_cnt = r.zremrangebyscore(db, score, score)  # 削除した件数取得
            if rm_cnt != 1:
                # 削除できなかったらおかしいのでエラーとする
                print("cannot data remove!!!", score)
                exit()

            r.zadd(db, json.dumps(tmps), score)

