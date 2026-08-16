import json
import socket

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
from util import *
import  send_mail as mail

"""
標準化または正規化を行う
"""
def tmp_std(l, m, v):
    return (l - m) / math.sqrt(v)

std = np.vectorize(tmp_std, otypes=[np.ndarray])

def make_std():

    type = "std" #std or minmax

    symbol = "USDJPY"
    bet_term = 0.2
    term = 5 #抽出元のレコード間隔秒数
    shift = 1 #学習対象のレコード間隔秒

    target_col = "d1" #標準化対象データ列
    target_data_list = []

    target_scaler = None

    host = "localhost"
    target_db_no = 3
    start_target = datetime.datetime(2016, 1, 1)
    end_target = datetime.datetime(2023, 4, 1)

    start_end_str = start_target.strftime("%Y%m%d") + "-" + end_target.strftime("%Y%m%d")

    start_stp_target = int(time.mktime(start_target.timetuple()))
    end_stp_target = int(time.mktime(end_target.timetuple())) - 1

    tmp_file_name = symbol + "_bt" + str(bet_term) + "_t" + str(term) + "_s" + str(shift) + "_" + target_col + "_" + type + "_" + start_end_str

    print(tmp_file_name)

    # win2のDBを参照してモデルのナンバリングを行う
    db_name_file = "DATA_STD_" + symbol

    r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
    result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
    if len(result) == 0:
        print("CANNOT GET DATA_STD")
        exit(1)

    newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)
    print("newest_no:", newest_no)

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
        'no': newest_no,
    }

    r.zadd(db_name_file, json.dumps(child), newest_no)


    redis_db_target = redis.Redis(host=host, port=6379, db=target_db_no, decode_responses=True)

    db_list = make_db_list(symbol, term, bet_term)
    print(db_list)

    for db in db_list:
        print(db)

        result_data = redis_db_target.zrangebyscore(db, start_stp_target, end_stp_target, withscores=True)
        print("result_data length:", len(result_data))

        for line in result_data:
            body = line[0]
            score = line[1]
            if get_decimal_mod(score, shift) == 0.0:
                #対象のシフトのみ学習対象とする
                tmps = json.loads(body)
                tmp_val = tmps.get(target_col)
                if tmp_val != None and np.isnan(tmp_val) == False:
                    target_data_list.append(tmp_val)

        print("target_data_list lenght:", len(target_data_list))

    print("make scaler")

    if type == "std":
        target_data_list = np.array(target_data_list).reshape(-1, 1)  # サンプル数, 特徴量の二次元配列にしなければならない
        scaler = StandardScaler()
        scaler.fit(target_data_list)
        target_scaler = scaler

        print("平均",target_scaler.mean_, "分散", target_scaler.var_)

    elif type == "minmax":
        #target_data_list = [-200,200] #2016-202012までのデータではd1は最大81.69193503247519 最小-180.82256726324576なので-200 ~ 200で正規化
        target_data_list = np.array(target_data_list).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(target_data_list)
        target_scaler = scaler

    pickle_path = '/app/scaler/' + tmp_file_name

    with open(pickle_path, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
        pickle.dump(target_scaler, f)

    return pickle_path

def standarize(conf):
    print(conf)

    type = "std"

    symbol = "USDJPY"

    host = "127.0.0.1"

    pickle_path = conf["pickle_path"]
    bet_term = conf["bet_term"]
    term = conf["term"] #抽出元のレコード間隔秒数
    db_no = conf["db_no"]
    target_col = conf["target_col"] #標準化対象データ列

    start = conf["start_dt"]
    end = conf["end_dt"]

    #pickle_path = '/app/scaler/USDJPY_bt0.2_t0.2_s1_d1_std_20160101-20230401'
    pickle_bt = pickle_path.split("_")[1].split("bt")[1]
    pickle_t = pickle_path.split("_")[2].split("t")[1]

    if pickle_bt != str(bet_term):
        print("bet_term is incorrect",pickle_bt)

    if pickle_t != str(term):
        print("term is incorrect", pickle_t)

    with open(pickle_path, 'rb') as f:
        target_scaler = pickle.load(f)

    start_stp = int(time.mktime(start.timetuple()))
    end_stp = int(time.mktime(end.timetuple())) - 1

    redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

    db_list = make_db_list(symbol, term, bet_term)
    print(db_list)

    mean_ = target_scaler.mean_[0]
    var_ = target_scaler.var_[0]
    var_ = math.sqrt(var_) #標準偏差にする
    print("標準偏差:", var_)

    #データを標準化して保存する
    for db in db_list:
        print(db)

        result_data = redis_db.zrangebyscore(db, start_stp, end_stp, withscores=True)
        print("result_data length:" + str(len(result_data)))

        cnt = 0
        for line in result_data:
            if cnt % 1000000 == 0:
                print(datetime.datetime.now(), cnt)
            cnt += 1

            body = line[0]
            score = line[1]
            tmps = json.loads(body)

            tmp_val = tmps.get(target_col)

            if tmp_val != None and np.isnan(tmp_val) == False:
                #tmp_val = np.array([tmp_val]).reshape(-1, 1)
                #tmp_val = target_scaler.transform(tmp_val)

                # レート変化しない場合は0としたいので平均は0とする
                tmp_val = float(tmp_val / var_)

            if type == "minmax":
                tmps["mm~" + target_col] = tmp_val
            elif type == "std":
                tmps["std~" + target_col] = tmp_val

            del tmps[target_col] #元のカラムはデータ節約のため削除

            #削除して改めて追加
            rm_cnt = redis_db.zremrangebyscore(db, score, score)  # 削除した件数取得
            if rm_cnt != 1:
                # 削除できなかったらおかしいのでエラーとする
                print("cannot data remove!!!", score)
                exit()

            redis_db.zadd(db, json.dumps(tmps), score)


if __name__ == "__main__":
    host = socket.gethostname()

    #pickle_path = make_std()

    std_conf_list =[
        {
            "pickle_path" : "/app/scaler/USDJPY_bt0.2_t1_s1_d1_std_20160101-20230401",
            "start_dt" : datetime.datetime(2025, 4, 27),
            "end_dt": datetime.datetime(2025, 7, 26),
            "db_no" : 2,
            "bet_term" : 0.2,
            "term" : 1,
            "target_col" : "d1",
        },
        {
            "pickle_path": "/app/scaler/USDJPY_bt0.2_t5_s1_d1_std_20160101-20230401",
            "start_dt": datetime.datetime(2025, 4, 27),
            "end_dt": datetime.datetime(2025, 7, 26),
            "db_no": 2,
            "bet_term": 0.2,
            "term": 5,
            "target_col": "d1",
        },
    ]

    for conf in std_conf_list:
        standarize(conf)

    # 終わったらメールで知らせる
    mail.send_message(host, ": make_std finished!!!")