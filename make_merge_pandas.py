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
from datetime import datetime

"""
lgbm用のpandasデータをマージする
"""
start_time = time.perf_counter()

symbol = "USDJPY"

dates = [
    [datetime(2024, 6, 30), datetime(2024, 8, 10)],

]

for startDt, endDt in dates:
    #startDt = datetime(2017, 1, 1)
    #endDt = datetime(2023, 9, 3)
    print(startDt, endDt)

    start_score = int(time.mktime(startDt.timetuple()))
    end_score = int(time.mktime(endDt.timetuple()))

    end_tmp = endDt + timedelta(days=-1)

    mode = "pickle" #csv:csvだけ保存 pickle:pickleだけ保存
    csv_file_name = "/db2/lgbm/" + symbol + "/csv_file/tmp.csv"

    if mode == "csv" and csv_file_name == "":
        print("csv_file_name is needed")
        exit(1)

    bet_term = 2
    data_term = 2

    #leftのファイルを基本とする
    left_df_file = "MF218"
    left_df_file_path = "/db2/lgbm/" + symbol + get_lgbm_file_type(left_df_file) + left_df_file + ".pickle"

    right_df_file = "IF276"
    right_df_file_path = "/db2/lgbm/" + symbol + get_lgbm_file_type(right_df_file) + right_df_file + ".pickle"

    """
    #使用する列名
    left_input = ["score","o",]
    #left_tmp_input = [["2-d-" + str(i+1) for i in range(200)]]

    left_tmp_input = [
        "704-4-REG@704-4-REG-12@704-4-REG-4@704-4-REG-8@712-36-DW@712-36-DW-12@712-36-DW-4@712-36-DW-8@712-36-SAME@712-36-SAME-12@712-36-SAME-4@712-36-SAME-8@712-36-UP@712-36-UP-12@712-36-UP-4@712-36-UP-8@714-36-DW@714-36-DW-12@714-36-DW-4@714-36-DW-8@714-36-SAME@714-36-SAME-12@714-36-SAME-4@714-36-SAME-8@714-36-UP@714-36-UP-12@714-36-UP-4@714-36-UP-8@715-40-DW@715-40-DW-12@715-40-DW-4@715-40-DW-8@715-40-SAME@715-40-SAME-12@715-40-SAME-4@715-40-SAME-8@715-40-UP@715-40-UP-12@715-40-UP-4@715-40-UP-8".split("@")
    ]
    for t_l in left_tmp_input:
        left_input.extend(t_l)

    right_input = ["score"]
    right_tmp_input = [
        "716-26-DW@716-26-DW-12@716-26-DW-4@716-26-DW-8@716-26-SAME@716-26-SAME-12@716-26-SAME-4@716-26-SAME-8@716-26-UP@716-26-UP-12@716-26-UP-4@716-26-UP-8".split("@")
    ]
    for t_l in right_tmp_input:
        right_input.extend(t_l)
    """
    on_col = "score"

    #ファイル読み込み
    left_df = pd.read_pickle(left_df_file_path)

    print("left_df info")
    print(left_df.info())
    print("memory1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    # 開始、終了期間で絞る
    left_df.query('@start_score <= score < @end_score', inplace=True)
    print("memory1.1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    #del left_df_tmp
    gc.collect()
    print("memory1.2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    right_df = pd.read_pickle(right_df_file_path)
    print("right_df info")
    print(right_df.info())
    # 開始、終了期間で絞る
    right_df.query('@start_score <= score < @end_score', inplace=True)
    print("memory2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")


    print("left_df cnt ",len(left_df.index))
    print("right_df cnt ",len(right_df.index))


    #以下はメモリを食うのでコメントアウト
    """
    #カラム抽出
    left_df = left_df.loc[:, left_input]
    print("memory2.1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    right_df = right_df.loc[:, right_input]
    print("memory2.2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    """

    #マージするためにインデックスを一時的に削除
    left_df.reset_index(inplace=True, drop=True)
    print("memory2.3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    right_df.reset_index(inplace=True, drop=True)
    print("memory2.4", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    #マージ実施 left join.
    # 左側にはスコアデータが続いているmake_ind_db.pyなどで作成したpandasをもってくる
    left_df = pd.merge(left_df, right_df, on=on_col, how='left', copy=False)
    print("memory2.5", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    del right_df
    gc.collect()

    left_df.set_index("score", drop=False,inplace=True)
    print("memory2.6", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    left_df.sort_index(ascending=True,inplace=True)  # scoreの昇順　古い順にする
    print("memory2.7", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    print("merge finished")
    print("merge_df info")
    print(left_df.info())

    print("memory3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    print(left_df[:100])
    print(left_df[-100:])
    csv_regist_cols = left_df.columns.tolist()

    #oがleft,rightそれぞれにある場合
    if 'o_y' in csv_regist_cols:
        left_df.drop(columns=['o_y'], inplace=True)
    if 'o_x' in csv_regist_cols:
        left_df.rename(columns={'o_x': 'o'}, inplace=True)

    csv_regist_cols = left_df.columns.tolist()

    csv_regist_cols.sort()  # カラムを名前順にする
    # o,scoreは最後にしたいので削除後、追加
    if 'o' in csv_regist_cols:
        csv_regist_cols.remove('o')
        csv_regist_cols.append('o')
    if 'score' in csv_regist_cols:
        csv_regist_cols.remove('score')
        csv_regist_cols.append('score')

    print("cols", csv_regist_cols)
    left_df.sort_index(ascending=False,inplace=True)  # scoreで降順 新しい順にする

    if mode == "csv":

        if os.path.isfile(csv_file_name):
            print("file alerady exists")
            # すでにファイルがあるなら追記
            left_df.to_csv(csv_file_name, index=False, mode="a", header=False)
        else:
            print("file not exists")
            # 指定したパスが存在しない場合は新規作成、存在する場合は上書き
            left_df.to_csv(csv_file_name, index=False, mode="w")

        print("csv make finished")
        print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    elif mode == "pickle":

        input_name = list_to_str(csv_regist_cols, "@")
        merge_files = "_MFS-" + list_to_str([left_df_file, right_df_file])

        tmp_file_name = symbol + "_B" + str(bet_term) + "_D" + str(data_term) + "_IN-" + input_name + "_" + \
                        date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,
                                                                                  format='%Y%m%d') + merge_files + "_" + socket.gethostname()

        db_name_file = "MERGE_FILE_NO_" + symbol
        # win2のDBを参照してモデルのナンバリングを行う
        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
        if len(result) == 0:
            print("CANNOT GET MERGE_FILE_NO")
            exit(1)
        else:
            newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

            for line in result:
                body = line[0]
                score = line[1]
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

        tmp_path = "/db2/lgbm/" + symbol + "/merge_file/" + "MF" + str(newest_no)
        pickle_file_name = tmp_path + ".pickle"

        print("newest_no", newest_no)
        print("input_name", tmp_file_name)

        left_df.to_pickle(pickle_file_name)
        print("save pickle finished")
        print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    else:
        print("mode is incorrect")
        exit(1)

print(datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)