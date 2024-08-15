import pandas as pd
import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
import talib
import seaborn as sns
from matplotlib import pyplot as plt
from util import *

"""
特徴量から相関係数をもとめる
"""


host = "127.0.0.1"
symbol = "USDJPY"
bet_term = 5

org_term = 600

db_no = 2
db_name = symbol + "_" + str(bet_term) + "_IND" + str(org_term)

#相関係数を求める対象の特徴量
cols = [

    "600-satr-5",
    "600-satr-15",

]

pred_terms = [600,900, 1800,3600] #正解までの秒数

need_len = calc_need_len(cols, bet_term)
print("need_len:",need_len)

start = datetime.datetime(2016, 1, 1)
start_stp = int(time.mktime(start.timetuple()))
end = datetime.datetime(2022, 9, 1)

end_stp = int(time.mktime(end.timetuple())) -1

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

print(datetime.datetime.now())
print("START")
start_time = time.perf_counter()

print(db_name)
result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

for pred_term in pred_terms:

    pred_term_length = int(Decimal(str(pred_term)) / Decimal(str(bet_term)))
    print("pred_term_length", pred_term_length)

    real_cols = cols.copy()
    # 正解列を追加
    real_cols.append(str(pred_term) + "-" + "a")
    real_cols.append(str(pred_term) + "-" + "aa")

    cnt = 0
    skip_cnt = 0

    tmp_list = []  # あとでpandasのdataframeへ変換し相関係数を求める
    index_score = []  # scoreを入れていく

    for i, line in enumerate(result_data):
        cnt += 1
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        try:
            start_score = result_data[i - need_len][1]
            end_score = result_data[i + pred_term_length][1]

            if end_score != start_score + ((need_len + pred_term_length) * bet_term):
                #print(start_score, end_score, start_score + ((need_len + pred_term_length) * bet_term))
                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                skip_cnt += 1
                continue
        except Exception:
            # start_score end_scoreのデータなしなのでスキップ
            skip_cnt += 1
            continue

        list_child = []
        break_flg = False
        for col in real_cols:
            try:
                if np.isnan(tmps.get(col)):
                    #特徴量の値がnanである場合はデータなしなので飛ばす
                    break_flg = True
                    break
                elif tmps.get(col) == None:
                    break_flg = True
                    break
                else:
                    list_child.append(tmps.get(col))
            except Exception:
                print("col:", col)
                print(tmps)
        if break_flg:
            skip_cnt += 1
            continue
        else:
            tmp_list.append(list_child)
            index_score.append(score)

    #データ長があっているか確認
    if len(index_score) != len(tmp_list):
        print("length is wrong!!!",len(index_score),len(tmp_list))
        exit()

    print("org data length:", cnt)
    print("data length:", len(index_score))
    print("skip data length:", skip_cnt)

    df = pd.DataFrame(tmp_list,
                      index=index_score,
                      columns=real_cols)

    res=df.corr()   # 相関係数を求める pandasのDataFrameに格納される

    print(real_cols)

    print("index=" + str(pred_term) + "-"+"a")
    for index, item in zip(res.index,res.loc[str(pred_term) + "-"+"a"]):
        print(index,item)
    #print(res.loc[str(pred_term) + "-"+"a"])
    print("")

    print("index=" + str(pred_term) + "-"+"aa")
    for index, item in zip(res.index,res.loc[str(pred_term) + "-"+"aa"]):
        print(index,item)
    print("")

    #sns.heatmap(df.corr(),cmap= sns.color_palette('coolwarm', 10))
    #plt.show()

print(datetime.datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)