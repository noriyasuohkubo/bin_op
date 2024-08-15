import json
import numpy as np
import redis
from datetime import datetime
import time
from decimal import Decimal, ROUND_HALF_UP
from util import *
from scipy.stats import norm
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import configparser
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
import logging.config
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd
import conf_class
from indices import index
from decimal import Decimal
from DataSequence2 import DataSequence2
import math
import sys
from silence_tensorflow import silence_tensorflow
silence_tensorflow() #ログ抑制 import tensorflowの前におく
import tensorflow as tf
import gc
from util import *
import socket

"""
賭けた時点から任意の秒数後の、賭けた時点からのレート変化と賭けた時点からの決済時点までのレート変化の統計を取る
例：２秒後に-0.05で賭けた時点からは-0.04になった

この統計の意味：
任意の秒数後でのレート変化に対して決済時までに回復が見込めない場合に損切りの指標にするため
"""

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def make_avg_stoploss(start, end):
    print(start, end)

    end = end + timedelta(days=-1)

    host = "127.0.0.1"

    db_no_new = 1

    bet_term = 2

    border_list = [0.55]

    term = 15 #決済までの時間をbet_termで割った数を設定する 決済30秒、bet_term2秒なら15

    points_tmp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    points =  [int(Decimal(str(p)) / Decimal(str(bet_term))) for p in points_tmp] #DBのレコード間隔に換算する

    #これ以上のレート変化は統計対象外
    target_range = 0.5

    model_type = "BIN" #モデルのタイプ:BOTH or BIN(UP or DW)
    up_dw_type = "UP" #UP or DWどちらの予想をDBに登録するか

    #DB名はモデル名とする
    model_name = "USDJPY_LT3_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN194-7"
    model_no = model_name.split("_MN")[1]

    load_dir = "/app/model/bin_op/" + model_name
    model = tf.keras.models.load_model(load_dir)


    conf = conf_class.ConfClass()
    conf.change_fx_real_spread_flg(False)
    conf.FX_TICK_DB = ""
    conf.DB_EVAL_NO = 2

    dataSequence2 = DataSequence2(conf, start, end, True, False)

    # 全close値のリスト
    close_list = dataSequence2.get_close_list()
    # 全score値のリスト
    score_list = dataSequence2.get_score_list()
    # 全spread値のリスト
    spread_list = dataSequence2.get_spread_list()
    # 全tick値のリスト
    tick_list = dataSequence2.get_tick_list()

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())


    predict_list = model.predict(dataSequence2,
                                           steps=None,
                                           max_queue_size=conf.PROCESS_COUNT * 1,
                                           use_multiprocessing=True,
                                           verbose=0)

    # 予想結果と予想時のスコアを辞書で保持
    if len(predict_list) != len(target_score_list):
        print("length of predict_list_ext and length of target_score_list_ext are not same:", len(predict_list),len(target_score_list))
        exit(1)

    score_pred_dict = dict(zip(target_score_list, predict_list))

    print("predict length:", len(predict_list))

    rate_change_dict = {}
    for border in border_list:
        for point in points:
            # レート変化をkey、決済時までの変化をvalue
            db_name_new = model_no + "_" + up_dw_type + "_" + str(border) + "_" + str(int(Decimal(str(point))) * Decimal(str(bet_term)))
            rate_change_dict[db_name_new] = {}

    print(rate_change_dict)

    for i, (sc, close, ) in enumerate(zip(score_list, close_list,  )):

        try:
            pred = score_pred_dict[sc]  # scoreをもとに予想を取得
        except Exception:
            #予想がないのでスキップ
            continue

        if model_type == "BOTH" and up_dw_type == "UP":
            pred = float(pred[0])  # up or dwの確率を格納
        elif model_type == "BOTH" and up_dw_type == "DW":
            pred = float(pred[2])  # up or dwの確率を格納
        elif model_type == "BIN":
            pred = float(pred[0])

        #print("pred",pred)

        for border in border_list:

            if pred < border:
                continue

            for point in points:
                db_name_new = model_no + "_" + up_dw_type + "_" + str(border) + "_" + str(int(Decimal(str(point))) * Decimal(str(bet_term)))
                #print("point:", point)
                try:
                    target_close = close_list[i + point]
                    target_deal_close = close_list[i + term]
                except Exception:
                    #データなしなのでスキップ
                    continue
                #新規発注してから指定秒後の変化
                change = float(Decimal(str(target_close)) - Decimal(str(close)))
                if abs(change) > target_range:
                    #レート変化が統計対象外
                    continue
                elif up_dw_type == "UP" and change >=0:
                    continue #勝っているので集計外
                elif up_dw_type == "DW" and change <=0:
                    continue #勝っているので集計外

                #統計結果が細かくならないように変化を集約する
                if change >= 0:
                    change = float(Decimal(str(change)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                else:
                    change = float(Decimal(str(abs(change))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) * -1

                if up_dw_type == "UP":
                    #利益を取得
                    change_deal = float(Decimal(str(target_deal_close)) - Decimal(str(close)))
                elif up_dw_type == "DW":
                    #利益を取得
                    change_deal = float(Decimal(str(target_deal_close)) - Decimal(str(close))) * -1

                if change in rate_change_dict[db_name_new].keys():
                    rate_change_dict[db_name_new][change].append(change_deal)
                else:
                    rate_change_dict[db_name_new][change] = [change_deal]


    redis_db = redis.Redis(host=host, port=6379, db=db_no_new, decode_responses=True)

    rate_change_dict_sort = sorted(rate_change_dict.items(),reverse=True)
    rate_change_dict_sort = dict((x, y) for x, y in rate_change_dict_sort)

    for k,v in rate_change_dict_sort.items():
        v_sort = sorted(v.items(),reverse=True)
        v_sort = dict((x, y) for x, y in v_sort)


        for k2, v2 in v_sort.items():
            v_np = np.array(v2)
            avg_pips = np.average(v_np)
            std = np.std(v_np)
            cnt = len(v_np)
            win_cnt = len(np.where(v_np >= 0)[0])
            win_rate = win_cnt / cnt

            """
            pct = 1 #k=0の時は1(100%)とする
            if k > 0:
                # SELLの場合
                pct = norm.cdf(x=k*-1, loc=avg, scale=std) #上がってしまっているレートが決済時に回復する確率(累積分布関数にて計算)
            elif k < 0:
                #BUYの場合
                pct = 1 - norm.cdf(x=k*-1, loc=avg, scale=std)
            """

            #print(k,avg, std, cnt)
            child = {
                "avg_pips":avg_pips,
                "win_cnt":win_cnt,
                "win_rate":win_rate,
                "cnt":cnt,
            }

            # 既存レコードがあるばあい、合算して追加
            tmp_val = redis_db.zrangebyscore(k, k2, k2,withscores=True)
            if len(tmp_val) == 1:
                body = tmp_val[0][0]
                tmps = json.loads(body)
                prev_win_cnt = int(tmps.get("win_cnt"))
                prev_avg_pips = float(tmps.get("avg_pips"))
                prev_cnt = int(tmps.get("win_cnt"))

                total_avg_pips = (prev_avg_pips*prev_cnt + avg_pips*cnt)/(prev_cnt + cnt)
                total_win_cnt = prev_win_cnt + win_cnt
                total_cnt = prev_cnt + cnt
                total_win_rate = total_win_cnt/total_cnt
                child = {
                    "avg_pips": total_avg_pips,
                    "win_cnt": total_win_cnt,
                    "win_rate": total_win_rate,
                    "cnt": total_cnt,
                }

                rm_cnt = redis_db.zremrangebyscore(k, k2, k2)  # 削除した件数取得
                if rm_cnt != 1:
                    # 削除できなかったらおかしいのでエラーとする
                    print("cannot remove!!!", k, k2)
                    exit()
            elif len(tmp_val) > 1:
                print("tmp_val over 1")
                exit(1)
            ret = redis_db.zadd(k, json.dumps(child), k2)

if __name__ == "__main__":

    dates = [
        [datetime(2022, 11, 1, ), datetime(2023, 8, 1, )],
    ]

    for dt in dates:
        start, end = dt
        make_avg_stoploss(start, end)