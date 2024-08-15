import random

import numpy as np
import tensorflow.keras.models
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
import logging.config
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from datetime import datetime
from datetime import timedelta
import time
from indices import index
from decimal import Decimal
from flask import Flask, request
import subprocess
import send_mail as m
from datetime import datetime
from datetime import date
from tensorflow.keras import initializers
from util import *
import talib

"""
nginxとflaskを使ってhttpによりAiの予想を呼び出す方式
systemctl start nginxでwebサーバを起動後、以下のコマンドによりuwsgiを起動し、localhost:80へアクセス
cat_binタイプのモデルを使用し、過去のモデルの予想結果とその過去からのレート変化を参考にベットするか決定する
"""

# uwsgi --ini /app/bin_op/uwsgi.ini

# tf.compat.v1.disable_eager_execution()

# ubuntuではGPU使わない
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
machine = "ub4"

model_dir = "/app/model/bin_op/"

cat_bin_both_models = {
    0: [
        "NZDJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT30-30-24_D-UNIT16-8-4_DROP0.0_SPREAD1_201201_202012_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1_IDL1_BS10240_SEED0_SHUFFLE-5",
        "NZDJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT30-30-24_D-UNIT16-8-4_DROP0.0_SPREAD1_201201_202012_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1_IDL1_BS10240_SEED0_SHUFFLE-21"
    ]
}
#for 0.585
border_list_both = {-1: [0.57, 0.55], 0: [0.56, 0.58],}


BET_TERM = 2
MAX_CLOSE_LEN = 3601
INPUT_LEN = {1:300, 2:300, 3:240,}
TERM_LEN = {1:2, 2:10, 3:30, }

#atrによって対象を絞り込む場合、範囲を-区切りでリストにする
ATR = ["0-7", ] #["0.5-1", "7-100"]
ATR_TERM = 3  #atr算出に使用するTERM_LENのキー
ATR_PERIOD = 5

SEC_OH_LEN = int(Decimal("60") / Decimal(str(BET_TERM)))
MIN_OH_LEN = 60
HOUR_OH_LEN = 24

#　トレード回数の調整　1なら調整なし、2なら1/2にする、3なら1/3にトレード回数を減らす　
# トレード回数が多くてハイローの制限にひっかからないようにするため
TRADE_DIVIDE = 1

SHIFT = {
    1: int(Decimal(str(TERM_LEN[1])) / Decimal(str(BET_TERM))),
    2: int(Decimal(str(TERM_LEN[2])) / Decimal(str(BET_TERM))),
    3: int(Decimal(str(TERM_LEN[3])) / Decimal(str(BET_TERM))),
#    4: int(Decimal(str(TERM_LEN[4])) / Decimal(str(BET_TERM))),
#    5: int(Decimal(str(TERM_LEN[5])) / Decimal(str(BET_TERM))),
}

def get_x(init_flg, closes=None):
    now = datetime.now()
    now_sec = now.second - int(now.second % BET_TERM)
    sec_oh_arr = [int(Decimal(str(now_sec)) / Decimal(str(BET_TERM)))]  # 2秒間隔データなら０から29に変換しなければならないのでbet_termで割る
    min_oh_arr = [now.minute]
    hour_oh_arr = [now.hour]

    retX = []
    for i in range(len(INPUT_LEN)):
        idx = i + 1

        X = np.zeros((1, INPUT_LEN[idx], 1))
        if init_flg:
            X[:, :, 0] = np.ones(INPUT_LEN[idx])
        else:
            x_tmp = []
            shift = SHIFT[idx]
            shift_idx = len(closes) - int(Decimal(str(INPUT_LEN[idx])) * Decimal(str(shift))) - 1
            while True:
                if shift_idx >= len(closes) - 1:
                    break

                divide_org = 1 if closes[shift_idx + shift] == closes[shift_idx] else closes[shift_idx + shift] / closes[shift_idx]
                divide = (divide_org -1) * 10000
                x_tmp.append(divide)

                shift_idx = shift_idx + shift

            X[:, :, 0] = x_tmp
        retX.append(X)

    retX.append(np.identity(SEC_OH_LEN)[sec_oh_arr])
    retX.append(np.identity(MIN_OH_LEN)[min_oh_arr])
    retX.append(np.identity(HOUR_OH_LEN)[hour_oh_arr])

    return retX

#指定のtermのclose,high,lowデータを取得
def get_hlc(closes, input_term_len, hl_flg=False):

    close_data = [] #INPUT_LEN の長さのデータになる
    high_data = [] #INPUT_LEN の長さのデータになる
    low_data = [] #INPUT_LEN の長さのデータになる
    shift = SHIFT[input_term_len]
    shift_idx = len(closes) - int(Decimal(str(INPUT_LEN[input_term_len])) * Decimal(str(shift))) - 1
    while True:
        if shift_idx + shift >= len(closes) :
            break

        close_data.append(closes[shift_idx + shift])
        #高値安値を取得する場合、終値をあつめていく
        if hl_flg:
            tmp_datas = []
            for j in range(shift):
                tmp_datas.append(closes[shift_idx + (j + 1)] )

            high_data.append(max(tmp_datas))
            low_data.append(min(tmp_datas))

        shift_idx = shift_idx + shift

    return close_data, high_data, low_data

def get_atr(close_data, high_data, low_data, period):
    close_np = np.array(close_data)
    high_np = np.array(high_data)
    low_np = np.array(low_data)
    atr_data = talib.ATR(high_np, low_np, close_np, timeperiod=period)

    target = get_divide_arr(close_data, (atr_data + close_np))

    return target

models = {}

app = Flask(__name__)

# 最初に一度推論させてグラフ作成し二回目以降の推論を早くする
for spread_tmp in cat_bin_both_models.keys():

    models_tmp = []
    err_flg = False
    for i, saved_file in enumerate(cat_bin_both_models[spread_tmp]):
        file_path = model_dir + saved_file

        if os.path.isdir(file_path):

            model_tmp = load_model(file_path)

            #model_tmp.summary()
            #print("load model")

            #start = time.time()

            res = model_tmp.predict(get_x(True), verbose=0, batch_size=1)

            print("init:",spread_tmp,i, res)
            #elapsed_time = time.time() - start
            #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            models_tmp.append(model_tmp)

        else:
            msg = "the Model not exists! " + saved_file
            print(msg)
            m.send_message("uwsgi " + machine + " ", msg)
            err_flg = True

    if err_flg == False:
        models[spread_tmp] = models_tmp


def do_predict(retX, spr):
    # print(retX.shape)
    #start = time.time()

    res_up = models[spr][0].predict_on_batch(retX)
    res_dw = models[spr][1].predict_on_batch(retX)

    res_str = [res_up[0][0], 0, res_dw[0][0]]
    res_str = np.array(res_str)
    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # print(res)

    # K.clear_session()


    return res_str


@app.route("/", methods=['GET', 'POST'])
def hello():
    data = request.get_json()
    # print(data)
    #print(datetime.fromtimestamp(1609362000))
    """
    for k, v in sorted(data.items()):
        print(k)
        print(v)
    """
    spr = int(data["spr"]) #spread
    res = 1 #same予想にしておく

    # トレード回数を調整する
    now = datetime.now()
    now_sec = now.second - int(now.second % BET_TERM)
    now_min = now.minute
    tmp_divide_min = now_min % TRADE_DIVIDE

    if now_sec % (TRADE_DIVIDE * BET_TERM) == (tmp_divide_min * BET_TERM):

        if spr in models:
            tmp_closes = data["vals"]
            closes = tmp_closes[:]
            if len(tmp_closes) != MAX_CLOSE_LEN:
                print("error!!! data length not correct:", len(tmp_closes), MAX_CLOSE_LEN)
                exit(1)
            if len(ATR) != 0:
                close_data, high_data, low_data = get_hlc(closes, ATR_TERM, True)

                tmp_atrs = get_atr(close_data, high_data, low_data, ATR_PERIOD)
                atr = tmp_atrs[-1]  # 一番最後のみ使用する
                print("atr", atr)
                if INPUT_LEN[ATR_TERM] != len(close_data) or INPUT_LEN[ATR_TERM] != len(high_data) or INPUT_LEN[ATR_TERM] != len(
                        low_data) or INPUT_LEN[ATR_TERM] != len(tmp_atrs):
                    print("error", len(close_data), len(high_data), len(low_data), len(tmp_atrs))
                    exit(1)

                ok_flg = False
                for t_atr in ATR:
                    min,max = t_atr.split("-")
                    min = float(min)
                    max = float(max)
                    if min <= atr and atr < max:
                        ok_flg = True
                        break

                if ok_flg == False:
                    #指定した範囲にATRの値がなければSAMEにしてリターン
                    #print("atr same", atr)
                    return str(1)

            predict_now = do_predict(get_x(False, closes), spr)
            #predict_old = do_predict(get_x(False, closes_old), spr)

            max_now = predict_now.argmax()
            probe_now = predict_now[max_now]

            if spr in border_list_both.keys():
                if max_now == 0 :
                    b = border_list_both[spr][0]
                    if probe_now >= b:
                        res = 0

                elif max_now == 2 :
                    b = border_list_both[spr][1]
                    if probe_now >= b:
                        res = 2

            """
            #ランダムに数百ミリ秒遅らせる
            if res in [0,2]:
                ran = random.randrange(10)
                time.sleep(ran * 0.1)
            """

    #print(res_str)
    return str(res)

if __name__ == "__main__":
    #app.run(port=5001) #for theoption
    app.run() #for highlow