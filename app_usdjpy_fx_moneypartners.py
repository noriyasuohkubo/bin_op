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
machine = "win4"

model_dir = "/app/model/bin_op/"

cat_bin_both_models = {
    0: [
        "USDJPY_LT3_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN194-7",
        "USDJPY_LT4_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN196-11"
    ]
}
#for 0.572
#border_list_both = {-1: [0.6, 0.6], 0: [0.6, 0.6],}

border_list_both = {-1: [0.6, 0.6], 0: [0.57, 0.57],}

BET_TERM = 2
MAX_CLOSE_LEN = 7201
INPUT_LEN = {1:300, 2:300, 3:240, 4:48 }
TERM_LEN = {1:2, 2:10, 3:60, 4:300}

#IND_COLS: ['10-satr-5', '10-satr-15', '10-satr-45', '60-satr-5', '60-satr-15', '60-satr-45', '300-satr-5', '300-satr-15', '300-satr-45']
#update:20230731 ind_range_bases: [['0-0.04'], ['0-0.04'], ['0-0.03'], ['0-0.13'], ['0-0.08'], ['0-0.08'], ['0-0.16'], ['0.02-0.19'], ['0.02-0.11']]
#indによって対象を絞り込む場合、範囲を-区切りでリストにする
INDS = [
    ["10-satr-5", ['0-0.04']],
    ["10-satr-15", ['0-0.04']],
    ["10-satr-45", ['0-0.03']],
    ["60-satr-5", ['0-0.13']],
    ["60-satr-15", ['0-0.08']],
    ["60-satr-45", ['0-0.08']],
    ["300-satr-5", ['0-0.16']],
    ["300-satr-15", ['0.02-0.19']],
    ["300-satr-45", ['0.02-0.11']],
]

#indの計算に必要な１番長い長さを保持
INDS_NEED_LEN = {}
tmp_need_lens = get_need_len()
for child in INDS:
    idx_term, idx_name , idx_len = child[0].split("-")
    idx_term = int(idx_term)

    tmp_need_len = tmp_need_lens[idx_name] + 1 + int(idx_len)
    if (idx_term in INDS_NEED_LEN.keys()) == False :
        INDS_NEED_LEN[idx_term] = tmp_need_len
    else:
        if INDS_NEED_LEN[idx_term] < tmp_need_len:
            INDS_NEED_LEN[idx_term] = tmp_need_len

for k in TERM_LEN.keys():
    if TERM_LEN[k] * INPUT_LEN[k] > (MAX_CLOSE_LEN -1)  * BET_TERM:
        print("error!!! data length not correct:", MAX_CLOSE_LEN)
        exit(1)

for tmp_a in INDS:
    idx_term, idx_name, idx_len= tmp_a[0].split("-")

    tmp_need_len = tmp_need_lens[idx_name] + 1 + int(idx_len)
    if int(idx_term) * tmp_need_len > (MAX_CLOSE_LEN -1)  * BET_TERM:
        print("error!!! atr length not correct:", MAX_CLOSE_LEN)
        exit(1)

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
    4: int(Decimal(str(TERM_LEN[4])) / Decimal(str(BET_TERM))),
    #5: int(Decimal(str(TERM_LEN[5])) / Decimal(str(BET_TERM))),
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
            if shift_idx < 0:
                print("shift_idx is minus", shift_idx)
                exit(1)

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
#def get_hlc(closes, input_term_len, hl_flg=False):
def get_hlc(closes, shift, i_len, hl_flg=False):
    close_data = [] #INPUT_LEN の長さのデータになる
    high_data = [] #INPUT_LEN の長さのデータになる
    low_data = [] #INPUT_LEN の長さのデータになる
    #shift = SHIFT[input_term_len]
    #shift_idx = len(closes) - int(Decimal(str(INPUT_LEN[input_term_len])) * Decimal(str(shift))) - 1
    shift_idx = len(closes) - int(Decimal(str(i_len)) * Decimal(str(shift))) - 1
    if shift_idx < 0:
        print("shift_idx is minus", shift_idx)
        exit(1)

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

def get_satr(close_data, high_data, low_data, period):
    close_np = np.array(close_data)
    high_np = np.array(high_data)
    low_np = np.array(low_data)
    atr_data = talib.ATR(high_np, low_np, close_np, timeperiod=period)

    target = get_sub_arr(close_np, (atr_data + close_np))

    return target

def get_rsi(close_data, period):
    close_np = np.array(close_data)
    target = talib.RSI(close_np, timeperiod=period)

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

            close_dates = {}
            high_dates = {}
            low_dates = {}
            for k,v in INDS_NEED_LEN.items():
                if (k in close_dates.keys()) == False:
                    #まだデータ取得していない場合のみ
                    ind_shift =int(Decimal(str(k)) / Decimal(str(BET_TERM)))
                    close_data, high_data, low_data = get_hlc(closes, ind_shift, v, True)
                    close_dates[k] = close_data
                    high_dates[k] = high_data
                    low_dates[k] = low_data
                    if v != len(close_data) or v != len(high_data) or v != len(low_data) :
                        print("error", len(close_data), len(high_data), len(low_data))
                        exit(1)

            bet_flg = True

            atr_list = []
            rsi_list = []
            if len(INDS) != 0:
                for j, tmp_a in enumerate(INDS):
                    ind_term, ind_name, ind_len = tmp_a[0].split("-")
                    ind_term = int(ind_term)
                    ind_len = int(ind_len)

                    ind_range = tmp_a[1]
                    if ind_name == "satr":
                        tmp_atrs = get_satr(close_dates[ind_term], high_dates[ind_term], low_dates[ind_term], ind_len)

                        atr = tmp_atrs[-1]  # 一番最後のみ使用する
                        if np.isnan(atr):
                            print("atr is nan")
                            exit(1)

                        atr_list.append(atr)
                        #print("atr", atr)

                        if len(ind_range) != 0:
                            # 値で絞る場合
                            ok_flg = False
                            for t_atr in ind_range:
                                min,max = t_atr.split("-")
                                min = float(min)
                                max = float(max)
                                if min <= atr and atr < max:
                                    ok_flg = True
                                    break

                            if ok_flg == False:
                                bet_flg = False

                    elif ind_name == "rsi":
                        tmp_rsis = get_rsi(close_dates[ind_term], ind_len)

                        rsi = tmp_rsis[-1]  # 一番最後のみ使用する
                        if np.isnan(rsi):
                            print("rsis is nan")
                            exit(1)

                        rsi_list.append(rsi)

                        if len(ind_range) != 0:
                            # 値で絞る場合
                            ok_flg = False
                            for t_rsi in ind_range:
                                min, max = t_rsi.split("-")
                                min = float(min)
                                max = float(max)
                                if min <= rsi and rsi < max:
                                    ok_flg = True
                                    break

                            if ok_flg == False:
                                bet_flg = False
                    else:
                        print("no idx setting")
                        exit(1)

            if bet_flg == True:
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
            else:
                res = 1

    #print(res_str)
    return str(res) + "-" + str(atr_list[0])

if __name__ == "__main__":
    app.run(port=7001) #thinkmarkets

