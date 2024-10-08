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
import pandas as pd
import lightgbm as lgb
from decimal import Decimal

"""
cat_binタイプのモデルを使用し、過去のモデルの予想結果とその過去からのレート変化を参考にベットするか決定する
"""
# ubuntuではGPU使わない
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
machine = "win7"

SYMBOL = "BTCUSD"

BASE_TERM = 2 #データの間隔
PREDICT_TERM = 2 #リクエストされる間隔

MAX_CLOSE_LEN = 7350 #渡されるcloseの長さ

SEC_OH_LEN = int(Decimal("60") / Decimal(str(BASE_TERM)))
MIN_OH_LEN = 60
HOUR_OH_LEN = 24

app = Flask(__name__)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_wma(value):
    # 加重移動平均
    weight = np.arange(len(value)) + 1
    wma = np.sum(weight * value) / weight.sum()

    return wma

def get_max(value):
    return np.max(value)

def get_min(value):
    return np.min(value)

def get_x(init_flg, score, data_length, input_datas, input_separate_flg, method, closes=None):
    dt = datetime.fromtimestamp(score)
    now_sec = dt.second
    sec_oh_arr = [int(Decimal(str(now_sec)) / Decimal(str(BASE_TERM)))]  # 2秒間隔データなら０から29に変換しなければならないのでBASE_TERMで割る
    min_oh_arr = [dt.minute]
    hour_oh_arr = [dt.hour]

    SEC_OH_LEN = int(Decimal("60") / Decimal(str(BASE_TERM)))
    MIN_OH_LEN = 60
    HOUR_OH_LEN = 24

    retX = []
    for i, dl in enumerate(data_length):
        s, len = dl

        if input_datas[i] == "d1":
            X = np.zeros((1, len, 1))
            if init_flg:
                X[:, :, 0] = np.ones(len)
            else:
                close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM))) ))

                aft = close_n[(-1 * len) -1:, -1]
                bef = np.roll(aft, 1) #1つ前にずらす
                target = get_divide_arr(bef, aft)[1:]
                X[:, :, 0] = target
            retX.append(X)

        elif input_datas[i] == "wmd1-1":
            X = np.zeros((1, len, 1))
            if init_flg:
                X[:, :, 0] = np.ones(len)
            else:
                close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM))) ))
                aft = np.apply_along_axis(get_wma, 1, close_n)[(-1 * len) -1:]
                bef = np.roll(aft, 1) #1つ前にずらす
                target = get_divide_arr(bef, aft)[1:]
                X[:, :, 0] = target
            retX.append(X)

        elif input_datas[i] == "ehd1-1_eld1-1":
            if input_separate_flg == True:
                X1 = np.zeros((1, len, 1))
                X2 = np.zeros((1, len, 1))
                if init_flg:
                    X1[:, :, 0] = np.ones(len)
                    X2[:, :, 0] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X1[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X2[:, :, 0] = target

                retX.append(X1)
                retX.append(X2)
            else:
                X = np.zeros((1, len, 2))
                if init_flg:
                    X[:, :, 0] = np.ones(len)
                    X[:, :, 1] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 1] = target

                retX.append(X)

        elif input_datas[i] == "d1_ehd1-1_eld1-1":
            if input_separate_flg == True:
                X1 = np.zeros((1, len, 1))
                X2 = np.zeros((1, len, 1))
                X3 = np.zeros((1, len, 1))
                if init_flg:
                    X1[:, :, 0] = np.ones(len)
                    X2[:, :, 0] = np.ones(len)
                    X3[:, :, 0] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))

                    aft = close_n[(-1 * len) - 1:, -1]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X1[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X2[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X3[:, :, 0] = target

                retX.append(X1)
                retX.append(X2)
                retX.append(X3)
            else:
                X = np.zeros((1, len, 3))

                if init_flg:
                    X[:, :, 0] = np.ones(len)
                    X[:, :, 1] = np.ones(len)
                    X[:, :, 2] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))

                    aft = close_n[(-1 * len) - 1:, -1]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 1] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 2] = target

                retX.append(X)


    if method == "LSTM7":
        retX.append(np.identity(SEC_OH_LEN)[sec_oh_arr])
        retX.append(np.identity(MIN_OH_LEN)[min_oh_arr])
        retX.append(np.identity(HOUR_OH_LEN)[hour_oh_arr])

    return retX

model_dir_lstm = "/app/model/bin_op/"
model_dir_lgbm = "/app/model_lgbm/bin_op/"

#lgbmモデル
lgbm_model_file = "MN679"
lgbm_model_file_suffix = 49

bst = lgb.Booster(model_file=model_dir_lgbm + lgbm_model_file)
#lgbmモデルの特徴量 DB(win2)に登録してあるモデル情報のinput値
INPUT_DATA = '482-38-DW@482-38-DW-102@482-38-DW-108@482-38-DW-114@482-38-DW-12@482-38-DW-120@482-38-DW-18@482-38-DW-24@482-38-DW-30@482-38-DW-36@482-38-DW-42@482-38-DW-48@482-38-DW-54@482-38-DW-6@482-38-DW-60@482-38-DW-66@482-38-DW-72@482-38-DW-78@482-38-DW-84@482-38-DW-90@482-38-DW-96@482-38-SAME@482-38-SAME-102@482-38-SAME-108@482-38-SAME-114@482-38-SAME-12@482-38-SAME-120@482-38-SAME-18@482-38-SAME-24@482-38-SAME-30@482-38-SAME-36@482-38-SAME-42@482-38-SAME-48@482-38-SAME-54@482-38-SAME-6@482-38-SAME-60@482-38-SAME-66@482-38-SAME-72@482-38-SAME-78@482-38-SAME-84@482-38-SAME-90@482-38-SAME-96@482-38-UP@482-38-UP-102@482-38-UP-108@482-38-UP-114@482-38-UP-12@482-38-UP-120@482-38-UP-18@482-38-UP-24@482-38-UP-30@482-38-UP-36@482-38-UP-42@482-38-UP-48@482-38-UP-54@482-38-UP-6@482-38-UP-60@482-38-UP-66@482-38-UP-72@482-38-UP-78@482-38-UP-84@482-38-UP-90@482-38-UP-96@582-19-DW@582-19-DW-102@582-19-DW-108@582-19-DW-114@582-19-DW-12@582-19-DW-120@582-19-DW-18@582-19-DW-24@582-19-DW-30@582-19-DW-36@582-19-DW-42@582-19-DW-48@582-19-DW-54@582-19-DW-6@582-19-DW-60@582-19-DW-66@582-19-DW-72@582-19-DW-78@582-19-DW-84@582-19-DW-90@582-19-DW-96@582-19-SAME@582-19-SAME-102@582-19-SAME-108@582-19-SAME-114@582-19-SAME-12@582-19-SAME-120@582-19-SAME-18@582-19-SAME-24@582-19-SAME-30@582-19-SAME-36@582-19-SAME-42@582-19-SAME-48@582-19-SAME-54@582-19-SAME-6@582-19-SAME-60@582-19-SAME-66@582-19-SAME-72@582-19-SAME-78@582-19-SAME-84@582-19-SAME-90@582-19-SAME-96@582-19-UP@582-19-UP-102@582-19-UP-108@582-19-UP-114@582-19-UP-12@582-19-UP-120@582-19-UP-18@582-19-UP-24@582-19-UP-30@582-19-UP-36@582-19-UP-42@582-19-UP-48@582-19-UP-54@582-19-UP-6@582-19-UP-60@582-19-UP-66@582-19-UP-72@582-19-UP-78@582-19-UP-84@582-19-UP-90@582-19-UP-96@667-5-DW@667-5-DW-102@667-5-DW-108@667-5-DW-114@667-5-DW-12@667-5-DW-120@667-5-DW-18@667-5-DW-24@667-5-DW-30@667-5-DW-36@667-5-DW-42@667-5-DW-48@667-5-DW-54@667-5-DW-6@667-5-DW-60@667-5-DW-66@667-5-DW-72@667-5-DW-78@667-5-DW-84@667-5-DW-90@667-5-DW-96@667-5-SAME@667-5-SAME-102@667-5-SAME-108@667-5-SAME-114@667-5-SAME-12@667-5-SAME-120@667-5-SAME-18@667-5-SAME-24@667-5-SAME-30@667-5-SAME-36@667-5-SAME-42@667-5-SAME-48@667-5-SAME-54@667-5-SAME-6@667-5-SAME-60@667-5-SAME-66@667-5-SAME-72@667-5-SAME-78@667-5-SAME-84@667-5-SAME-90@667-5-SAME-96@667-5-UP@667-5-UP-102@667-5-UP-108@667-5-UP-114@667-5-UP-12@667-5-UP-120@667-5-UP-18@667-5-UP-24@667-5-UP-30@667-5-UP-36@667-5-UP-42@667-5-UP-48@667-5-UP-54@667-5-UP-6@667-5-UP-60@667-5-UP-66@667-5-UP-72@667-5-UP-78@667-5-UP-84@667-5-UP-90@667-5-UP-96'.split("@")

"""
#d1をlgbmの特徴量とする場合
lgbm_ds =[
    {
        "data_length": 2,
        "data_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 101, 5)],
    }
]
"""

lgbm_ds = [] #d1をlgbmの特徴量としない場合

PAST_TERM_SEC = 6 #必要とするlstmモデルの過去分予想の間隔秒

# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 20

PAST_TERM = int(get_decimal_divide(PAST_TERM_SEC, PREDICT_TERM)) #必要とするlstmモデルの過去分予想のデータ間隔

PAST_LENGTH_LIST = 20 * PAST_TERM

#lstmのモデル
base_models =[
    {
        "name": 'BTCUSD_LT1_M7_LSTM1_B2_T10_I2-10-60_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.1_201601_202210_L-RATE0.001_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_ub2_MN667-5',
        "no": "667-5",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1","d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'BTCUSD_LT1_M7_LSTM1_B2_T10_I2-10-60_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.1_201805_202210_L-RATE0.001_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS5120_SD0_SHU1_ub2_MN482-38',
        "no": "482-38",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1", "d1", "d1" ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'BTCUSD_LT1_M7_LSTM1_B2_T10_I2-10-60_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV1_201601_202210_L-RATE0.001_LT1_ADAM_DA4_RA8_RRA9_d1-M1_OT-d_OD-c_IDL1_BS5120_SD0_SHU1_ub2_MN582-19',
        "no": "582-19",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1", "d1", "d1"],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
]

prev_score = 0

#モデルをロードしておく
models = {}

for base_model in base_models:
    model_tmp = load_model(model_dir_lstm + base_model["name"],custom_objects={"root_mean_squared_error": root_mean_squared_error,})
    #最初に一度推論させてグラフ作成し二回目以降の推論を早くする
    tmp_key_index = 1680775116 #適当なunixタイム
    res = model_tmp.predict_on_batch(get_x(True, tmp_key_index, base_model["data_length"], base_model["input_datas"], base_model["input_separate_flg"], base_model["method"], closes=None))

    print(base_model["no"], res)
    models[base_model["name"]] = model_tmp

#lstm予想を、モデルNoをキー、予想をリストにして保持
predict_dict= {}

for model in base_models:
    if model["type"] == "CATEGORY":
        predict_dict[model["no"] + "-UP"] = []
        predict_dict[model["no"] + "-SAME"] = []
        predict_dict[model["no"] + "-DW"] = []
    elif model["type"] == "REGRESSION":
        predict_dict[model["no"] + "-REG"] = []

def do_predict(score, closes):
    res = "0_100_0"
    #start = time.time()
    global  prev_score
    if score - prev_score > PREDICT_TERM :
        #もし最後のリクエストからPREDICT_TERM 以上経過していたら、予想が続かなくなるので、今までの予想結果を削除
        for model in base_models:
            if model["type"] == "CATEGORY":
                predict_dict[model["no"] + "-UP"] = []
                predict_dict[model["no"] + "-SAME"] = []
                predict_dict[model["no"] + "-DW"] = []
            elif model["type"] == "REGRESSION":
                predict_dict[model["no"] + "-REG"] = []
        print("delete predict_dict. lastscore:", prev_score)

    prev_score = score

    #lstmのモデルによる予想
    for model in base_models:
        model_tmp = models[model["name"]]
        x = get_x(False, score, model["data_length"], model["input_datas"], model["input_separate_flg"], model["method"], closes=closes)

        predict = model_tmp.predict_on_batch(x)

        if model["type"] == "CATEGORY":
            predict_dict[model["no"] + "-UP"].append(predict[0][0])
            predict_dict[model["no"] + "-SAME"].append(predict[0][1])
            predict_dict[model["no"] + "-DW"].append(predict[0][2])
        elif model["type"] == "REGRESSION":
            predict_dict[model["no"] + "-REG"].append(predict[0][0])

    # lgbmモデルの予想に必要なlstmモデルの過去分予想のリスト確認
    predict_len_ok_flg = True
    for no, predict_list in predict_dict.items():
        if len(predict_list) == PAST_LENGTH_LIST + 2:
            del predict_list[0] #1つ多いので最初を削除

        elif len(predict_list) > PAST_LENGTH_LIST + 2:
            print("error!!! predict_list length not correct:", len(predict_list))
            exit(1)

        elif len(predict_list) == PAST_LENGTH_LIST + 1:
            continue

        elif len(predict_list) < PAST_LENGTH_LIST + 1:
            #長さが足りない場合
            predict_len_ok_flg = False
            break

    #lgbmモデルの予想に必要なlstmモデルの過去分予想が溜まっていたらlgbm予想実施
    if predict_len_ok_flg == True:
        # lgbmモデルに渡す特徴量(pandas)を作成する元の列
        base_col_dict = {}
        for no, predict_list in predict_dict.items():
            predict_list_rev = predict_list[::-1]  # リストを逆順にする
            for i, p in enumerate(predict_list_rev):
                if i % PAST_TERM == 0:
                    if i == 0:
                        base_col_dict[no] = [p]
                    else:
                        base_col_dict[no + "-" + str(int(get_decimal_multi(i, PREDICT_TERM)))] = [p]
        #print(base_col_dict)
        if len(lgbm_ds) != 0:
            # d1をlgbmモデルの特徴量とする場合
            for i, ds in enumerate(lgbm_ds):
                data_length = ds["data_length"]
                data_idx = ds["data_idx"]

                close_n = np.reshape(closes, (-1, int(Decimal(str(data_length)) / Decimal(str(BASE_TERM)))))

                for idx in data_idx:
                    aft = close_n[-1]
                    bef = close_n[-1 - idx]
                    base_col_dict[str(data_length) + "-d-" + str(idx)] = get_divide_arr(bef, aft)


        x_df = pd.DataFrame(base_col_dict, index=pd.Index([score]))
        csv_regist_cols = x_df.columns.tolist()
        csv_regist_cols.sort()  # カラムを名前順にするx_df_dixt = x_df.to_dict(orient='index')

        tmp_dict = {}
        for col in csv_regist_cols:
            tmp_dict[col] = 'float32'

        # レート変化以外をfloat32に型変換
        x_df = x_df.astype(tmp_dict)

        # 訓練時と同じ列名の順番で特徴量を取得する
        # pandasデータをxとしてモデルに渡すが、ヘッダーは見ていないため、訓練時と同じ特徴量の順番にする必要がある(INPUT_DATAを指定することによりそのとおりの順番で抽出してくれる)
        x_df = x_df.loc[[score], INPUT_DATA]
        #print(x_df.info())

        predict_lgbm = bst.predict(x_df, num_iteration=int(lgbm_model_file_suffix))
        predict_lgbm = predict_lgbm[0]
        print(predict_lgbm)
        probe_up = predict_lgbm[0]
        probe_same = predict_lgbm[1]
        probe_dw = predict_lgbm[2]

        res = str(probe_up) + "_" +  str(probe_same) + "_" +  str(probe_dw)


    return res

@app.route("/", methods=['GET', 'POST'])
def hello():
    data = request.get_json()
    # print(data)

    score = data["score"]
    tmp_closes = data["vals"]
    closes = tmp_closes[:]

    if len(tmp_closes) != MAX_CLOSE_LEN:
        print("error!!! data length not correct:", len(tmp_closes), MAX_CLOSE_LEN)
        exit(1)

    res = do_predict(score, closes)
    print(res)
    return res

if __name__ == "__main__":
    app.run(port=7001) #thinkmarkets

