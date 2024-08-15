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

# nginxとflaskを使ってhttpによりAiの予想を呼び出す方式
# systemctl start nginxでwebサーバを起動後、以下のコマンドによりuwsgiを起動し、localhost:80へアクセス
# uwsgi --ini /app/bin_op/uwsgi.ini

# tf.compat.v1.disable_eager_execution()

# ubuntuではGPU使わない
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
machine = "itl8"

FORMER_LIST = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,]

model_suffix ={
        0: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-7",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-35"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-7",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-35"
            ]
        ],
        1: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-16"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-13",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-16"
            ]
        ],
        2: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_EXCSEC-32-0_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-9",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-34"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_EXCSEC-2-30_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-48",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-5"
            ]
        ],
        3: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-6",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-29"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-5",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-30",
            ]
        ],
        4: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-6",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-8"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-8",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-23"
            ]
        ],
        5: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-20",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-24"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-15",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-27"
            ]
        ],
        6: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-21",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-36"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-24",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-34"
            ]
        ],
        7: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-27",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-7"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD8_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-22",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD8_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-8"
            ]
        ],
        8: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-2",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-23"
            ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-12",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-24"
            ]
        ],
        9: [
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-39",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32", ],
            [
                "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-39",
                "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32", ]
        ],

}



retX_tmp = []
X2 = np.zeros((1, 300, 1))
X2[:, :, 0] = np.ones(300)
retX_tmp.append(X2)

X10 = np.zeros((1, 300, 1))
X10[:, :, 0] = np.ones(300)
retX_tmp.append(X10)

X30 = np.zeros((1, 240, 1))
X30[:, :, 0] = np.ones(240)
retX_tmp.append(X30)

X90 = np.zeros((1, 80, 1))
X90[:, :, 0] = np.ones(80)
retX_tmp.append(X90)

X300 = np.zeros((1, 24, 1))
X300[:, :, 0] = np.ones(24)
retX_tmp.append(X300)


models = {}

app = Flask(__name__)

# 最初に一度推論させてグラフ作成し二回目以降の推論を早くする
for spread_tmp in model_suffix.keys():
    models[spread_tmp] = []
    for i, filegroup in enumerate(model_suffix[spread_tmp]):

        models_tmp = []
        err_flg = False

        for j, filename in enumerate(filegroup):
            filepath = "/app/model/bin_op/" + filename
            if os.path.isdir(filepath):

                model_tmp = load_model(filepath)

                #model_tmp.summary()
                #print("load model")

                #start = time.time()

                res = model_tmp.predict(retX_tmp, verbose=0, batch_size=1)

                print("init:",spread_tmp,i, res)
                #elapsed_time = time.time() - start
                #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                models_tmp.append(model_tmp)

            else:
                msg = "the Model not exists! " + filepath
                print(msg)
                m.send_message("uwsgi " + machine + " ", msg)
                err_flg = True

        if err_flg == False:
            models[spread_tmp].append(models_tmp)

def do_predict(retX, spr, sec):
    # print(retX.shape)
    #start = time.time()

    if spr in models:

        # 予想するときの秒数によってFORMER,LATTERどちらから予想を取得するか決める
        if sec in FORMER_LIST:
            res_up = models[spr][0][0].predict_on_batch(retX)
            res_dw = models[spr][0][1].predict_on_batch(retX)
        else:
            res_up = models[spr][1][0].predict_on_batch(retX)
            res_dw = models[spr][1][1].predict_on_batch(retX)

        res_str = str(res_up[0][0]) + "," + "0" + "," + str(res_dw[0][0])

    else:
        res_str = "ERROR"
    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #print(res)

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


    spr = data["spr"] #spread
    sec = data["sec"]
    vals2 = data["vals2"]
    vals10 = data["vals10"]
    vals30 = data["vals30"]
    vals90 = data["vals90"]
    vals300 = data["vals300"]

    # close = 10000 * np.log(closes / shift(closes, 1, cval=np.NaN))[1:]
    retX = []
    X2 = np.zeros((1, 300, 1))
    X2[:, :, 0] = vals2[:]
    retX.append(X2)

    X10 = np.zeros((1, 300, 1))
    X10[:, :, 0] = vals10[:]
    retX.append(X10)

    X30 = np.zeros((1, 240, 1))
    X30[:, :, 0] = vals30[:]
    retX.append(X30)

    X90 = np.zeros((1, 80, 1))
    X90[:, :, 0] = vals90[:]
    retX.append(X90)

    X300 = np.zeros((1, 24, 1))
    X300[:, :, 0] = vals300[:]
    retX.append(X300)

    res_str = do_predict(retX, spr, sec)

    print(res_str)
    return res_str

if __name__ == "__main__":
    app.run()

