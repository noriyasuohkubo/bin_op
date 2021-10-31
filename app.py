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
machine = "amd3"


save_file = "/app/model/bin_op/" + \
            "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-6-3-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_LOSS-HUBER-90*18"

app = Flask(__name__)

if os.path.isdir(save_file):

    model = load_model(save_file)

    model.summary()
    #print("load model")

    # 最初に一度推論させてグラフ作成し二回目以降の推論を早くする
    retX = []
    X2 = np.zeros((1, 300, 1))
    X2[:, :, 0] = np.ones(300)
    retX.append(X2)

    X10 = np.zeros((1, 300, 1))
    X10[:, :, 0] = np.ones(300)
    retX.append(X10)

    X30 = np.zeros((1, 240, 1))
    X30[:, :, 0] = np.ones(240)
    retX.append(X30)

    X90 = np.zeros((1, 80, 1))
    X90[:, :, 0] = np.ones(80)
    retX.append(X90)

    X300 = np.zeros((1, 24, 1))
    X300[:, :, 0] = np.ones(24)
    retX.append(X300)

    res = model.predict(retX, verbose=0, batch_size=1)
    print("init!", res)

else:
    msg = "the Model not exists!"
    print(msg)
    m.send_message("uwsgi " + machine + " ", msg)


def do_predict(retX):
    # print(retX.shape)
    #start = time.time()

    res = model.predict_on_batch(retX)

    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # print(res)

    # K.clear_session()
    return res


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

    res = do_predict(retX)
    res_str = str(res[0][0])
    #print(res_str)

    #bef = 100.001
    #mid = float(Decimal(str(bef)) * (Decimal(res_str) / Decimal("10000") + Decimal("1")))
    #print(str(mid))

    return res_str

if __name__ == "__main__":
    app.run()

