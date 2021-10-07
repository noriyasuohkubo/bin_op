import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
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
from indices import index
from decimal import Decimal
from DataSequence2 import DataSequence2
from readConf2 import *
import math
import sys

"""
あるモデルによって算出された予測値を予想時間(score)と共にDBに保存する
"""

db_no = 1
db_name_old = "GBPJPY_2_0"
db_name_new = "GBPJPY_2_0_NEW"

host = "127.0.0.1"
#DB名はモデル名とする
model_name = "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-6-3-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_LOSS-HUBER-90*18"


start = datetime(2020, 1, 1, )
end = datetime(2021, 6, 30, 23, 59, 59)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))

def make_predict():

    dataSequence2 = DataSequence2(0, start, end, True, False)

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    load_dir = "/app/model/bin_op/" + model_name

    model = tf.keras.models.load_model(load_dir)

    predict_list = model.predict_generator(dataSequence2,
                                           steps=None,
                                           max_queue_size=PROCESS_COUNT * 1,
                                           use_multiprocessing=True,
                                           verbose=0)

    redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

    print("predict length:", len(predict_list))

    predict_dict = {}

    #scoreをキーに予想値を辞書に一時的に格納
    for pred,  s,  in zip(predict_list,target_score_list,):
        predict_dict[s] = pred[0]

    old_data = redis_db.zrangebyscore(db_name_old, start_score, end_score, withscores=True)
    print("old data length:", len(old_data))


    cnt = 0
    for line in old_data:
        body = line[0]
        score = line[1]

        tmps = json.loads(body)
        child = {}

        #スプレッドがある本番データの場合
        if tmps.get("s") != None:
            # {"c": 142.77, "d": 0.35022589570443685, "t": "2020-01-06 19:56:08", "s": 1}
            #予想がある場合
            if score in predict_dict.keys():
                child = {'c': float(tmps.get("c")),
                         'd': float(tmps.get("d")),
                         't': tmps.get("t"),
                         's': int(tmps.get("s")),
                         'p': float(predict_dict[score])
                         }
            else:
                child = {'c': float(tmps.get("c")),
                         'd': float(tmps.get("d")),
                         't': tmps.get("t"),
                         's': int(tmps.get("s")),
                         }
        else:
            if score in predict_dict.keys():
                child = {'c': float(tmps.get("c")),
                         'd': float(tmps.get("d")),
                         't': tmps.get("t"),
                         'p': float(predict_dict[score])
                         }
            else:
                child = {'c': float(tmps.get("c")),
                         'd': float(tmps.get("d")),
                         't': tmps.get("t"),
                         }

        ret = redis_db.zadd(db_name_new, json.dumps(child), score)

        # もし登録できなかった場合
        if ret == 0:
            print(child)
            print(score)

        cnt += 1

        if cnt % 1000000 == 0:
            dt_now = datetime.now()
            print(dt_now, " ", cnt/len(old_data), "% 終了" )


if __name__ == "__main__":
    # 処理時間計測
    t1 = time.time()

    make_predict()

    print("END!!!")

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

