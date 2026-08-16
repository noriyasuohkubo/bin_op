from pathlib import Path
import numpy as np
from datetime import datetime
import time
from decimal import Decimal
import redis
import json
import random
import scipy.stats
import gc
import math
from subprocess import Popen, PIPE
import pandas as pd
import lightgbm as lgb
from tensorflow.keras.models import load_model
from util import *
from tensorflow.keras import backend as K
import requests
from app_usdjpy_fx_predict30_lgbm_2009_class import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
実際にclose値を渡して予想を取得するテストを行う

test_lgbm_flask_usdjpy_predict.pyで予め指定日時を決めておく
"""


redis_db = redis.Redis(host='win2', port=6379, db=0, decode_responses=True)
#redis_db = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

loop_cnt = 10
org_key_index = 1766722270
key_index = org_key_index - PAST_TERM_SEC * PAST_LENGTH

predict_class = AppUsdjpyFxPredict30Lgbm2009Class(mode="test")

while True:
    result_data = redis_db.zrangebyscore(db_name, key_index - (MAX_CLOSE_LEN -1), key_index, withscores=True)
    #print(len(result_data))
    closes = []

    for i, line in enumerate(result_data):
        body = line[0]
        score = float(line[1])
        tmps = json.loads(body)
        ask = float(tmps["ask"])
        bid = float(tmps["bid"])
        closes.append(get_decimal_divide(get_decimal_add(ask, bid),2))
        #closes.append(tmps["c"])

    t1 = time.time()
    """
    json_data = {'score': key_index, 'vals': closes}
    response = requests.post("http://127.0.0.1:8030", json=json_data)
    """

    response = predict_class.get_predict(key_index, closes)

    print("key_index", key_index, response, "経過時間：" + str(time.time() - t1))

    if key_index == get_decimal_add(org_key_index, loop_cnt):
        break

    key_index =  get_decimal_add(key_index,LOOP_TERM)
