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


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
起動しているflaskに対して実際にclose値を渡して予想を取得するテストを行う
"""


redis_db = redis.Redis(host='127.0.0.1', port=6379, db=2, decode_responses=True)
db_name = 'USDJPY_1_0'


org_key_index = 1706752593
key_index = org_key_index - 12


while True:
    result_data = redis_db.zrangebyscore(db_name, key_index - (7230 * 1), key_index - 1, withscores=True)
    # print(len(result_data))
    closes = []

    for i, line in enumerate(result_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)
        closes.append(tmps["c"])

    json_data = {'score': key_index, 'vals': closes}

    t1 = time.time()
    response = requests.post("http://127.0.0.1:7001", json=json_data)

    print(response.text)
    print("経過時間：" + str(time.time() - t1))

    if key_index == org_key_index + 4:
        break

    key_index += 2
