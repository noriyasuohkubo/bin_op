import numpy as np
import keras.models
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
import logging.config
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from keras.utils.training_utils import multi_gpu_model
import time
from indices import index
from decimal import Decimal

#symbol = "AUDUSD"
#symbol = "GBPJPY"
symbol = "EURUSD"

import_db_nos = {"ubuntu1":11,"ubuntu2":12,}

export_db_no = 8

export_host = "ubuntu2"
import_host = "127.0.0.1"

start = datetime(2018, 5, 16)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 5, 19)
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    import_db_no = import_db_nos.get(export_host)
    export_r = redis.Redis(host= export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host= import_host, port=6379, db=import_db_no)
    result_data = export_r.zrangebyscore(symbol, start_stp, end_stp, withscores=True)

    for line in result_data:
        body = line[0]
        score = line[1]
        imp = import_r.zrangebyscore(symbol , score, score)
        if len(imp) == 0:
            import_r.zadd(symbol, body, score)

    result_trade_data = export_r.zrangebyscore(symbol + "_TRADE", start_stp, end_stp, withscores=True)

    for line in result_trade_data:
        body = line[0]
        score = line[1]
        imp = import_r.zrangebyscore(symbol + "_TRADE" , score, score)
        if len(imp) == 0:
            import_r.zadd(symbol + "_TRADE", body, score)


if __name__ == "__main__":
    import_data()
