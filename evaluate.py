import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import redis
import traceback
import json
import time
import pandas as pd
from sklearn import preprocessing
from keras.models import load_model
from keras import backend as K
import configparser

np.random.seed(0)

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)

SYMBOL_DB = json.loads(config['lstm']['SYMBOL_DB'])
MODEL_DIR = config['lstm']['MODEL_DIR']
model_file = os.path.join(MODEL_DIR, "lstm3600.hdf5")

def get_db_no(db_name):
    return SYMBOL_DB[db_name]

def get_redis_data(symbol, maxlen):
    r = redis.Redis(host='localhost', port=6379, db=get_db_no(symbol))
    start = time.time()
    result = r.zrevrange(symbol, 0, (maxlen -1), withscores=False)
    #print(result)
    tmp =[]
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        tmp.append(tmps.get("close"))
        #print(tmps)
    raw = np.array(tmp)
    data_x = preprocessing.scale(raw)
    print("data_x.shape:", data_x.shape)

    #retX = np.array(close_data)
    retX = np.reshape(data_x, (1, data_x.shape[0],1))
    #print("TYPE:", type(retX))
    print("X SHAPE:", retX.shape)
    """
    for i in range(1):
        print(retX[i])
    """
    return retX


'''
Get Redis Data
'''
maxlen = 3600

start = time.time()
data_x = get_redis_data("EURUSD", maxlen)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

'''
Load model & Predict
'''
model = None

if os.path.isfile(model_file):
    model = load_model(model_file)
    result = model.predict(data_x)

    print("RESULT TYPE:", type(result))
    print("RESULT:", result)

    K.clear_session()
else:
    print("model_file not found")

print("END")
