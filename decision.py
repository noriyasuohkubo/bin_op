import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import initializers
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
import os
from keras.callbacks import CSVLogger
import configparser
from keras.callbacks import ModelCheckpoint
from scipy.ndimage.interpolation import shift
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import classification_report

# record number
# 2003 s1:14922000
# 2003 s30:493800
# 2018 s30:121200
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_count = 2
type = "category"
symbol = "EURUSD"
db_no = 3
train = True
start_day = 120 * 24 * 300 * 0

maxlen = 300
pred_term = 30
rec_num = 4000000 + maxlen + pred_term + 1
# rec_num = 3600 * 24 * 3
epochs = 1
batch_size = 2024 * gpu_count
s = "1"
np.random.seed(0)
n_hidden = 100
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0
current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir, "config", "config.ini")
history_file = os.path.join(current_dir, "history", "history.csv")

config = configparser.ConfigParser()
config.read(ini_file)

SYMBOL_DB = json.loads(config['lstm']['SYMBOL_DB'])
MODEL_DIR = config['lstm']['MODEL_DIR']

model_file = os.path.join(MODEL_DIR, "change_in1_" + s + "_m" + str(maxlen) + "_hid1_" + str(n_hidden)
                          + "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + ".hdf5")


def get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    start = time.time()
    result = r.zrevrange(symbol, 0 + start_day, rec_num + start_day
                         , withscores=False)
    # print(result)
    close_tmp, high_tmp, low_tmp = [], [], []
    ask_tmp, bid_tmp = [], []
    result.reverse()
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        close_tmp.append(tmps.get("close"))
        # high_tmp.append(tmps.get("high"))
        # low_tmp.append(tmps.get("low"))

    close = 10000 * np.log(close_tmp / shift(close_tmp, 1, cval=np.NaN))[1:]
    # high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    # low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    print(close_tmp[0:10])
    # print((np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN)))[0:10])
    print(close[0:10])

    close_data, high_data, low_data, label_data = [], [], [], []
    ask_data, bid_data = [], []

    up = 0
    same = 0
    data_length = len(close) - maxlen - pred_term - 1
    for i in range(data_length):
        close_data.append(close[i:(i + maxlen)])
        # high_data.append(high[i:(i + maxlen)])
        # low_data.append(low[i:(i + maxlen)])
        # ask_data.append(ask[i:(i + maxlen)])
        # bid_data.append(bid[i:(i + maxlen)])
        bef = close_tmp[1 + i + maxlen - 1]
        aft = close_tmp[1 + i + maxlen + pred_term - 1]
        # bef = data.ix[i + maxlen, "close"]
        # aft = data.ix[i+maxlen+pred_term, "close"]
        # print("val:", bef, aft)

        if type == "mean":
            label_data.append([close[i + maxlen + pred_term]])
        elif type == "category":
            if bef < aft:
                # 上がった場合
                label_data.append([0])
                up = up + 1
            elif bef > aft:
                label_data.append([1])
            else:
                label_data.append([2])
                same = same + 1
    retX = np.array(close_data)
    # high_np = np.array(high_data)
    # low_np = np.array(low_data)
    ask_np = np.array(ask_data)
    bid_np = np.array(bid_data)
    #retX = np.zeros((data_length, maxlen, 1))
    #retX[:, :, 0] = close_np[:]
    # retX[:, :, 1] = high_np[:]
    # retX[:, :, 2] = low_np[:]
    # retX[:, :, 3] = ask_np[:]
    # retX[:, :, 4] = bid_np[:]
    # retX = np.reshape(retX, (retX.shape[0], retX.shape[1],1))
    retY = np.array(label_data)
    # print("TYPE:", type(retX))
    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ", up / len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))
    # print("Y:", retY[0:30 ])
    # print("X:",retX[0][0:20])

    return retX, retY


'''
データの生成
'''


def get_train_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    start = time.time()
    X, Y = get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # データをランダム分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    return train_X, test_X, train_Y, test_Y


def get_predict_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    start = time.time()
    test_X, test_Y = get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return test_X, test_Y


def do_train():
    clf = DecisionTreeClassifier()
    train_X, test_X, train_Y, test_Y = get_train_data(symbol, rec_num, maxlen, pred_term, type, db_no)
    clf.fit(train_X, train_Y)

    # save model, not model_gpu
    pickle.dump(clf, open("model.pkl", "wb"))
    print('Model saved')

    do_predict(test_X, test_Y, clf)


def do_predict(test_X, test_Y, clf=None):
    if clf is None:
        clf = pickle.load(open("model.pkl", "rb"))

    pred = clf.predict(test_X)
    print("Predict finished")
    print (classification_report(test_Y, pred))


    print("END")


if __name__ == '__main__':
    if train:
        do_train()
    else:
        # use EURUSD testdata
        print("Predict start")
        test_X, test_Y = get_predict_data(symbol, 800000, maxlen, pred_term, type, 6)
        do_predict(test_X, test_Y)
