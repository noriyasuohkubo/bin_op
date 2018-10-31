import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
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
from datetime import datetime
from datetime import timedelta

# record number
# 2003 s1:14922000
# 2003 s30:493800
# 2018 s30:121200
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_count = 2

symbol = "EURUSD"
db_no = 3
train = True

start = datetime(2018, 1, 1)
start_score = int(time.mktime(start.timetuple()))

end = datetime(2000, 1, 1)
end_score = int(time.mktime(end.timetuple()))

maxlen = 600
pred_term = 6
rec_num = 1000000 + maxlen + pred_term + 1
# rec_num = 3600 * 24 * 3
epochs = 100
batch_size = 6144 * gpu_count

s = "30"

in_num=1

np.random.seed(0)

n_hidden = 20
n_hidden2 = 10
n_hidden3 = 5
n_hidden4 = 0


file_prefix = "multi_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4)

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir, "config", "config.ini")
history_file = os.path.join(current_dir,"history", file_prefix +"_history.csv")

config = configparser.ConfigParser()
config.read(ini_file)

SYMBOL_DB = json.loads(config['lstm']['SYMBOL_DB'])
MODEL_DIR = config['lstm']['MODEL_DIR']

model_file = os.path.join(MODEL_DIR, file_prefix +".hdf5")

def get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    start = time.time()
    #result = r.zrevrange(symbol, 0 + start_day, rec_num + start_day, withscores=False)
    result = r.zrevrangebyscore(symbol, start_score, end_score, start=0, num=rec_num + 1)
    # print(result)
    close_tmp, high_tmp, low_tmp = [], [], []
    ask_tmp, bid_tmp = [], []
    result.reverse()
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        close_tmp.append(tmps.get("close"))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))
        # ask_tmp.append(tmps.get("ask_volume"))
        # bid_tmp.append(tmps.get("bid_volume"))
        # print(tmps)
    # close = preprocessing.scale(np.array(close_tmp))
    # high = preprocessing.scale(np.array(high_tmp))
    # low = preprocessing.scale(np.array(low_tmp))
    # ask = preprocessing.scale(np.array(ask_tmp))
    # bid = preprocessing.scale(np.array(bid_tmp))

    close =  10000*np.log(close_tmp / shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 1000000 * (high_tmp / shift(high_tmp, 1, cval=np.NaN) - 1)[1:]
    #low = 1000000 * (low_tmp / shift(low_tmp, 1, cval=np.NaN) - 1)[1:]

    # data = pd.DataFrame(tmp)
    print(close_tmp[0:10])
    print(close[0:10])
    # print( pd.read_json(dum, orient='records'))

    # tmp = preprocessing.scale(data.ix[:, "close"])
    # elapsed_time = time.time() - start
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    close_data, high_data, low_data, label_data = [], [], [], []
    ask_data, bid_data = [], []

    up = 0
    same = 0
    data_length = len(close) - maxlen - pred_term - 1
    for i in range(data_length):

        close_data.append(close[i:(i + maxlen)])
        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])
        # ask_data.append(ask[i:(i + maxlen)])
        # bid_data.append(bid[i:(i + maxlen)])
        bef = close_tmp[1 + i + maxlen - 1]
        aft = close_tmp[1 + i + maxlen + pred_term - 1]
        # bef = data.ix[i + maxlen, "close"]
        # aft = data.ix[i+maxlen+pred_term, "close"]
        # print("val:", bef, aft)


        if bef < aft:
            # 上がった場合
            label_data.append([1, 0, 0])
            up = up + 1
        elif bef > aft:
            label_data.append([0, 0, 1])
        else:
            label_data.append([0, 1, 0])
            same = same + 1

    retX = np.array(close_data)
    #high_np = np.array(high_data)
    #low_np = np.array(low_data)
    #ask_np = np.array(ask_data)
    #bid_np = np.array(bid_data)
    #retX = np.zeros((data_length, maxlen, 3))
    #retX[:, :, 0] = close_np[:]
    #retX[:, :, 1] = high_np[:]
    #retX[:, :, 2] = low_np[:]
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


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


def one_or_zero(val):
    if val >= 0:
        return 1
    else:
        return 0


def mean_pred(y_true, y_pred):
    tmp = y_true * y_pred
    o = tf.constant(1, dtype=tf.float32)
    z = tf.constant(0, dtype=tf.float32)

    return K.mean(tf.map_fn(lambda x: tf.cond(tf.greater_equal(x[0], z), lambda: o, lambda: z), tmp))


def create_model(n_out=3):
    model = None

    with tf.device("/cpu:0"):
        model = Sequential()
        model.add(Dense(n_hidden,input_dim=maxlen))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_hidden2))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_hidden3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out))
        model.add(Activation('softmax'))

    return model


def get_model():
    model = None
    if os.path.isfile(model_file):
        model = load_model(model_file)
        print("Load Model")
    else:
        model = create_model()
    # model_gpu = model
    model_gpu = multi_gpu_model(model, gpus=gpu_count)
    model_gpu.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])

    return model, model_gpu

def do_train():
    model, model_gpu = get_model()
    early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            'lstm_{epoch:03d}_s' + s + '.hdf5'),
        save_best_only=False)
    train_X, test_X, train_Y, test_Y = get_train_data(symbol, rec_num, maxlen, pred_term, type, db_no)
    hist = model_gpu.fit(train_X, train_Y,
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=[CSVLogger(history_file),])

    # save model, not model_gpu
    # see http://tech.wonderpla.net/entry/2018/01/09/110000
    model.save(model_file)
    print('Model saved')

    do_predict(test_X, test_Y, model_gpu)


def do_predict(test_X, test_Y, model_gpu=None):
    if model_gpu is None:
        model_gpu = get_model()[1]
    total_num = len(test_Y)

    res = model_gpu.predict(test_X, verbose=0)
    print("Predict finished")

    Acc = np.mean(np.equal(res.argmax(axis=1), test_Y.argmax(axis=1)))
    print("Accuracy over ALL:", Acc)
    print("Total:", total_num)
    print("Correct:", total_num * Acc)

    ind5 = np.where(res >= 0.52)[0]
    x5 = res[ind5, :]
    y5 = test_Y[ind5, :]
    Acc5 = np.mean(np.equal(x5.argmax(axis=1), y5.argmax(axis=1)))
    print("Accuracy over 0.52:", Acc5)
    print("Total:", len(x5))
    print("Correct:", len(x5) * Acc5)

    ind55 = np.where(res >= 0.55)[0]
    x55 = res[ind55, :]
    y55 = test_Y[ind55, :]
    Acc55 = np.mean(np.equal(x55.argmax(axis=1), y55.argmax(axis=1)))
    print("Accuracy over 5.5:", Acc55)
    print("Total:", len(x55))
    print("Correct:", len(x55) * Acc55)

    ind6 = np.where(res >= 0.6)[0]
    x6 = res[ind6, :]
    y6 = test_Y[ind6, :]
    Acc6 = np.mean(np.equal(x6.argmax(axis=1), y6.argmax(axis=1)))
    print("Accuracy over 0.6:", Acc6)
    print("Total:", len(x6))
    print("Correct:", len(x6) * Acc6)

    K.clear_session()

    print("END")


if __name__ == '__main__':
    if train:
        do_train()
    else:
        # use EURUSD testdata
        test_X, test_Y = get_predict_data(symbol, 800000, maxlen, pred_term, type, 6)
        do_predict(test_X, test_Y)
