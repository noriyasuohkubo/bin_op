import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
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
from logging import getLogger

logger = getLogger(__name__)

#record number
#2003 s1:14922000
#2003 s30:493800
#2018 s30:121200
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_count = 2
type = "category"
symbol = "EURUSD"
db_no = 3
train = False
start_day = 120 * 24 * 300 * 0

maxlen = 200
pred_term = 3
rec_num = 2000000 + maxlen + pred_term + 1
#rec_num = 3600 * 24 * 3
epochs = 100
batch_size = 4048 * gpu_count
s = "10"
np.random.seed(0)
n_hidden =  15
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0
current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
history_file = os.path.join(current_dir,"history","history.csv")

config = configparser.ConfigParser()
config.read(ini_file)

SYMBOL_DB = json.loads(config['lstm']['SYMBOL_DB'])
MODEL_DIR = config['lstm']['MODEL_DIR']

model_file = os.path.join(MODEL_DIR, "bydrop_in1_" + s + "_m" + str(maxlen) + "_hid1_" + str(n_hidden)
                          + "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) +".hdf5")


def get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    start = time.time()
    result = r.zrevrange(symbol, 0 + start_day , rec_num + start_day
                      , withscores=False)
    #print(result)
    close_tmp, high_tmp, low_tmp = [], [], []
    ask_tmp, bid_tmp = [], []
    result.reverse()
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        close_tmp.append(tmps.get("close"))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))
        #ask_tmp.append(tmps.get("ask_volume"))
        #bid_tmp.append(tmps.get("bid_volume"))
        #print(tmps)
    #close = preprocessing.scale(np.array(close_tmp))
    #high = preprocessing.scale(np.array(high_tmp))
    #low = preprocessing.scale(np.array(low_tmp))
    #ask = preprocessing.scale(np.array(ask_tmp))
    #bid = preprocessing.scale(np.array(bid_tmp))

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    #low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    #data = pd.DataFrame(tmp)
    print(close_tmp[0:10])
    #print((np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN)))[0:10])
    print(close[0:10])
    #print( pd.read_json(dum, orient='records'))

    #tmp = preprocessing.scale(data.ix[:, "close"])
    #elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    close_data, high_data, low_data, label_data = [], [], [], []
    ask_data, bid_data = [], []

    up =0
    same =0
    data_length = len(close) - maxlen - pred_term -1
    for i in range(data_length):
        #close_data.append(preprocessing.scale(close[i:(i+maxlen)]))
        #high_data.append(preprocessing.scale(high[i:(i + maxlen)]))
        #low_data.append(preprocessing.scale(low[i:(i + maxlen)]))
        """
        c = close[i:(i + maxlen)]
        c[0] = 1
        h = high[i:(i + maxlen)]
        h[0] = 1
        l = low[i:(i + maxlen)]
        l[0] = 1
        close_data.append(c)
        high_data.append(h)
        low_data.append(l)
        """
        close_data.append(close[i:(i + maxlen)])
        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])
        #ask_data.append(ask[i:(i + maxlen)])
        #bid_data.append(bid[i:(i + maxlen)])
        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]
        #bef = data.ix[i + maxlen, "close"]
        #aft = data.ix[i+maxlen+pred_term, "close"]
        #print("val:", bef, aft)

        if type=="mean":
            label_data.append([close[i + maxlen +  pred_term]])
        elif type=="category":
            if bef < aft:
                #上がった場合
                label_data.append([1,0,0])
                up = up + 1
            elif bef > aft:
                label_data.append([0,0,1])
            else:
                label_data.append([0,1,0])
                same = same + 1
    close_np = np.array(close_data)
    #high_np = np.array(high_data)
    #low_np = np.array(low_data)
    ask_np = np.array(ask_data)
    bid_np = np.array(bid_data)
    retX = np.zeros((data_length, maxlen, 1))
    retX[:, :, 0] = close_np[:]
    #retX[:, :, 1] = high_np[:]
    #retX[:, :, 2] = low_np[:]
    #retX[:, :, 3] = ask_np[:]
    #retX[:, :, 4] = bid_np[:]
    #retX = np.reshape(retX, (retX.shape[0], retX.shape[1],1))
    retY = np.array(label_data)
    #print("TYPE:", type(retX))
    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ",up/len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))
    #print("Y:", retY[0:30 ])
    #print("X:",retX[0][0:20])

    return retX, retY


'''
データの生成
'''
def get_train_data(symbol, rec_num, maxlen, pred_term, type, db_no):
    start = time.time()
    X, Y = get_redis_data(symbol, rec_num, maxlen, pred_term, type, db_no)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #データをランダム分割
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
    if val >=0:
        return 1
    else:
        return 0

def mean_pred(y_true, y_pred):
    tmp = y_true * y_pred
    o = tf.constant(1,dtype=tf.float32)
    z = tf.constant(0,dtype=tf.float32)

    return K.mean(tf.map_fn(lambda x:  tf.cond(tf.greater_equal(x[0] , z),lambda :o,lambda :z) , tmp))

def create_model(n_in = 1, n_out = 3):
    model = None

    with tf.device("/cpu:0") :
        if type=="category":
            model = Sequential()
            """
            model.add(LSTM(n_hidden,input_shape=(maxlen, n_in)
                           , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
                           #,return_sequences = True))
            """
            model.add(Bidirectional(LSTM(n_hidden
                            ,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
                            #, return_sequences=True)
                           ,input_shape=(maxlen, n_in)
                            ))
            model.add(Dropout(0.1))
            """
            model.add(Bidirectional(LSTM(n_hidden,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
                           ))
            
            model.add(Dropout(0.1))
                           #,return_sequences = True))
            

            
            model.add(LSTM(n_hidden2,
                             kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
                             #, return_sequences = True))
            
            model.add(LSTM(n_hidden3,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
                           #, return_sequences=True))
                      
            model.add(LSTM(n_hidden4,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                           ))
            """
            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
            model.add(Activation('softmax'))

        elif type=='mean':
            model = Sequential()
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           input_shape=(maxlen, n_in)
                           , return_sequences=True))
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
                           ))
            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
            model.add(Activation('linear'))

    return model

def get_model():
    model = None
    if os.path.isfile(model_file):
        model = load_model(model_file, custom_objects={"mean_pred": mean_pred})
        print("Load Model")
    else:
        model = create_model()
    #model_gpu = model
    model_gpu = multi_gpu_model(model, gpus=gpu_count)
    if type == "category":
        model_gpu.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop', metrics=['accuracy'])
    elif type == 'mean':
        model_gpu.compile(loss='mean_squared_error',
                          optimizer="rmsprop", metrics=[mean_pred])
    return model, model_gpu



'''

モデル学習
'''

#epochs = 30
#batch_size = 128

'''
# バッチサイズに従って訓練データの数を調整、余ったデータはテストデータに足す
num = train_X.shape[0] - (train_X.shape[0] % batch_size)
print("NUM:", num)
train_x = train_X[:num,:,:]
train_y = train_Y[:num,:]
print("train_x.shape:", train_x.shape)
print("train_y.shape:", train_y.shape)

test_x = np.r_[train_X[num:,:,:], test_X]
test_y = np.r_[train_Y[num:,:], test_Y]
'''

#callbacks.append(CSVLogger("history.csv"))
# look
# https://qiita.com/yukiB/items/f45f0f71bc9739830002

def do_train():
    model,model_gpu = get_model()
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
                     callbacks=[early_stopping, CSVLogger(history_file)])

    '''
    学習の進み具合を可視化
    
    loss = hist.history['loss']
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss, label='loss', color='black')
    plt.xlabel('epochs')
    plt.show()
    #plt.savefig(__file__ + '.eps')
    '''
    #save model, not model_gpu
    #see http://tech.wonderpla.net/entry/2018/01/09/110000
    model.save(model_file)
    print('Model saved')

    do_predict(test_X, test_Y, model_gpu)

def do_predict(test_X, test_Y,model_gpu=None):

    if model_gpu is None:
        model_gpu = get_model()[1]
    total_num = len(test_Y)

    res = model_gpu.predict(test_X, verbose=0)
    print("Predict finished")

    Acc = np.mean(np.equal(res.argmax(axis=1),test_Y.argmax(axis=1)))
    print("Accuracy over ALL:", Acc)
    print("Total:", total_num)
    print("Correct:", total_num * Acc)

    ind5 = np.where(res >=0.52)[0]
    x5 = res[ind5,:]
    y5= test_Y[ind5,:]
    Acc5 = np.mean(np.equal(x5.argmax(axis=1),y5.argmax(axis=1)))
    print("Accuracy over 0.52:", Acc5)
    print("Total:", len(x5))
    print("Correct:", len(x5) * Acc5)

    ind55 = np.where(res >=0.55)[0]
    x55 = res[ind55,:]
    y55= test_Y[ind55,:]
    Acc55 = np.mean(np.equal(x55.argmax(axis=1),y55.argmax(axis=1)))
    print("Accuracy over 5.5:", Acc55)
    print("Total:", len(x55))
    print("Correct:", len(x55) * Acc55)

    """
    res5UP = res[:,0]
    print(res5UP[0:10])
    ind5UP = np.where(res5UP >=0.52)[0]
    print(ind5UP)
    print(len(ind5UP))
    x5UP = res[ind5UP,:]
    y5UP= test_Y[ind5UP,:]
    Acc5UP = np.mean(np.equal(x5UP.argmax(axis=1),y5UP.argmax(axis=1)))
    print("Accuracy over 0.52:", Acc5UP)
    print("Total:", len(ind5UP))
    print("Correct:", len(ind5UP) * Acc5UP)
    """

    ind6 = np.where(res >=0.6)[0]
    x6 = res[ind6,:]
    y6= test_Y[ind6,:]
    Acc6 = np.mean(np.equal(x6.argmax(axis=1),y6.argmax(axis=1)))
    print("Accuracy over 0.6:", Acc6)
    print("Total:", len(x6))
    print("Correct:", len(x6) * Acc6)

    ind7 = np.where(res >=0.7)[0]
    x7 = res[ind7,:]
    y7= test_Y[ind7,:]
    Acc7 = np.mean(np.equal(x7.argmax(axis=1),y7.argmax(axis=1)))
    print("Accuracy over 0.7:", Acc7)
    print("Total:", len(x7))
    print("Correct:", len(x7) * Acc7)

    """
    ind8 = np.where(res >=0.8)[0]
    x8 = res[ind8,:]
    y8= test_Y[ind8,:]
    Acc8 = np.mean(np.equal(x8.argmax(axis=1),y8.argmax(axis=1)))
    print("Accuracy over 0.8:", Acc8)
    print("Total:", len(x8))
    print("Correct:", len(x8) * Acc8)

    ind9 = np.where(res >=0.9)[0]
    x9 = res[ind9,:]
    y9= test_Y[ind9,:]
    Acc9 = np.mean(np.equal(x9.argmax(axis=1),y9.argmax(axis=1)))
    print("Accuracy over 0.9:", Acc9)
    print("Total:", len(x9))
    print("Correct:", len(x9) * Acc9)
    """
    K.clear_session()

    print("END")


if __name__ == '__main__':
    if train:
        do_train()
    else:
        # use EURUSD testdata
        print("Predict start")
        test_X,test_Y = get_predict_data(symbol, 800000, maxlen, pred_term, type, 6)
        do_predict(test_X,test_Y)
