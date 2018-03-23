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

symbol ="EURUSD"
db_no = 0

maxlen = 200
pred_term = 3
rec_num = 10000 + maxlen + pred_term + 1

s = "10"
np.random.seed(0)
n_hidden =  15
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0
border = 0.52

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)
MODEL_DIR = config['lstm']['MODEL_DIR']

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

model_file = os.path.join(MODEL_DIR, "bydrop_in1_" + s + "_m" + str(maxlen) + "_hid1_" + str(n_hidden)
                          + "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) +".hdf5")

signal = ['UP','SAME','DOWN','SHORT']

def get_redis_data():
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    result = r.zrevrange(symbol, 0  , rec_num
                      , withscores=False)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    result.reverse()
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        close_tmp.append(tmps.get("close"))
        time_tmp.append(tmps.get("time"))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    #low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    close_data, high_data, low_data, label_data, time_data, price_data = [], [], [], [], [], []

    up =0
    same =0
    data_length = len(close) - maxlen - pred_term -1
    for i in range(data_length):

        close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])
        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])

        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]

        #正解をいれる
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
    time_np = np.array(time_data)
    price_np = np.array(price_data)
    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)
    #high_np = np.array(high_data)
    #low_np = np.array(low_data)

    retX = np.zeros((data_length, maxlen, 1))
    retX[:, :, 0] = close_np[:]
    #retX[:, :, 1] = high_np[:]
    #retX[:, :, 2] = low_np[:]

    retY = np.array(label_data)

    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ",up/len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))

    return retX, retY, price_np, time_np, close_tmp_np, time_tmp_np

def get_model():

    if os.path.isfile(model_file):
        model = load_model(model_file)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        print("Load Model")
        return model
    else:
        print("the Model not exists!")
        return None

def do_predict(test_X, test_Y):

    model = get_model()
    if model is None:
        return None

    total_num = len(test_Y)
    res = model.predict(test_X, verbose=0)
    print("Predict finished")

    K.clear_session()
    return res


if __name__ == "__main__":
    dataX, dataY, price_data, time_data, close, time = get_redis_data()
    res = do_predict(dataX,dataY)

    ind5 = np.where(res >=0.55)[0]
    x5 = res[ind5,:]
    y5= dataY[ind5,:]
    p5 = price_data[ind5]
    t5 = time_data[ind5]
    print(t5[0:10])
    Acc5 = np.mean(np.equal(x5.argmax(axis=1),y5.argmax(axis=1)))
    total_length = len(x5)
    print("Accuracy over 0.55:", Acc5)
    print("Total:", len(x5))
    print("Correct:", len(x5) * Acc5)

    up = res[:,0]
    down = res[:,2]
    up_ind5 = np.where(up >= 0.55)[0]
    down_ind5 = np.where(down >= 0.55)[0]

    x5_up = res[up_ind5,:]
    y5_up= dataY[up_ind5,:]
    p5_up = price_data[up_ind5]
    t5_up = time_data[up_ind5]
    up_total_length = len(x5_up)
    Acc5_up = np.mean(np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1)))
    up_cor_length = int(Acc5_up * up_total_length)
    up_wrong_length = int(up_total_length - up_cor_length)
    print("up_cor_length:"+ str(up_cor_length))
    print("up_wrong_length:" + str(up_wrong_length))

    x5_down = res[down_ind5,:]
    y5_down= dataY[down_ind5,:]
    p5_down = price_data[down_ind5]
    t5_down = time_data[down_ind5]
    down_total_length = len(x5_down)
    Acc5_down = np.mean(np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1)))
    down_cor_length = int(Acc5_down * down_total_length)
    down_wrong_length = int(down_total_length - down_cor_length)
    print("down_cor_length:"+ str(down_cor_length))
    print("down_wrong_length:" + str(down_wrong_length))

    cor_list_up_x, cor_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_cor_length) ]), np.ones(up_cor_length, dtype=np.float64)
    wrong_list_up_x, wrong_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_wrong_length) ]), np.ones(up_wrong_length, dtype=np.float64)
    cor_list_down_x, cor_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_cor_length) ]), np.ones(down_cor_length, dtype=np.float64)
    wrong_list_down_x, wrong_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_wrong_length) ]), np.ones(down_wrong_length, dtype=np.float64)

    money_x, money_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(len(time)) ]), np.ones(len(time), dtype=np.float64)
    money_tmp = {}

    money = 1000000
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

    cnt_up_cor = 0
    cnt_up_wrong = 0
    cnt_down_cor = 0
    cnt_down_wrong = 0
    for x,y,p,t in zip(x5,y5,p5,t5):
        #Up predict
        max = x.argmax()
        if max == 0:
            if max == y.argmax():
                money = money + 950
                cor_list_up_x[cnt_up_cor] = t
                cor_list_up_y[cnt_up_cor] = p
                cnt_up_cor = cnt_up_cor + 1
            else :
                money = money - 1000
                wrong_list_up_x[cnt_up_wrong] = t
                wrong_list_up_y[cnt_up_wrong] = p
                cnt_up_wrong = cnt_up_wrong + 1
        elif max == 2:
            if max == y.argmax():
                money = money + 950
                cor_list_down_x[cnt_down_cor] = t
                cor_list_down_y[cnt_down_cor] = p
                cnt_down_cor = cnt_down_cor + 1
            else:
                money = money - 1000
                wrong_list_down_x[cnt_down_wrong] = t
                wrong_list_down_y[cnt_down_wrong] = p
                cnt_down_wrong = cnt_down_wrong + 1

        money_tmp[t] = money
    print(cor_list_up_x[0:5])
    print(cor_list_up_y[0:5])
    prev_money = 1000000
    for i, ti in enumerate(time):
        if ti in money_tmp.keys():
            prev_money = money_tmp[ti]
        money_x[i] = ti
        money_y[i] = prev_money

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
    fig = plt.figure()
    #価格の遷移
    ax1 = fig.add_subplot(111)
    ax1.plot(time,close)

    ax1.plot(cor_list_up_x, cor_list_up_y, 'b^')
    ax1.plot(wrong_list_up_x, wrong_list_up_y, 'r^')
    ax1.plot(cor_list_down_x, cor_list_down_y, 'bv')
    ax1.plot(wrong_list_down_x, wrong_list_down_y, 'rv')

    ax2 = ax1.twinx()
    ax2.plot(money_x,money_y)

    plt.show()

