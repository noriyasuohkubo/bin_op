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
symbol = "GBPJPY"

db_no = 3
gpu_count = 2
maxlen = 300
pred_term = 3
rec_num = 10000 + maxlen + pred_term + 1
batch_size = 8192 * gpu_count

start = datetime(2018, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 3, 24)
end_stp = int(time.mktime(end.timetuple()))

s = "10"
suffix = ""

except_index = False
except_highlow = True

drop = 0.1
np.random.seed(0)
n_hidden =  25
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

border = 0.52
payout = 950

spread = 2

fx = False
fx_position = 10000
fx_spread = 1

in_num=1

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)
MODEL_DIR = config['lstm']['MODEL_DIR']

type = 'bydrop'

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

model_file = os.path.join(MODEL_DIR, file_prefix +".hdf5" + suffix)

def get_redis_data():
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    result = r.zrangebyscore(symbol, start_stp, end_stp, withscores=False)
    #result = r.zrevrange(symbol, 0  , rec_num  , withscores=False)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    print(result[0:5])
    #result.reverse()
    #print(index)
    indicies = np.ones(len(index), dtype=np.int32)
    #経済指標発表前後2時間は予想対象からはずす
    for i,ind in enumerate(index):
        tmp_datetime = datetime.strptime(ind, '%Y-%m-%d %H:%M:%S')
        indicies[i] = int(time.mktime(tmp_datetime.timetuple()))

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
    print("data_length:" + str(data_length))

    for i in range(data_length):
        continue_flg = False

        if except_index:
            tmp_datetime = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
            score = int(time.mktime(tmp_datetime.timetuple()))
            for ind in indicies:
                ind_datetime = datetime.fromtimestamp(ind)

                bef_datetime = ind_datetime - timedelta(hours=1)
                aft_datetime = ind_datetime + timedelta(hours=1)
                bef_time = int(time.mktime(bef_datetime.timetuple()))
                aft_time = int(time.mktime(aft_datetime.timetuple()))

                if bef_time <= score and score <= aft_time:
                    continue_flg = True
                    break;

            if continue_flg:
                continue;
        #ハイローオーストラリアの取引時間外を学習対象からはずす
        except_list = [20, 21, 22]
        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue;
        close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])
        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])

        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]

        #正解をいれる
        if float(Decimal(str(aft)) - Decimal(str(bef))) >= 0.00001 * spread:
            #上がった場合
            label_data.append([1,0,0])
            up = up + 1
        elif  float(Decimal(str(bef)) - Decimal(str(aft))) >= 0.00001 * spread:
            label_data.append([0,0,1])
        else:
            label_data.append([0,1,0])
            same = same + 1
        """
        if bef < aft:
            #上がった場合
            label_data.append([1,0,0])
            up = up + 1
        elif bef > aft:
            label_data.append([0,0,1])
        else:
            label_data.append([0,1,0])
            same = same + 1
        """
    close_np = np.array(close_data)
    time_np = np.array(time_data)
    price_np = np.array(price_data)
    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)

    #high_np = np.array(high_data)
    #low_np = np.array(low_data)

    retX = np.zeros((len(close_np), maxlen, in_num))
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
        model_gpu = multi_gpu_model(model, gpus=gpu_count)
        model_gpu.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        print("Load Model")
        return model_gpu
    else:
        print("the Model not exists!")
        return None

def do_predict(test_X, test_Y):

    model = get_model()
    if model is None:
        return None

    total_num = len(test_Y)
    res = model.predict(test_X, verbose=0, batch_size=batch_size)
    print("Predict finished")

    K.clear_session()
    return res


if __name__ == "__main__":
    dataX, dataY, price_data, time_data, close, time = get_redis_data()
    res = do_predict(dataX,dataY)

    ind5 = np.where(res >=border)[0]
    x5 = res[ind5,:]
    y5= dataY[ind5,:]
    p5 = price_data[ind5]
    t5 = time_data[ind5]
    print(t5[0:10])

    Acc = np.mean(np.equal(res.argmax(axis=1),dataY.argmax(axis=1)))
    print("Accuracy over ALL:", Acc)
    print("Total:", len(dataY))
    print("Correct:", len(dataY) * Acc)


    Acc5 = np.mean(np.equal(x5.argmax(axis=1),y5.argmax(axis=1)))
    total_length = len(x5)
    print("Accuracy over " + str(border) + ":", Acc5)
    print("Total:", len(x5))
    print("Correct:", len(x5) * Acc5)

    ind53 = np.where(res >=0.53)[0]
    x53 = res[ind53,:]
    y53= dataY[ind53,:]
    Acc53 = np.mean(np.equal(x53.argmax(axis=1),y53.argmax(axis=1)))
    print("Accuracy over 0.53:", Acc53)
    print("Total:", len(x53))
    print("Correct:", len(x53) * Acc53)

    ind54 = np.where(res >=0.54)[0]
    x54 = res[ind54,:]
    y54= dataY[ind54,:]
    Acc54 = np.mean(np.equal(x54.argmax(axis=1),y54.argmax(axis=1)))
    print("Accuracy over 0.54:", Acc54)
    print("Total:", len(x54))
    print("Correct:", len(x54) * Acc54)

    ind55 = np.where(res >=0.55)[0]
    x55 = res[ind55,:]
    y55= dataY[ind55,:]
    Acc55 = np.mean(np.equal(x55.argmax(axis=1),y55.argmax(axis=1)))
    print("Accuracy over 0.55:", Acc55)
    print("Total:", len(x55))
    print("Correct:", len(x55) * Acc55)

    ind56 = np.where(res >=0.56)[0]
    x56 = res[ind56,:]
    y56= dataY[ind56,:]
    Acc56 = np.mean(np.equal(x56.argmax(axis=1),y56.argmax(axis=1)))
    print("Accuracy over 0.56:", Acc56)
    print("Total:", len(x56))
    print("Correct:", len(x56) * Acc56)

    ind57 = np.where(res >=0.57)[0]
    x57 = res[ind57,:]
    y57= dataY[ind57,:]
    Acc57 = np.mean(np.equal(x57.argmax(axis=1),y57.argmax(axis=1)))
    print("Accuracy over 0.57:", Acc57)
    print("Total:", len(x57))
    print("Correct:", len(x57) * Acc57)

    """
    tmp_pred = res.argmax(axis=1)
    tmp_pred_up_ind = np.where(tmp_pred == 0)[0]
    tmp_pred_down_ind = np.where(tmp_pred == 2)[0]
    up = res[:, 0]
    down = res[:, 2]
    tmp_up_ind5 = np.where(up >= border)[0]
    tmp_down_ind5 = np.where(down >= border)[0]
    up_ind5 = []
    down_ind5 = []
    for u in tmp_up_ind5:
        if u in tmp_pred_up_ind:
            up_ind5.append(u)
    for d in tmp_down_ind5:
        if d in tmp_pred_down_ind:
            down_ind5.append(d)
    """
    up = res[:, 0]
    down = res[:, 2]
    up_ind5 = np.where(up >= border)[0]
    down_ind5 = np.where(down >= border)[0]

    x5_up = res[up_ind5,:]
    y5_up= dataY[up_ind5,:]
    p5_up = price_data[up_ind5]
    t5_up = time_data[up_ind5]
    up_total_length = len(x5_up)
    up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
    #Acc5_up = np.mean(np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1)))
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    up_wrong_length = int(up_total_length - up_cor_length)
    print("up_cor_length:"+ str(up_cor_length))
    print("up_wrong_length:" + str(up_wrong_length))

    x5_down = res[down_ind5,:]
    y5_down= dataY[down_ind5,:]
    p5_down = price_data[down_ind5]
    t5_down = time_data[down_ind5]
    down_total_length = len(x5_down)
    #Acc5_down = np.mean(np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1)))
    down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
    down_cor_length = int(len(np.where(down_eq == True)[0]))
    down_wrong_length = int(down_total_length - down_cor_length)
    print("down_cor_length:"+ str(down_cor_length))
    print("down_wrong_length:" + str(down_wrong_length))

    cor_list_up_x, cor_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_cor_length) ]), np.ones(up_cor_length, dtype=np.float64)
    wrong_list_up_x, wrong_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_wrong_length) ]), np.ones(up_wrong_length, dtype=np.float64)
    cor_list_down_x, cor_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_cor_length) ]), np.ones(down_cor_length, dtype=np.float64)
    wrong_list_down_x, wrong_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_wrong_length) ]), np.ones(down_wrong_length, dtype=np.float64)

    money_x, money_y = np.array([ "00:00:00" for i in range(len(time)) ]), np.ones(len(time), dtype=np.float64)
    money_tmp = {}

    money = 1000000
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

    cnt_up_cor = 0
    cnt_up_wrong = 0
    cnt_down_cor = 0
    cnt_down_wrong = 0
    loop_cnt = 0
    for x,y,p,t in zip(x5,y5,p5,t5):

        max = x.argmax()
        if max == 0:
            # Up predict
            if fx:
                buy = p * fx_position
                tmp_buy = price_data[ind5[loop_cnt]]
                if p != tmp_buy:
                    print("ERROR")

                sell = price_data[ind5[loop_cnt] + pred_term] * fx_position
                # DB上の値は実際の1／100なので100倍している
                profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * fx_spread * fx_position)))) * 100
                money = money + profit
            if max == y.argmax():
                if fx == False:
                    money = money + payout
                cor_list_up_x[cnt_up_cor] = t
                cor_list_up_y[cnt_up_cor] = p
                cnt_up_cor = cnt_up_cor + 1
            else :
                if fx == False:
                    money = money - 1000
                wrong_list_up_x[cnt_up_wrong] = t
                wrong_list_up_y[cnt_up_wrong] = p
                cnt_up_wrong = cnt_up_wrong + 1
        elif max == 2:
            if fx:
                sell = p * fx_position
                buy =  price_data[ind5[loop_cnt] + pred_term] * fx_position
                profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * fx_spread * fx_position)))) * 100
                money = money + profit
            if max == y.argmax():
                if fx == False:
                    money = money + payout
                cor_list_down_x[cnt_down_cor] = t
                cor_list_down_y[cnt_down_cor] = p
                cnt_down_cor = cnt_down_cor + 1
            else:
                if fx == False:
                    money = money - 1000
                wrong_list_down_x[cnt_down_wrong] = t
                wrong_list_down_y[cnt_down_wrong] = p
                cnt_down_wrong = cnt_down_wrong + 1

        money_tmp[t] = money
        loop_cnt = loop_cnt + 1

    prev_money = 1000000
    #T = time[0]
    #print("T:" + T[11:])
    for i, ti in enumerate(time):
        if ti in money_tmp.keys():
            prev_money = money_tmp[ti]

        money_x[i] = ti[11:13]
        money_y[i] = prev_money

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
    fig = plt.figure()
    #価格の遷移
    ax1 = fig.add_subplot(111)
    #ax1.plot(time,close)
    ax1.plot(close, 'g')

    """
    ax1.plot(cor_list_up_x, cor_list_up_y, 'b^')
    ax1.plot(wrong_list_up_x, wrong_list_up_y, 'r^')
    ax1.plot(cor_list_down_x, cor_list_down_y, 'bv')
    ax1.plot(wrong_list_down_x, wrong_list_down_y, 'rv')
    """

    ax2 = ax1.twinx()
    ax2.plot(money_y)


    #index = np.arange(0,len(money_x),3600// int(s))
    #plt.xticks(index,money_x[index])

    plt.title('border:' + str(border) + " payout:" + str(payout) + " except index:" + str(except_index))
    plt.show()

