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
from DataSequence import DataSequence
from readConf import *

start = datetime(2018, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 11, 1)
end_stp = int(time.mktime(end.timetuple()))

payout = 950
if bin_type == "turbo_spread":
    payout = 1000

except_index = False
except_highlow = True

np.random.seed(0)

fx = False
fx_position = 10000
#fx_spread = 10

in_num=1

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

def get_redis_data():
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(symbol + db_suffix, start_stp, end_stp, withscores=False)
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
        tmps = json.loads(line)
        if askbid == "_ask":
            close_tmp.append(tmps.get("ask"))
        else:
            close_tmp.append(float((Decimal(str(tmps.get("close"))) + Decimal(str(tmps.get("ask")))) / Decimal("2")))
            #close_tmp.append(tmps.get("close"))

        time_tmp.append(tmps.get("time"))

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]

    close_data, high_data, low_data, label_data, time_data, price_data, end_price_data, close_abs_data = [], [], [], [], [], [], [], []

    up =0
    same =0
    data_length = len(close) - maxlen - pred_term -1

    for i in range(data_length):
        continue_flg = False
        #ハイローオーストラリアの取引時間外を学習対象からはずす

        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue;
        if except_index:
            tmp_datetime = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
            score = int(time.mktime(tmp_datetime.timetuple()))
            for ind in indicies:
                ind_datetime = datetime.fromtimestamp(ind)

                bef_datetime = ind_datetime - timedelta(minutes=10)
                aft_datetime = ind_datetime + timedelta(minutes=30)
                bef_time = int(time.mktime(bef_datetime.timetuple()))
                aft_time = int(time.mktime(aft_datetime.timetuple()))

                if bef_time <= score and score <= aft_time:
                    continue_flg = True
                    break;

            if continue_flg:
                continue;
        #maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
        tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
        delta =tmp_time_aft - tmp_time_bef

        if delta.total_seconds() > ((maxlen-1) * int(s)):
            #print(tmp_time_aft)
            continue;

        ##close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])
        end_price_data.append(close_tmp[1 + i + maxlen + pred_term -1])
        #close_abs_data.append(abs(close[i + maxlen -1]))
        #tmp_abs= abs(close[i + maxlen -11:i + maxlen -1])
        #close_abs_data.append(sum(tmp_abs)/len(tmp_abs))

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
    ##close_np = np.array(close_data)
    time_np = np.array(time_data)
    price_np = np.array(price_data)
    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)
    end_price_np = np.array(end_price_data)
    close_abs_np = np.array(close_abs_data)

    ##retX = np.zeros((len(close_np), maxlen, in_num))
    ##retX[:, :, 0] = close_np[:]

    retY = np.array(label_data)

    ##print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ",up/len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))

    ##return retX, retY, price_np, time_np, close_tmp_np, time_tmp_np,end_price_np,close_abs_np
    return retY, price_np, time_np, close_tmp_np, time_tmp_np,end_price_np,close_abs_np

def get_model():

    if os.path.isfile(model_file):
        model = load_model(model_file)
        model_gpu = multi_gpu_model(model, gpus=gpu_count)
        model_gpu.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        print("Load Model")
        return model_gpu
    else:
        print("the Model not exists!")
        return None

#def do_predict(test_X, test_Y):
def do_predict():
    model = get_model()
    if model is None:
        return None

    dataSequence = DataSequence(maxlen, pred_term, s, in_num, batch_size, symbol, spread, 0, symbols, start, end, True)
    res = model.predict_generator(dataSequence, steps=None, max_queue_size=process_count * 1, use_multiprocessing=False, verbose=0)

    #res = model.predict(test_X, verbose=0, batch_size=batch_size)
    print("Predict finished")

    K.clear_session()
    return res

class TimeRate():
    def __init__(self):
        self.all_cnt = 0
        self.correct_cnt = 0

def getAcc(res, border, dataY):
    up = res[:, 0]
    down = res[:, 2]
    up_ind5 = np.where(up >= border)[0]
    down_ind5 = np.where(down >= border)[0]
    x5_up = res[up_ind5,:]
    y5_up= dataY[up_ind5,:]
    x5_down = res[down_ind5,:]
    y5_down= dataY[down_ind5,:]

    up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
    down_cor_length = int(len(np.where(down_eq == True)[0]))

    Acc = (up_cor_length + down_cor_length) / (len(up_ind5) + len(down_ind5))
    total = len(up_ind5) + len(down_ind5)
    correct = int(total * Acc)

    return Acc, total, correct

if __name__ == "__main__":
    ##dataX, dataY, price_data, time_data, close, time, end_price_data, close_abs_data = get_redis_data()
    dataY, price_data, time_data, close, time, end_price_data, close_abs_data = get_redis_data()
    res = do_predict()

    border_list=[0.55,0.56,0.57,0.58,0.59,0.60,0.61,]
    border_list_show=[0.56,0.57,0.58]
    result_txt = []
    for b in border_list:
        border = b
        Acc5 = getAcc(res,border,dataY)

        result_txt.append("Accuracy over " + str(border) + ":" + str(Acc5[0]))
        result_txt.append("Total:" + str(Acc5[1]) + " Correct:" + str(Acc5[2]))

        if border not in border_list_show:
            continue

        ind5 = np.where(res >=border)[0]
        x5 = res[ind5,:]
        y5= dataY[ind5,:]
        p5 = price_data[ind5]
        t5 = time_data[ind5]
        ep5 = end_price_data[ind5]
        #ca5 = close_abs_data[ind5]

        #print(t5[0:10])
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
        #print("up_cor_length:"+ str(up_cor_length))
        #print("up_wrong_length:" + str(up_wrong_length))

        x5_down = res[down_ind5,:]
        y5_down= dataY[down_ind5,:]
        p5_down = price_data[down_ind5]
        t5_down = time_data[down_ind5]
        down_total_length = len(x5_down)
        #Acc5_down = np.mean(np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1)))
        down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
        down_cor_length = int(len(np.where(down_eq == True)[0]))
        down_wrong_length = int(down_total_length - down_cor_length)
        #print("down_cor_length:"+ str(down_cor_length))
        #print("down_wrong_length:" + str(down_wrong_length))

        cor_list_up_x, cor_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_cor_length) ]), np.ones(up_cor_length, dtype=np.float64)
        wrong_list_up_x, wrong_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_wrong_length) ]), np.ones(up_wrong_length, dtype=np.float64)
        cor_list_down_x, cor_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_cor_length) ]), np.ones(down_cor_length, dtype=np.float64)
        wrong_list_down_x, wrong_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_wrong_length) ]), np.ones(down_wrong_length, dtype=np.float64)

        #cor_list_abs, wrong_list_abs = np.ones(up_cor_length + down_cor_length, dtype=np.float64), np.ones(up_wrong_length + down_wrong_length, dtype=np.float64)

        money_x, money_y = np.array([ "00:00:00" for i in range(len(time)) ]), np.ones(len(time), dtype=np.float64)
        money_tmp = {}

        money = 1000000
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

        cnt_up_cor = 0
        cnt_up_wrong = 0
        cnt_down_cor = 0
        cnt_down_wrong = 0
        loop_cnt = 0
        cnt_cor_abs = 0
        cnt_wrong_abs = 0

        time_rate_list = {}
        for i in range(0, 24):
            time_rate_list[i] = (TimeRate())

        for x,y,p,t,ep in zip(x5,y5,p5,t5,ep5):

            max = x.argmax()
            """
            time_rate_list[int(t[11:13])].all_cnt += 1
            if max == y.argmax():
                time_rate_list[int(t[11:13])].correct_cnt += 1
            """
            if max == 0:
                # Up predict
                if fx:
                    buy = p * fx_position
                    sell = ep * fx_position
                    # DB上の値は実際の1／100なので100倍している
                    profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * spread * fx_position)))) * 100
                    money = money + profit
                if max == y.argmax():
                    if fx == False:
                        money = money + payout
                    cor_list_up_x[cnt_up_cor] = t
                    cor_list_up_y[cnt_up_cor] = p
                    cnt_up_cor = cnt_up_cor + 1

                    #cor_list_abs[cnt_cor_abs] = ca
                    #cnt_cor_abs += 1
                else :
                    if fx == False:
                        money = money - 1000
                    wrong_list_up_x[cnt_up_wrong] = t
                    wrong_list_up_y[cnt_up_wrong] = p
                    cnt_up_wrong = cnt_up_wrong + 1

                    #wrong_list_abs[cnt_wrong_abs] = ca
                    #cnt_wrong_abs += 1
            elif max == 2:
                if fx:
                    sell = p * fx_position
                    buy =  ep * fx_position
                    profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * spread * fx_position)))) * 100
                    money = money + profit
                if max == y.argmax():
                    if fx == False:
                        money = money + payout
                    cor_list_down_x[cnt_down_cor] = t
                    cor_list_down_y[cnt_down_cor] = p
                    cnt_down_cor = cnt_down_cor + 1

                    #cor_list_abs[cnt_cor_abs] = ca
                    #cnt_cor_abs += 1
                else:
                    if fx == False:
                        money = money - 1000
                    wrong_list_down_x[cnt_down_wrong] = t
                    wrong_list_down_y[cnt_down_wrong] = p
                    cnt_down_wrong = cnt_down_wrong + 1

                    #wrong_list_abs[cnt_wrong_abs] = ca
                    #cnt_wrong_abs += 1

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

        #print("close_abs_cor mean:" + str(np.mean(cor_list_abs)) + " std:" + str(np.std(cor_list_abs)))
        #print("close_abs_wrong_mean:" + str(np.mean(wrong_list_abs)) + " std:" + str(np.std(wrong_list_abs)))

        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")

        """
        #plt.subplot(1, 2, 1)
        plt.plot(cor_list_abs, "ro")
        #plt.subplot(1, 2, 2)
        plt.plot(wrong_list_abs, "bo")
        plt.show()
        """
        """
        for k, v in time_rate_list.items():
            if v.all_cnt != 0:
                print("time:",k," all_cnt:",v.all_cnt," correct_cnt:",v.correct_cnt," correct_rate:",v.correct_cnt/v.all_cnt)
        """
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

    for i in result_txt:
        print(i)