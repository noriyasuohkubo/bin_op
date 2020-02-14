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
import pandas as pd

start = datetime(2018, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2019, 12, 31)

end_stp = int(time.mktime(end.timetuple()))

np.random.seed(0)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

def get_redis_data(sym):
    myLogger("Model is ", model_file)
    model = get_model()
    if model is None:
        return None
    r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(sym, start_stp, end_stp, withscores=False)

    close_tmp, time_tmp = [], []
    not_except_index_cnts = []

    myLogger(result[0:5])

    indicies = np.ones(len(index), dtype=np.int32)
    #経済指標発表前後2時間は予想対象からはずす
    for i,ind in enumerate(index):
        tmp_datetime = datetime.strptime(ind, '%Y-%m-%d %H:%M:%S')
        indicies[i] = int(time.mktime(tmp_datetime.timetuple()))

    for line in result:
        tmps = json.loads(line)
        close_tmp.append(tmps.get("close"))
        time_tmp.append(tmps.get("time"))

    close_data, high_data, low_data, label_data, time_data, price_data, end_price_data, close_abs_data = [], [], [], [], [], [], [], []
    not_except_data = []
    spread_data = []
    up =0
    same =0
    data_length = len(time_tmp) - (maxlen * close_shift) - (pred_term * close_shift) -1
    index_cnt = -1

    for i in range(data_length):
        continue_flg = False
        except_flg = False

        if except_index:
            tmp_datetime = datetime.strptime(time_tmp[i + (maxlen * close_shift) -1], '%Y-%m-%d %H:%M:%S')
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
        tmp_time_bef = datetime.strptime(time_tmp[i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[i + (maxlen * close_shift) -1], '%Y-%m-%d %H:%M:%S')
        delta =tmp_time_aft - tmp_time_bef

        if delta.total_seconds() >= (maxlen * int(s)):
            continue;

        # sよりmergの方が大きい数字の場合、
        # 検証時(testLstm.py)は秒をmergで割った余りが0のデータだけを使って結果をみる、なぜならDB内データの間隔の方がトレードタイミングより短いため
        if int(s) < int(merg):
            sec = time_tmp[i][-2:]
            if Decimal(str(sec)) % Decimal(merg) != 0:
                continue

        index_cnt = index_cnt +1
        #ハイローオーストラリアの取引時間外を学習対象からはずす、予想させる方は対象から外していない
        #→取引時間を通常営業時間外より更に絞ることによりトレード結果がどう変わるか見るため(例えば、午前中は取引が少ないから勝率低そうなど)
        if except_highlow:
            if datetime.strptime(time_tmp[i + (maxlen * close_shift) -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                except_flg = True
                not_except_data.append(False)
            else:
                not_except_index_cnts.append(index_cnt)
                not_except_data.append(True)

        # 取引する時間
        time_data.append(time_tmp[i + (maxlen * close_shift) -1])
        # 取引した時のレート
        price_data.append(close_tmp[i + (maxlen * close_shift) -1])
        # 取引した時の判定レート
        end_price_data.append(close_tmp[i + (maxlen * close_shift) + (pred_term * close_shift) -1])

        bef = close_tmp[i + (maxlen * close_shift) -1]
        aft = close_tmp[i + (maxlen * close_shift) + (pred_term * close_shift) -1]

        #正解をいれる
        if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
            #上がった場合
            label_data.append([1,0,0])
            if except_flg != True:
                up = up + 1
        elif  float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
            label_data.append([0,0,1])
        else:
            label_data.append([0,1,0])
            if except_flg != True:
                same = same + 1

    ##close_np = np.array(close_data)
    time_np = np.array(time_data)
    price_np = np.array(price_data)
    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)
    end_price_np = np.array(end_price_data)
    #close_abs_np = np.array(close_abs_data)
    #spread_np = np.array(spread_data)
    ##retX = np.zeros((len(close_np), maxlen, in_num))
    ##retX[:, :, 0] = close_np[:]

    retY = np.array(label_data)
    retZ = np.array(not_except_data)

    #myLogger("spread_tmp length:",len(spread_tmp))
    ##myLogger("X SHAPE:", retX.shape)
    myLogger("Y SHAPE:", retY.shape)
    myLogger("Z SHAPE:", retZ.shape)
    myLogger("UP: ",up/len(not_except_index_cnts))
    myLogger("SAME: ", same / len(not_except_index_cnts))
    myLogger("DOWN: ", (len(not_except_index_cnts) - up - same) / len(not_except_index_cnts))
    """
    for k, v in sorted(spread_cnt.items()):
        myLogger(k,v)
    """
    not_except_index_cnts_np = np.array(not_except_index_cnts)

    ##return retX, retY, price_np, time_np, close_tmp_np, time_tmp_np,end_price_np,close_abs_np
    return retY, retZ, price_np, time_np, close_tmp_np, time_tmp_np,end_price_np,not_except_index_cnts_np

def get_model():

    if os.path.isfile(model_file):
        model = load_model(model_file)
        model_gpu = multi_gpu_model(model, gpus=gpu_count)
        model_gpu.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        myLogger("Load Model")
        return model_gpu
    else:
        myLogger("the Model not exists!")
        return None

#def do_predict(test_X, test_Y):
def do_predict(symbol):
    model = get_model()
    if model is None:
        return None

    dataSequence = DataSequence(0, start, end, True)

    res = model.predict_generator(dataSequence, steps=None, max_queue_size=process_count * 1, use_multiprocessing=False, verbose=0)
    myLogger("res", res[0:10])
    #res = model.predict(test_X, verbose=0, batch_size=batch_size)
    myLogger("Predict finished")

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

    if (len(up_ind5) + len(down_ind5)) ==0:
        Acc =0
    else:
        Acc = (up_cor_length + down_cor_length) / (len(up_ind5) + len(down_ind5))
    total = len(up_ind5) + len(down_ind5)
    correct = int(total * Acc)

    return Acc, total, correct

def countDrawdoan(max_drawdowns, max_drawdown, drawdown, money):
    drawdown = drawdown + money
    if max_drawdown > drawdown:
        #最大ドローダウンを更新してしまった場合
        max_drawdown = drawdown

    if drawdown > 0:
        if max_drawdown != 0:
            max_drawdowns.append(max_drawdown)
        drawdown = 0
        max_drawdown = 0

    return max_drawdown, drawdown

if __name__ == "__main__":
    ##dataX, dataY, price_data, time_data, close, time, end_price_data, close_abs_data = get_redis_data()
    result = {}
    time_list = {}
    not_except_result = {}

    for sym in symbols:

        dataY_tmp, dataZ_tmp, price_data_tmp, time_data_tmp, close_tmp, time_tmp, end_price_data_tmp,not_except_index_cnts = get_redis_data(sym)
        res_tmp = do_predict(sym)

        """
        myLogger("Length")
        myLogger(len(res_tmp))
        myLogger(len(dataY_tmp))
        myLogger(len(price_data_tmp))
        myLogger(len(time_data_tmp))
        myLogger(len(close_tmp))
        myLogger(len(time_tmp))
        myLogger(len(end_price_data_tmp))
        myLogger(len(spread_data_tmp))
        """

        for j in range(len(dataY_tmp)):
            tmp_result = {}
            tmp_result["res_tmp"] = res_tmp[j]
            tmp_result["dataY_tmp"] = dataY_tmp[j]
            tmp_result["dataZ_tmp"] = dataZ_tmp[j]
            tmp_result["price_data_tmp"] = price_data_tmp[j]
            tmp_result["time_data_tmp"] = time_data_tmp[j]
            tmp_result["end_price_data_tmp"] = end_price_data_tmp[j]
            #tmp_result["spread_data_tmp"] = spread_data_tmp[j]
            result[time_data_tmp[j]] = tmp_result

            if dataZ_tmp[j] == True:
                tmp_not_except_result = {}
                tmp_not_except_result["res_tmp"] = res_tmp[j]
                tmp_not_except_result["dataY_tmp"] = dataY_tmp[j]
                tmp_not_except_result["dataZ_tmp"] = dataZ_tmp[j]
                not_except_result[time_data_tmp[j]] = tmp_not_except_result

        for k in range(len(close_tmp)):
            tmp_time = {}
            tmp_time["close_tmp"] = close_tmp[k]
            tmp_time["time_tmp"] = time_tmp[k]
            time_list[time_tmp[k]] = tmp_time

    res, dataY, dataZ, price_data, time_data, close, time, end_price_data, spread_data = [],[],[],[],[],[],[],[],[]
    tmp_x, tmp_y, tmp_z = [],[],[]

    for k, v in sorted(result.items()):
        res.append(v["res_tmp"])
        dataY.append(v["dataY_tmp"])
        dataZ.append(v["dataZ_tmp"])
        price_data.append(v["price_data_tmp"])
        time_data.append(v["time_data_tmp"])
        end_price_data.append(v["end_price_data_tmp"])
        #spread_data.append(v["spread_data_tmp"])

    for k, v in sorted(not_except_result.items()):
        tmp_x.append(v["res_tmp"])
        tmp_y.append(v["dataY_tmp"])
        tmp_z.append(v["dataZ_tmp"])


    for k, v in sorted(time_list.items()):
        close.append(v["close_tmp"])
        time.append(v["time_tmp"])

    res = np.array(res)
    dataY = np.array(dataY)
    dataZ = np.array(dataZ)
    price_data = np.array(price_data)
    time_data = np.array(time_data)
    close = np.array(close)
    time = np.array(time)
    end_price_data = np.array(end_price_data)
    #spread_data = np.array(spread_data)
    #border_list=[0.55,0.56,0.57,0.58]
    #border_list_show=[0.55,0.56,0.57]
    border_list=[0.558,0.56,0.562]
    border_list_show=[0.558,0.56,0.562]
    result_txt = []

    tmp_x = np.array(tmp_x)
    tmp_y = np.array(tmp_y)
    tmp_z = np.array(tmp_z)
    #tmp_x = res[not_except_index_cnts, :]
    #tmp_y =  dataY[not_except_index_cnts, :]
    #tmp_z = dataZ[not_except_index_cnts]

    for b in border_list:
        spread_trade = {}
        spread_win = {}
        max_drawdown = 0
        drawdown = 0
        max_drawdowns = []

        border = b
        Acc5 = getAcc(tmp_x,border,tmp_y)

        result_txt.append("Accuracy over " + str(border) + ":" + str(Acc5[0]))
        result_txt.append("Total:" + str(Acc5[1]) + " Correct:" + str(Acc5[2]))

        if fx == False:
            win_money = (payout * Acc5[2]) - ((Acc5[1] - Acc5[2]) * payoff)
            result_txt.append("Money:" + str(win_money))
        if border not in border_list_show:
            continue

        perTimeRes = {}
        for j in range(24):
            perTimeRes[str(j)] = {"count": 0, "win_count": 0}

        ind5 = np.where(res >=border)[0]
        x5 = res[ind5,:]
        y5= dataY[ind5,:]
        z5= dataZ[ind5]
        p5 = price_data[ind5]
        t5 = time_data[ind5]
        ep5 = end_price_data[ind5]
        #sp5 = spread_data[ind5]
        #ca5 = close_abs_data[ind5]

        #myLogger(t5[0:10])
        up = tmp_x[:, 0]
        down = tmp_x[:, 2]

        up_ind5 = np.where(up >= border)[0]
        down_ind5 = np.where(down >= border)[0]

        x5_up = tmp_x[up_ind5,:]
        y5_up= tmp_y[up_ind5,:]

        up_total_length = len(x5_up)
        up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
        #Acc5_up = np.mean(np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1)))
        up_cor_length = int(len(np.where(up_eq == True)[0]))
        up_wrong_length = int(up_total_length - up_cor_length)
        #myLogger("up_cor_length:"+ str(up_cor_length))
        #myLogger("up_wrong_length:" + str(up_wrong_length))

        x5_down = tmp_x[down_ind5,:]
        y5_down= tmp_y[down_ind5,:]
        down_total_length = len(x5_down)
        #Acc5_down = np.mean(np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1)))
        down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
        down_cor_length = int(len(np.where(down_eq == True)[0]))
        down_wrong_length = int(down_total_length - down_cor_length)
        #myLogger("down_cor_length:"+ str(down_cor_length))
        #myLogger("down_wrong_length:" + str(down_wrong_length))

        cor_list_up_x, cor_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_cor_length) ]), np.ones(up_cor_length, dtype=np.float64)
        wrong_list_up_x, wrong_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_wrong_length) ]), np.ones(up_wrong_length, dtype=np.float64)
        cor_list_down_x, cor_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_cor_length) ]), np.ones(down_cor_length, dtype=np.float64)
        wrong_list_down_x, wrong_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_wrong_length) ]), np.ones(down_wrong_length, dtype=np.float64)

        #cor_list_abs, wrong_list_abs = np.ones(up_cor_length + down_cor_length, dtype=np.float64), np.ones(up_wrong_length + down_wrong_length, dtype=np.float64)

        money_x, money_y = np.array([ "00:00:00" for i in range(len(time)) ]), np.ones(len(time), dtype=np.float64)
        money_tmp = {}

        money = 1000000
        myLogger(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

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

        ind_tmp = -1
        for x,y,z,p,t,ep in zip(x5,y5,z5,p5,t5,ep5):
            hourT = str(datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour)
            ind_tmp = ind_tmp +1
            not_except_flg = z

            max = x.argmax()
            """
            time_rate_list[int(t[11:13])].all_cnt += 1
            if max == y.argmax():
                time_rate_list[int(t[11:13])].correct_cnt += 1
            """
            trade_flg = False
            win_flg = False
            if max == 0:
                if not_except_flg:
                    trade_flg = True
                perTimeRes[hourT]["count"] += 1

                # Up predict
                if fx:
                    buy = p * fx_position
                    sell = ep * fx_position
                    # DB上の値は実際の1／100なので100倍している
                    profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * spread * fx_position)))) * 100
                    if not_except_flg:
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                if max == y.argmax():
                    if fx == False:
                        if not_except_flg:
                            money = money + payout
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payout)
                    if not_except_flg:
                        cor_list_up_x[cnt_up_cor] = t
                        cor_list_up_y[cnt_up_cor] = p
                        cnt_up_cor = cnt_up_cor + 1
                        win_flg = True
                    perTimeRes[hourT]["win_count"] += 1
                    #cor_list_abs[cnt_cor_abs] = ca
                    #cnt_cor_abs += 1
                else :
                    if fx == False:
                        if not_except_flg:
                            money = money - payoff
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payoff * -1)
                    if not_except_flg:
                        wrong_list_up_x[cnt_up_wrong] = t
                        wrong_list_up_y[cnt_up_wrong] = p
                        cnt_up_wrong = cnt_up_wrong + 1

                    #wrong_list_abs[cnt_wrong_abs] = ca
                    #cnt_wrong_abs += 1
            elif max == 2:
                if not_except_flg:
                    trade_flg = True
                perTimeRes[hourT]["count"] += 1
                if fx:
                    sell = p * fx_position
                    buy =  ep * fx_position
                    profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * spread * fx_position)))) * 100
                    if not_except_flg:
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                if max == y.argmax():
                    if fx == False:
                        if not_except_flg:
                            money = money + payout
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payout)
                    if not_except_flg:
                        cor_list_down_x[cnt_down_cor] = t
                        cor_list_down_y[cnt_down_cor] = p
                        cnt_down_cor = cnt_down_cor + 1
                        win_flg = True
                    perTimeRes[hourT]["win_count"] += 1

                    #cor_list_abs[cnt_cor_abs] = ca
                    #cnt_cor_abs += 1
                else:
                    if fx == False:
                        if not_except_flg:
                            money = money - payoff
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payoff * -1)
                    if not_except_flg:
                        wrong_list_down_x[cnt_down_wrong] = t
                        wrong_list_down_y[cnt_down_wrong] = p
                        cnt_down_wrong = cnt_down_wrong + 1

                    #wrong_list_abs[cnt_wrong_abs] = ca
                    #cnt_wrong_abs += 1
            flg = False
            """
            if trade_flg:
                for k, v in spread_list.items():
                    if sp > v[0] and sp <= v[1]:
                        spread_trade[k] = spread_trade.get(k, 0) + 1
                        if win_flg:
                            spread_win[k] = spread_win.get(k, 0) + 1
                        flg = True
                        break
                if flg == False:
                    if sp < 0:
                        spread_trade["spread0"] = spread_trade.get("spread0", 0) + 1
                        if win_flg:
                            spread_win["spread0"] = spread_win.get("spread0", 0) + 1
                    else:
                        spread_trade["spread16Over"] = spread_trade.get("spread16Over", 0) + 1
                        if win_flg:
                            spread_win["spread16Over"] = spread_win.get("spread16Over", 0) + 1
            """
            money_tmp[t] = money
            loop_cnt = loop_cnt + 1

        prev_money = 1000000
        #T = time[0]
        #myLogger("T:" + T[11:])
        for i, ti in enumerate(time):
            if ti in money_tmp.keys():
                prev_money = money_tmp[ti]

            money_x[i] = ti[11:13]
            money_y[i] = prev_money

        #myLogger("close_abs_cor mean:" + str(np.mean(cor_list_abs)) + " std:" + str(np.std(cor_list_abs)))
        #myLogger("close_abs_wrong_mean:" + str(np.mean(wrong_list_abs)) + " std:" + str(np.std(wrong_list_abs)))

        myLogger(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
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
                myLogger("time:",k," all_cnt:",v.all_cnt," correct_cnt:",v.correct_cnt," correct_rate:",v.correct_cnt/v.all_cnt)
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
        if fx == True:
            plt.title('border:' + str(border) + " position:" + str(fx_position) + " spread:" + str(spread) + " money:" + str(money))
        else:
            plt.title('border:' + str(border) + " payout:" + str(payout) + " spread:" + str(spread) + " money:" + str(money))
        plt.show()

        for k in range(24):
            cnt = perTimeRes[str(k)]["count"]
            winCnt = perTimeRes[str(k)]["win_count"]
            if cnt != 0:
                myLogger(str(k) + ": win rate " + str(winCnt / cnt) + " ,count " + str(cnt) + " ,win count " + str(winCnt))
        """
        for k, v in sorted(spread_list.items()):
            if spread_trade.get(k, 0) != 0:
                myLogger(k, " cnt:", spread_trade.get(k, 0), " win rate:", spread_win.get(k,0)/spread_trade.get(k))
            else:
                myLogger(k, " cnt:", spread_trade.get(k, 0))
        """
        max_drawdowns.sort()
        myLogger(max_drawdowns[0:10])

        drawdown_cnt = {}
        for i in max_drawdowns:
            for k, v in drawdown_list.items():
                if i < v[0] and i >= v[1]:
                    drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                    break
        for k, v in sorted(drawdown_list.items()):
            myLogger(k, drawdown_cnt.get(k,0))

        max_drawdowns_np = np.array(max_drawdowns)
        df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
        myLogger(df)

    for i in result_txt:
        myLogger(i)
