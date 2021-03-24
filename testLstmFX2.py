import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
import logging.config
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import time
from indices import index
from decimal import Decimal
from DataSequence2 import DataSequence2
from readConf2 import *
import pandas as pd
import sys

start = datetime(2018, 1, 1)
end = datetime(2018, 1, 31)
#end = datetime(2019, 12, 31)

#start = datetime(2020, 1, 1, 23)
#end = datetime(2020, 12, 31, 22)

# 2秒ごとの成績を計算する場合
per_sec_flg = True

def getAccFx(res, border, dataY, change):

    up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] > res[:, 2]) & (res[:, 0] > border))[0]
    down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > res[:, 1]) & (res[:, 2] > border))[0]

    x5_up = res[up_ind,:]
    y5_up = dataY[up_ind,:]
    c5_up = change[up_ind]

    #儲けを合算
    c5_up_sum = np.sum(c5_up)

    x5_down = res[down_ind,:]
    y5_down = dataY[down_ind,:]
    c5_down = change[down_ind]

    # 儲けを合算(売りなので-1を掛ける)
    c5_down_sum = np.sum(c5_down) * -1

    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float( Decimal("0.001") * Decimal(SPREAD) ))
    c5_sum = c5_sum * FX_POSITION

    up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
    down_cor_length = int(len(np.where(down_eq == True)[0]))

    total_num = len(up_ind) + len(down_ind)
    correct_num = up_cor_length + down_cor_length

    if total_num ==0:
        Acc =0
    else:
        Acc = correct_num / total_num

    return Acc, total_num, correct_num, c5_sum


def countDrawdoan(max_drawdowns, max_drawdown, drawdown, money):
    drawdown = drawdown + money
    if max_drawdown > drawdown:
        # 最大ドローダウンを更新してしまった場合
        max_drawdown = drawdown

    if drawdown > 0:
        if max_drawdown != 0:
            max_drawdowns.append(max_drawdown)
        drawdown = 0
        max_drawdown = 0

    return max_drawdown, drawdown


def do_predict():
    dataSequence2 = DataSequence2(0, start, end, True)

    # 正解ラベル(ndarray)
    correct_list = dataSequence2.get_correct_list()

    # 予想時のレート(FX用)
    pred_close_list = np.array(dataSequence2.get_pred_close_list())

    # 決済時のレート(FX用)
    real_close_list = np.array(dataSequence2.get_real_close_list())

    # レートの変化幅を保持
    change_list = real_close_list - pred_close_list

    # 全close値のリスト
    close_list = dataSequence2.get_close_list()

    # 全score値のリスト
    score_list = dataSequence2.get_score_list()

    # spread値のリスト
    target_spread_list = np.array(dataSequence2.get_target_spread_list())

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    train_list_index =  dataSequence2.get_train_list_index()

    model_suffix = [
        "90*50",
    ]


    border_list = [0.55, 0.56 ,0.57, 0.58, 0.59,  ]  # 正解率と獲得金額のみ表示
    # border_list = []  # 正解率と獲得金額のみ表示
    border_list_show = [0.55, 0.56 ,0.57, 0.58, 0.59, ]  # グラフも表示
    # print("model:", FILE_PREFIX)

    for suffix in model_suffix:
        print(suffix)

        load_dir = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix
        model = tf.keras.models.load_model(load_dir)
        # model.summary()

        # START tensorflow1で作成したモデル用
        """
        model_file = "/app/bin_op/model/GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5" + "." + suffix
        model = tf.keras.models.load_model(model_file)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        """
        # END tensorflow1で作成したモデル用

        # ndarrayで返って来る
        predict_list = model.predict_generator(dataSequence2,
                                               steps=None,
                                               max_queue_size=PROCESS_COUNT * 1,
                                               use_multiprocessing=False,
                                               verbose=0)

        """
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("train_list_index", len(train_list_index), train_list_index[:10])
        """

        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

        for border in border_list:

            # 予想結果表示用テキストを保持
            result_txt = []

            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            Acc, total_num, correct_num, profit = getAccFx(predict_list, border, correct_list, change_list)

            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
            result_txt.append("Earned Money:" + str(profit))

            if border not in border_list_show:
                for i in result_txt:
                    myLogger(i)
                continue

            money_y = []
            money_tmp = {}

            money = START_MONEY  # 始めの所持金

            bet_price = {} #注文した際のレートを保持
            bet_type = {} #BUY or SELL

            now_price = 0

            j = 0
            target_idx = -1
            bet_cnt = 0
            bet_len = {}
            bet_len_dict = {}

            for i in range(60):
                if i % BET_TERM == 0:
                    bet_price[i] = 0
                    bet_type[i] = ""
                    bet_len[i] = 0

            for cnt ,(sc, close, ind) in enumerate(zip(score_list, close_list, train_list_index)):

                shift = sc % 60

                j += 1

                #今のレート
                now_price = close

                #今のスコアで予想をした時のインデックスを取得
                #target_idx = np.where(target_score_list == sc)[0]
                if ind == -1:
                    if bet_price[shift] != 0 :
                        #ポジションがあり、予想がない場合はその日の最後の予想だったので決済する
                        print("predict not exist! score:", sc)
                        c = now_price - bet_price[shift]
                        if bet_type[shift] == "BUY":
                            profit = (c - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                        elif bet_type[shift] == "SELL":
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                        else:
                            #想定外エラー
                            print("ERROR1")
                            sys.exit()

                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                        money_tmp[sc] = money
                        bet_price[shift] = 0
                        bet_type[shift] = ""
                        bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                              0) != 0 else 1
                        bet_len[shift] = 0
                    continue

                #予想取得
                pred = predict_list[ind, :]
                max = pred.argmax()
                percent = pred[max]

                #ストップロス確認
                if bet_price[shift] != 0:
                    # 既にポジションがある場合
                    c = now_price - bet_price[shift]
                    if bet_type[shift] == "BUY":
                        if c <= FX_STOP_LOSS:
                            print("stop loss!", sc)
                            #ストップロス以上の損失を出している場合は決済する
                            profit = (c - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                  0) != 0 else 1
                            bet_len[shift] = 0

                    elif bet_type[shift] == "SELL":
                        if (c * -1) <= FX_STOP_LOSS:
                            print("stop loss!", sc)
                            #ストップロス以上の損失を出している場合は決済する
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                  0) != 0 else 1
                            bet_len[shift] = 0
                    else:
                        #想定外エラー
                        print("ERROR2")
                        sys.exit()


                if bet_price[shift] != 0:
                    #既にポジションがある場合
                    c = now_price - bet_price[shift]

                    if bet_type[shift] == "BUY":
                        #買いポジションがある場合
                        if max == 0 :
                            #更に上がると予想されている場合、決済しないままとする
                            bet_len[shift] += 1
                        else:
                            #更に上がらなければ決済する
                            profit = (c - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                  0) != 0 else 1
                            bet_len[shift] = 0

                    elif bet_type[shift] == "SELL":
                        #売りポジションがある場合
                        if max == 2 :
                            #更に下がると予想されている場合、決済しないままとする
                            bet_len[shift] += 1
                        else:
                            #更に下がらなければ決済する
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                  0) != 0 else 1
                            bet_len[shift] = 0
                    else:
                        #想定外エラー
                        print("ERROR3")
                        sys.exit()

                if bet_price[shift] == 0:
                    #ポジションがない場合
                    if max == 0 and percent >= border:
                        bet_price[shift] = now_price
                        bet_type[shift] = "BUY"
                        bet_cnt += 1
                        bet_len[shift] = 1
                    elif max == 2 and percent >= border:
                        bet_price[shift] = now_price
                        bet_type[shift] = "SELL"
                        bet_cnt += 1
                        bet_len[shift] = 1


            print("loop cnt:", j)

            prev_money = START_MONEY


            for i, score in enumerate(score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)


            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")

            fig = plt.figure()
            # 価格の遷移
            ax1 = fig.add_subplot(111)

            ax1.plot(close_list, 'g')

            ax2 = ax1.twinx()
            ax2.plot(money_y)

            plt.title(
                'border:' + str(border) + " spread:" + str(SPREAD) + " money:" + str(
                    money))

            print("Earned Money: ",prev_money - START_MONEY)
            for i in result_txt:
                myLogger(i)

            print("bet cnt:", bet_cnt)
            print("bet期間 件数")
            for k, v in sorted(bet_len_dict.items()):
                print(k, bet_len_dict.get(k, 0))

            print("MAX DrawDowns(理論上のドローダウン)")
            max_drawdowns.sort()
            myLogger(max_drawdowns[0:10])

            drawdown_cnt = {}
            for i in max_drawdowns:
                for k, v in DRAWDOWN_LIST.items():
                    if i < v[0] and i >= v[1]:
                        drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                        break
            for k, v in sorted(DRAWDOWN_LIST.items()):
                print(k, drawdown_cnt.get(k, 0))


            plt.show()


if __name__ == "__main__":
    do_predict()

    print("END!!!")