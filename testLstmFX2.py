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

start = datetime(2021, 1, 1)
end = datetime(2021, 9, 30,22)
#end = datetime(2021, 1, 11)

#start = datetime(2020, 1, 1, 23)
#end = datetime(2020, 12, 31, 22)

# 2秒ごとの成績を計算する場合
per_sec_flg = True

def getAccFxCat(res, border, dataY, change):

    up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] >= res[:, 2]) & (res[:, 0] >= border))[0]
    down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > res[:, 1]) & (res[:, 2] >= border))[0]

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

    # 儲けからスプレッド分をひく
    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float( Decimal("0.001") * Decimal(FX_SPREAD) ))
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

def getAccFxRgr(res, pred_close_list, real_close_list, change, border):
    mid_tmp = res[:, 0]

    # print(mid_tmp)
    # 現実のレートに換算する
    mid = pred_close_list * ((mid_tmp / 10000) + 1)

    # 上昇予想 レートより予想が上の場合 ベット対象
    if EXCEPT_DIVIDE_MAX != 0:
        mid_max = pred_close_list * ((EXCEPT_DIVIDE_MAX / 10000) + 1)
        up_ind = np.where(
            (mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(FX_SPREAD))) + border) & (mid <= mid_max))
    else:
        up_ind = np.where(mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(FX_SPREAD))) + border)

    up_win = np.where(
        real_close_list[up_ind] >= pred_close_list[up_ind] + float(Decimal("0.001") * Decimal(str(FX_SPREAD))))

    # 下降予想 レートより予想が下の場合 ベット対象
    if EXCEPT_DIVIDE_MAX != 0:
        mid_min = pred_close_list * ((EXCEPT_DIVIDE_MAX * -1 / 10000) + 1)
        down_ind = np.where(
            (mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(FX_SPREAD))) - border) & (mid >= mid_min))
    else:
        down_ind = np.where(mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(FX_SPREAD))) - border)
    down_win = np.where(
        real_close_list[down_ind] <= pred_close_list[down_ind] - float(Decimal("0.001") * Decimal(str(FX_SPREAD))))

    c5_up = change[up_ind]
    # 儲けを合算
    c5_up_sum = np.sum(c5_up)

    c5_down = change[down_ind]
    # 儲けを合算(売りなので-1を掛ける)
    c5_down_sum = np.sum(c5_down) * -1

    # 儲けからスプレッドをひく
    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float(Decimal("0.001") * Decimal(FX_SPREAD)))
    c5_sum = c5_sum * FX_POSITION

    total_num = len(down_ind[0]) + len(up_ind[0])
    correct_num = len(down_win[0]) + len(up_win[0])

    Acc = 0
    if total_num > 0:
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
    dataSequence2 = DataSequence2(0, start, end, True, False)

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
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
                          ]

    border_list = [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64,  ]  # spread1
    #border_list = [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51 ,0.52,  ]  # spread10
    #border_list = []

    #border_list = [-0.006, -0.004, -0.002, 0, 0.002, 0.004,   ]  # RGR用

    border_list_show = [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64,]  # グラフも表示

    show_detail = False

    #ベット延長するモデルを別途設定する場合
    ext_flg = False

    #ベット延長するか判断するTERM
    ext_term = 30
    if ext_flg == False:
        ext_term = TERM

    #ベット延長しはじめる期間 例えば本来のモデルが120秒予想で、延長決定するモデルが30秒予想の場は4
    #120秒経過してはじめて延長するか判断する　120秒経過するまではロスカットされない限りベットしつづける
    ext_begin_len = int(Decimal(str(TERM)) / Decimal(str(ext_term)))
    print("ext_begin_len", ext_begin_len)

    ext_predict_list = []

    FILE_PREFIX = "GBPJPY_CATEGORY_LSTM_BET2_TERM120_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU_FX"

    if ext_flg:
    #ベット延長するかを決定するモデル
        EXT_FILE_PREFIX = "GBPJPY_CATEGORY_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU_FX-11"

        if len(border_list_show) != 0 and (LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION"):
            #詳細を表示する場合(ベット延長するかを決定するモデルを使用する場合)
            ext_load_dir = "/app/model/bin_op/" + EXT_FILE_PREFIX
            ext_model = tf.keras.models.load_model(ext_load_dir)
            ext_predict_list = ext_model.predict_generator(dataSequence2,
                                                       steps=None,
                                                       max_queue_size=PROCESS_COUNT * 1,
                                                       use_multiprocessing=False,
                                                       verbose=0)

    #FILE_PREFIX = "GBPJPY_CATEGORY_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU_FX"
    #FILE_PREFIX = "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-4-2-1_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_LOSS-HUBER"
    # print("FILE_PREFIX:", FILE_PREFIX)

    # 一番良い結果を保持
    max_result = {"suffix": None, "profit": None, "border": None}
    max_result_detail = {"suffix": None, "profit": None, "border": None}

    for suffix in model_suffix:

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
        if ext_flg == False:
            ext_predict_list = predict_list

        print("train_list_index length:", len(train_list_index))
        print("predict_list length:", len(predict_list))
        print("ext_predict_list length:", len(ext_predict_list))

        """
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("train_list_index", len(train_list_index), train_list_index[:10])
        """

        #print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")
        print("")
        print("suffix:", suffix)

        for border in border_list:

            # 予想結果表示用テキストを保持
            result_txt = []

            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            if LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION":
                Acc, total_num, correct_num, profit = getAccFxRgr(predict_list, pred_close_list, real_close_list, change_list, border)
            else:
                Acc, total_num, correct_num, profit = getAccFxCat(predict_list, border, correct_list, change_list)


            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
            result_txt.append("Earned Money:" + str(profit))

            if max_result["suffix"] == None or max_result["profit"] < profit:
                max_result["suffix"] = suffix
                max_result["profit"] = int(profit)
                max_result["border"] = str(border)

            if border not in border_list_show:
                for i in result_txt:
                    myLogger(i)
                continue

            money_y = []
            money_tmp = {}

            money = START_MONEY  # 始めの所持金

            bet_price = {} #注文した際のレートを保持
            bet_type = {} #BUY or SELL
            profit_per_shift = {} #shiftごとの利益を保持

            now_price = 0

            j = 0
            target_idx = -1
            bet_cnt = 0
            bet_len = {} #betしている期間
            bet_len_dict = {}

            for i in range(ext_term):
                if i % BET_TERM == 0:
                    bet_price[i] = 0
                    bet_type[i] = ""
                    bet_len[i] = 0
                    profit_per_shift[i] = 0

            stop_loss_cnt = 0

            for cnt ,(sc, close, ind) in enumerate(zip(score_list, close_list, train_list_index)):

                shift = sc % ext_term

                j += 1

                #今のレート
                now_price = close

                #1周り前のShiftで決済したタイプを保持
                #stoplossや延長をしないなどの理由で決済したら以下にBUY,SELLをいれる
                bet_type_prev_shift = ""

                #今のスコアで予想をした時のインデックスを取得
                #target_idx = np.where(target_score_list == sc)[0]
                if ind == -1: #予想がない場合
                    if bet_price[shift] != 0 :
                        #ポジションがあり、予想がない場合はその日の最後の予想だったので決済する
                        #print("predict not exist! score:", sc)
                        c = now_price - bet_price[shift]
                        if bet_type[shift] == "BUY":
                            profit = (c - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
                        elif bet_type[shift] == "SELL":
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
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
                        profit_per_shift[shift] = profit_per_shift[shift] + profit
                    continue

                if LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION":
                    pred = predict_list[ind, :]
                    mid = float(Decimal(str(now_price)) * (Decimal(str(pred[0])) / Decimal("10000") + Decimal("1")))
                else:
                    # 予想取得
                    pred = predict_list[ind, :]
                    max = pred.argmax()
                    percent = pred[max]

                    # ベット延長用
                    ext_pred = ext_predict_list[ind, :]
                    ext_max = ext_pred.argmax()
                    ext_percent = ext_pred[ext_max]

                # 今保持している建玉すべてでストップロス確認
                for sh in bet_price:

                    if bet_price[sh] != 0:
                        # 既にポジションがある場合
                        c = now_price - bet_price[sh]
                        if bet_type[sh] == "BUY":
                            if c <= FX_STOP_LOSS:
                                stop_loss_cnt += 1
                                #ストップロス以上の損失を出している場合は決済する
                                profit = (FX_STOP_LOSS - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[sh] = 0
                                bet_type[sh] = ""
                                bet_len_dict[bet_len[sh]] = bet_len_dict[bet_len[sh]] + 1 if bet_len_dict.get(bet_len[sh],
                                                                                                      0) != 0 else 1
                                bet_len[sh] = 0
                                profit_per_shift[sh] = profit_per_shift[sh] + profit
                                if shift == sh:
                                    bet_type_prev_shift = "BUY"

                        elif bet_type[sh] == "SELL":
                            if (c * -1) <= FX_STOP_LOSS:
                                stop_loss_cnt += 1
                                #ストップロス以上の損失を出している場合は決済する
                                profit = (FX_STOP_LOSS - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[sh] = 0
                                bet_type[sh] = ""
                                bet_len_dict[bet_len[sh]] = bet_len_dict[bet_len[sh]] + 1 if bet_len_dict.get(bet_len[sh],
                                                                                                      0) != 0 else 1
                                bet_len[sh] = 0
                                profit_per_shift[sh] = profit_per_shift[sh] + profit
                                if shift == sh:
                                    bet_type_prev_shift = "SELL"
                        else:
                            #想定外エラー
                            print("ERROR2")
                            sys.exit()

                if bet_price[shift] != 0:
                    #既にポジションがある場合
                    if ext_begin_len > bet_len[shift]:
                        #ベット延長するか判断できる秒数が経過していない場合
                        #ベットし続ける
                        bet_len[shift] += 1
                    else:
                        # ベット延長するか判断できる秒数が経過している場合
                        c = now_price - bet_price[shift]

                        if bet_type[shift] == "BUY":
                            #買いポジションがある場合
                            if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and now_price + FX_MORE_BORDER_RGR < mid) or \
                                    ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and ext_max == 0 and FX_MORE_BORDER_CAT < ext_percent) :
                                # 更に上がると予想されている場合、決済しないままとする
                                bet_len[shift] += 1
                            else:
                                #更に上がらなければ決済する
                                profit = (c - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[shift] = 0
                                bet_type[shift] = ""
                                bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                      0) != 0 else 1
                                bet_len[shift] = 0
                                profit_per_shift[shift] = profit_per_shift[shift] + profit
                                if shift == sh:
                                    bet_type_prev_shift = "BUY"

                        elif bet_type[shift] == "SELL":
                            #売りポジションがある場合
                            if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and now_price - FX_MORE_BORDER_RGR > mid) or \
                                    ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and  ext_max == 2 and FX_MORE_BORDER_CAT < ext_percent):
                                # 更に上がると予想されている場合、決済しないままとする
                                bet_len[shift] += 1
                            else:
                                #更に下がらなければ決済する
                                profit = ((c * -1) - float(Decimal("0.001") * Decimal(FX_SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[shift] = 0
                                bet_type[shift] = ""
                                bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(bet_len[shift],
                                                                                                      0) != 0 else 1
                                bet_len[shift] = 0
                                profit_per_shift[shift] = profit_per_shift[shift] + profit
                                if shift == sh:
                                    bet_type_prev_shift = "SELL"
                        else:
                            #想定外エラー
                            print("ERROR3")
                            sys.exit()

                if bet_price[shift] == 0:
                    #ポジションがない場合
                    if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and \
                            now_price + float(Decimal("0.001") * Decimal(str(FX_SPREAD)) + Decimal(str(border))) <= mid) or \
                        ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and max == 0 and percent >= border):

                        if bet_type_prev_shift != "BUY":
                            #すでに決済済みでない場合のみ新規ポジションをもつ
                            bet_price[shift] = now_price
                            bet_type[shift] = "BUY"
                            bet_cnt += 1
                            bet_len[shift] = 1
                    elif ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and \
                            now_price - float(Decimal("0.001") * Decimal(str(FX_SPREAD)) + Decimal(str(border))) >= mid) or \
                          ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and max == 2 and percent >= border):
                        if bet_type_prev_shift != "SELL":
                            #すでに決済済みでない場合のみ新規ポジションをもつ
                            bet_price[shift] = now_price
                            bet_type[shift] = "SELL"
                            bet_cnt += 1
                            bet_len[shift] = 1


            #print("loop cnt:", j)

            prev_money = START_MONEY


            for i, score in enumerate(score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)


            #for i in result_txt:
            #    myLogger(i)

            detail_profit = prev_money - START_MONEY
            print("Accuracy over ",border)
            print("bet cnt:", bet_cnt)
            print("Detail Earned Money: ", detail_profit)

            if max_result_detail["suffix"] == None or max_result_detail["profit"] < detail_profit:
                max_result_detail["suffix"] = suffix
                max_result_detail["profit"] = int(detail_profit)
                max_result_detail["border"] = str(border)

            #print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")
            if show_detail:
                fig = plt.figure()
                # 価格の遷移
                ax1 = fig.add_subplot(111)

                ax1.plot(close_list, 'g')

                ax2 = ax1.twinx()
                ax2.plot(money_y)

                plt.title(
                    'border:' + str(border) + " spread:" + str(FX_SPREAD) + " money:" + str(
                        money))

                print("stop loss cnt:", stop_loss_cnt)
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

                print("Shift毎の利益")
                for k, v in sorted(profit_per_shift.items()):
                    print(k, profit_per_shift.get(k, 0))

                plt.show()

    print("max_suffix:", max_result["suffix"])
    print("max_border:", max_result["border"])
    print("max_profit:", max_result["profit"])

    print("max_detail_suffix:", max_result_detail["suffix"])
    print("max_detail_border:", max_result_detail["border"])
    print("max_detail_profit:", max_result_detail["profit"])

if __name__ == "__main__":
    do_predict()

    print("END!!!")