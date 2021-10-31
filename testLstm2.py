import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
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


"""
LEARNING_TYPE == "CATEGORY"
LEARNING_TYPE == "CATEGORY_BIN_UP"
LEARNING_TYPE == "CATEGORY_BIN_DW"
の場合専用
"""

start = datetime(2021, 1, 1)
end = datetime(2021, 9, 30, 22)

#2秒ごとの成績を計算する場合
per_sec_flg = True

#border以上の予想パーセントをしたものから正解率と予想数と正解数を返す

def getAcc(res, border, dataY, bin_both_ind_up, bin_both_ind_dw):
    if LEARNING_TYPE == "CATEGORY":
        up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] >= res[:, 2]) & (res[:, 0] >= border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > res[:, 1]) & (res[:, 2] >= border))[0]
    elif LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        up_ind = bin_both_ind_up
        down_ind = bin_both_ind_dw
    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
        up_ind = np.where(res[:, 0] >= border)[0]
        down_ind = []
    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
        up_ind = []
        down_ind = np.where(res[:, 0] >= border)[0]

    x5_up = res[up_ind,:]
    if LEARNING_TYPE == "CATEGORY_BIN_UP":
        x5_up[:,0] = 1

    y5_up= dataY[up_ind,:]

    x5_down = res[down_ind,:]
    if LEARNING_TYPE == "CATEGORY_BIN_DW":
        x5_down[:,0] = 1

    y5_down= dataY[down_ind,:]

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

    return Acc, total_num, correct_num

#全体の正解率を返す

def getAccTotal(res, dataY):

    eq = np.equal(res.argmax(axis=1), dataY.argmax(axis=1))
    cor_length = int(len(np.where(eq == True)[0]))

    total_num = len(res)
    correct_num = cor_length

    if total_num ==0:
        Acc =0
    else:
        Acc = correct_num / total_num

    return Acc

def getAccFx(res, border, dataY, change):
    if LEARNING_TYPE == "CATEGORY":
        up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] > res[:, 2]) & (res[:, 0] > border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > res[:, 1]) & (res[:, 2] > border))[0]
    elif LEARNING_TYPE == "CATEGORY_BIN":
        up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] > border))[0]
        down_ind = np.where((res[:, 1] > res[:, 0])  & (res[:, 1] > border))[0]
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
        #最大ドローダウンを更新してしまった場合
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

    # 直近の変化率リスト
    target_divide_prev_list = np.array(dataSequence2.get_target_divide_prev_list())

    # 正解までのの変化率リスト
    target_divide_aft_list = np.array(dataSequence2.get_target_divide_aft_list())

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

    #model_suffix = ["90*13",]


    #target_spreadをkey,Valueは[UP,DW]それぞれのモデル
    cat_bin_both_models = [
            {0:[
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-7",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-35"
                    ],
            1:[
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001-40",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001-25"

                    ] ,
            2: [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-6",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.0001-33"
                    ],
            3:[
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-27",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001-40"
                    ],
            4:[
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-10",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-36"

                    ],
            5:[
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-10",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-13"

                    ],
            6: [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-19",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-29"

                    ],
            7: [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-38",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-5"

            ],
            8: [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-40",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-6"

            ],
            9: [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-39",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32"

            ],
            }
    ]
    # CATEGORY_BIN_BOTH用 スプレッドに対応する予想結果がない場合に対応させるtarget_spreadsのインデックス
    idx_other = 9 + 1  # リストの最初にtarget_spread_listが入っているのでプラス1する

    #border_list = [0.56,0.562,0.564,0.566,0.568,0.57,0.572,0.574,0.576,0.578,0.58,]  # Turbo用
    #border_list = [0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,]  # TurboSpread 0用
    #border_list = [0.46,0.47, 0.48,0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,  0.56,]  # TurboSpread 1用
    border_list = [ 0.45, 0.46,0.47, 0.48,0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55]   # TurboSpread 1 BIN_UP,BIN_DW用
    #border_list = [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, ]  # Turbo
    border_list_show = []

    #CATEGORY_BIN_BOTH用
    border_list_both = [ {-1:[0.50,0.50],0:[0.51,0.51],1:[0.48,0.48],2:[0.51,0.48],3:[0.49,0.50],4:[0.49,0.50],5:[0.52,0.50],6:[0.5,0.48],7:[0.50,0.50],8:[0.50,0.50],9:[0.52,0.50],}, ]  # BIN_BOTH用 それぞれのスプレッド対応モデルごとのborder(UP,DW) -1スプレッドは対応するモデルがない場合用
    border_list_show_both = border_list_both

    #border_list_show = [ 0.46,0.47, 0.48,0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,  0.56,]  # グラフも表示



    #FILE_PREFIX = "GBPJPY_CATEGORY_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU"

    print("FILE_PREFIX:", FILE_PREFIX)

    if LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        model_suffix = cat_bin_both_models
        border_list = border_list_both
        border_list_show = border_list_show_both

    total_acc_txt = []
    total_money_txt = []
    max_val_suffix = {"val":0,}

    for suffix in model_suffix:
        print(suffix)
        if LEARNING_TYPE == "CATEGORY_BIN_BOTH":
            #target_spreadのリスト
            target_spreads = []
            dic = sorted(suffix.items())

            #zip関数に渡すスプレッドと予想結果のリスト
            arg_lists = [target_spread_list]
            print("target_spread_list length:", len(target_spread_list))
            for k, v in dic:
                target_spreads.append(k)

                #CATEGORY_BIN_UP とDWで予想した結果を合わせる
                load_dir_up = "/app/model/bin_op/" + v[0]
                model_up = tf.keras.models.load_model(load_dir_up)

                load_dir_dw = "/app/model/bin_op/" + v[1]
                model_dw = tf.keras.models.load_model(load_dir_dw)

                # ndarrayで返って来る
                predict_list_up = model_up.predict_generator(dataSequence2,
                                                       steps=None,
                                                       max_queue_size=PROCESS_COUNT * 1,
                                                       use_multiprocessing=False,
                                                       verbose=0)

                predict_list_dw = model_dw.predict_generator(dataSequence2,
                                                       steps=None,
                                                       max_queue_size=PROCESS_COUNT * 1,
                                                       use_multiprocessing=False,
                                                       verbose=0)
                #SAMEの予想結果は0とする
                predict_list_zero = np.zeros((len(predict_list_up), 2))

                #UP,SAME,DWの予想結果を合算する
                all = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
                predict_list_tmp = all[:, [0, 2, 4]]
                arg_lists.append(predict_list_tmp)
                #print("predict_list_tmp length:",k, len(predict_list_tmp))
            #各スプレッドに対応する予想結果をまとめる
            predict_list = []


            print("target_spreads")
            print(target_spreads)
            """
            print("arg_lists[:20]")
            print(arg_lists[:20])
            """

            for j in zip(*arg_lists): #arg_listsを展開して渡す
                tmp_spr = j[0]
                if tmp_spr in target_spreads:
                    #target_spreadsの何番目に予想結果が入っているか取得
                    idx = target_spreads.index(tmp_spr) + 1 #リストの最初にtarget_spread_listが入っているのでプラス1する
                    predict_list.append(j[idx].tolist()) #numpy.arrayを一旦リストにして追加
                else:
                    #スプレッドに対応する予想結果がない場合
                    predict_list.append(j[idx_other].tolist())

            predict_list = np.array(predict_list) #listからnumpy.arrayにもどす
            #print(predict_list[:20])
        else:
            #suffix = "90*" + suffix

            load_dir = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix

            model = tf.keras.models.load_model(load_dir)

            # ndarrayで返って来る
            predict_list = model.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=PROCESS_COUNT * 1,
                                                   use_multiprocessing=False,
                                                   verbose=0)

        #model.summary()

        print("suffix:", suffix)

        # START tensorflow1で作成したモデル用
        """
        model_file = "/app/bin_op/model/GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5" + "." + suffix
        model = tf.keras.models.load_model(model_file)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        """
        # END tensorflow1で作成したモデル用

        under_dict = {}
        over_dict = {}
        line_val = 0.505
        #line_val = 0.582 #pyoutが950なので
        #line_val = 0.583

        """
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("pred_close", len(pred_close_list), pred_close_list[:10])
        """

        #print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

        r = redis.Redis(host='localhost', port=6379, db=DB_TRADE_NO)

        AccTotal = getAccTotal(predict_list, correct_list)
        total_acc_txt.append(str(AccTotal) )

        for border_ind, border in enumerate(border_list):

            # 予想結果表示用テキストを保持
            result_txt = []
            result_txt_trade = []

            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            max_drawdown_trade = 0
            drawdown_trade = 0
            max_drawdowns_trade = []

            #BIN_BOTHの場合、borderが一律でない為、先に対象indを求めておく
            bin_both_ind = []
            bin_both_ind_up = []
            bin_both_ind_dw = []
            if LEARNING_TYPE == "CATEGORY_BIN_BOTH":

                for l, m in enumerate(zip(target_spread_list, predict_list)):
                    spr_t = m[0]
                    pred_t = m[1]
                    if spr_t in border:
                        up_border = border[spr_t][0]
                        dw_border = border[spr_t][1]
                    else:
                        up_border = border[-1][0]
                        dw_border = border[-1][1]
                    if pred_t[0] > pred_t[1] and pred_t[0] >= pred_t[2] and  pred_t[0] >= up_border:
                        bin_both_ind_up.append(l)
                        bin_both_ind.append(l)
                    elif pred_t[2] > pred_t[0] and pred_t[2] >= pred_t[1] and  pred_t[2] >= dw_border:
                        bin_both_ind_dw.append(l)
                        bin_both_ind.append(l)

            if FX == False:
                Acc, total_num, correct_num = getAcc(predict_list, border, correct_list, bin_both_ind_up, bin_both_ind_dw)
                profit = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)
            else:
                Acc, total_num, correct_num, profit = getAccFx(predict_list, border, correct_list, change_list)

            #全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            if LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                result_txt.append("Accuracy border_ind " + str(border_ind) + ":" + str(Acc))
                result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
            else:
                result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

            if FX == False:
                result_txt.append("Earned Money:" + str(profit))

            else:
                result_txt.append("Earned Money:" + str(profit))

            if line_val >= Acc:
                under_dict["acc"] = Acc
                under_dict["money"] = profit
            else:
                if not "acc" in over_dict.keys():
                    over_dict["acc"] = Acc
                    over_dict["money"] = profit

            if border not in border_list_show:
                for i in result_txt:
                    print(i)
                continue

            if (LEARNING_TYPE == "CATEGORY_BIN_UP" or LEARNING_TYPE == "CATEGORY_BIN_DW" ):
                # up,downどれかが閾値以上の予想パーセントであるもののみ抽出
                ind = np.where(predict_list[:, 0] >= border)[0]
            elif LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                #up,downどちらかが閾値以上
                ind = bin_both_ind
            else:
                # up,same,downどれかが閾値以上の予想パーセントであるもののみ抽出
                ind = np.where(predict_list >=border)[0]

            x5 = predict_list[ind,:]
            y5 = correct_list[ind,:]
            s5 = target_score_list[ind]
            c5 = change_list[ind]
            sp5 = target_spread_list[ind]
            sc5 = pred_close_list[ind]
            ec5 = real_close_list[ind]
            dp5 = target_divide_prev_list[ind]
            da5 = target_divide_aft_list[ind]

            """
            up = predict_list[:, 0]
            down = predict_list[:, 2]
    
            up_ind = np.where(up >= border)[0]
            down_ind = np.where(down >= border)[0]
    
            x_up = predict_list[up_ind,:]
            y_up= predict_list[up_ind,:]
    
            up_total_length = len(x_up)
            up_eq = np.equal(x_up.argmax(axis=1), y_up.argmax(axis=1))
    
            #upと予想して正解だった数
            up_cor_length = int(len(np.where(up_eq == True)[0]))
            #upと予想して不正解だった数
            up_wrong_length = int(up_total_length - up_cor_length)
    
            x_down = predict_list[down_ind,:]
            y_down= predict_list[down_ind,:]
    
            down_total_length = len(x_down)
            down_eq = np.equal(x_down.argmax(axis=1), y_down.argmax(axis=1))
    
            #downと予想して正解だった数
            down_cor_length = int(len(np.where(down_eq == True)[0]))
            #downと予想して不正解だった数
            down_wrong_length = int(down_total_length - down_cor_length)
            """

            money_y = []
            money_trade_y = []
            money_tmp = {}
            money_trade_tmp = {}

            money = START_MONEY #始めの所持金
            money_trade = START_MONEY

            cnt_up_cor = 0 #upと予想して正解した数
            cnt_up_wrong = 0 #upと予想して不正解だった数

            cnt_down_cor = 0 #downと予想して正解した数
            cnt_down_wrong = 0 #downと予想して不正解だった数

            spread_trade = {}
            spread_win = {}
            spread_trade_up = {}
            spread_win_up = {}
            spread_trade_dw = {}
            spread_win_dw = {}

            spread_trade_real = {}
            spread_win_real = {}
            spread_trade_real_up = {}
            spread_win_real_up = {}
            spread_trade_real_dw = {}
            spread_win_real_dw = {}

            divide_trade = {}
            divide_win = {}

            divide_trade_real = {}
            divide_win_real = {}

            divide_aft_trade = {}
            divide_aft_win = {}

            divide_aft_trade_real = {}
            divide_aft_win_real = {}

            cnt = 0

            # 2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
            # {key:[trade_num,win_num]}
            per_sec = 2
            per_sec_dict = {}
            per_sec_dict_real = {}
            if per_sec_flg:
                for i in range(60):
                    if i % per_sec == 0:
                        per_sec_dict[i] = [0, 0]
                        per_sec_dict_real[i] = [0, 0]
            # 分ごとの成績
            per_min_dict = {}
            per_min_dict_real = {}
            for i in range(60):
                per_min_dict[i] = [0, 0]
                per_min_dict_real[i] = [0, 0]

            # 時間ごとの成績
            per_hour_dict = {}
            per_hour_dict_real = {}
            for i in range(24):
                per_hour_dict[i] = [0, 0]
                per_hour_dict_real[i] = [0, 0]

            # 理論上の予測確率ごとの勝率 key:確率 val:{win_cnt:勝った数, lose_cnt:負けた数}
            prob_list = {}
            # 実際のトレードの予測確率ごとの勝率 key:確率 val:{win_cnt:勝った数, lose_cnt:負けた数}
            prob_real_list = {}

            true_cnt = 0

            trade_cnt = 0
            trade_win_cnt = 0
            trade_wrong_win_cnt = 0
            trade_wrong_lose_cnt = 0

            for x, y, s, c, sp, sc, ec, dp, da in zip(x5, y5, s5, c5, sp5, sc5, ec5, dp5, da5):
                cnt += 1
                max = x.argmax()
                if (LEARNING_TYPE == "CATEGORY_BIN_UP" or LEARNING_TYPE == "CATEGORY_BIN_DW"):
                    max = 0
                probe_float = x[max]
                probe = str(x[max])
                percent = probe[0:4]

                #予想した時間
                predict_t = datetime.fromtimestamp(s)
                win_flg = False
                win_trade_flg = False

                startVal = "NULL"
                endVal = "NULL"
                result = "NULL"
                correct = "NULL"

                tradeReult = []

                if ((LEARNING_TYPE == "CATEGORY_BIN_UP" or LEARNING_TYPE == "CATEGORY_BIN_DW" ) and max == 0)  \
                        or (LEARNING_TYPE == "CATEGORY" and (max == 0 or max == 2))\
                        or (LEARNING_TYPE == "CATEGORY_BIN_BOTH" and (max == 0 or max == 2)):
                    win_flg = True if max == y.argmax() else False

                    if TRADE_FLG:
                        tradeReult = r.zrangebyscore(DB_TRADE_NAME, s, s)
                        if len(tradeReult) == 0:
                            # 取引履歴がない場合1秒後の履歴として残っているかもしれないので取得
                            tradeReult = r.zrangebyscore(DB_TRADE_NAME, s + 1, s + 1)

                        if len(tradeReult) != 0:
                            trade_cnt = trade_cnt + 1
                            tmps = json.loads(tradeReult[0].decode('utf-8'))
                            startVal = tmps.get("startVal")
                            endVal = tmps.get("endVal")
                            result = tmps.get("result")
                            if result == "win":
                                win_trade_flg = True
                                trade_win_cnt = trade_win_cnt + 1
                                money_trade = money_trade + PAYOUT
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade, drawdown_trade, PAYOUT)
                            else:
                                win_trade_flg = False
                                money_trade = money_trade - PAYOFF
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade, drawdown_trade, PAYOFF * -1)

                    # 理論上のスプレッド毎の勝率
                    flg = False
                    for k, v in SPREAD_LIST.items():
                        if sp > v[0] and sp <= v[1]:
                            spread_trade[k] = spread_trade.get(k, 0) + 1
                            if win_flg:
                                spread_win[k] = spread_win.get(k, 0) + 1

                            if max == 0:
                                spread_trade_up[k] = spread_trade_up.get(k, 0) + 1
                                if win_flg:
                                    spread_win_up[k] = spread_win_up.get(k, 0) + 1
                            elif max == 2:
                                spread_trade_dw[k] = spread_trade_dw.get(k, 0) + 1
                                if win_flg:
                                    spread_win_dw[k] = spread_win_dw.get(k, 0) + 1

                            flg = True
                            break

                    if flg == False:
                        if sp < 0:
                            spread_trade["spread0"] = spread_trade.get("spread0", 0) + 1
                            if win_flg:
                                spread_win["spread0"] = spread_win.get("spread0", 0) + 1

                            if max == 0:
                                spread_trade_up["spread0"] = spread_trade_up.get("spread0", 0) + 1
                                if win_flg:
                                    spread_win_up["spread0"] = spread_win_up.get("spread0", 0) + 1
                            elif max == 2:
                                spread_trade_dw["spread0"] = spread_trade_dw.get("spread0", 0) + 1
                                if win_flg:
                                    spread_win_dw["spread0"] = spread_win_dw.get("spread0", 0) + 1

                        else:
                            spread_trade["spread16Over"] = spread_trade.get("spread16Over", 0) + 1
                            if win_flg:
                                spread_win["spread16Over"] = spread_win.get("spread16Over", 0) + 1

                            if max == 0:
                                spread_trade_up["spread16Over"] = spread_trade_up.get("spread16Over", 0) + 1
                                if win_flg:
                                    spread_win_up["spread16Over"] = spread_win_up.get("spread16Over", 0) + 1
                            elif max == 2:
                                spread_trade_dw["spread16Over"] = spread_trade_dw.get("spread16Over", 0) + 1
                                if win_flg:
                                    spread_win_dw["spread16Over"] = spread_win_dw.get("spread16Over", 0) + 1

                    # 実際のスプレッド毎の勝率
                    if len(tradeReult) != 0:
                        flg = False
                        for k, v in SPREAD_LIST.items():
                            if sp > v[0] and sp <= v[1]:
                                spread_trade_real[k] = spread_trade_real.get(k, 0) + 1
                                if win_trade_flg:
                                    spread_win_real[k] = spread_win_real.get(k, 0) + 1

                                if max == 0:
                                    spread_trade_real_up[k] = spread_trade_real_up.get(k, 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_up[k] = spread_win_real_up.get(k, 0) + 1
                                elif max == 2:
                                    spread_trade_real_dw[k] = spread_trade_real_dw.get(k, 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_dw[k] = spread_win_real_dw.get(k, 0) + 1

                                flg = True
                                break
                        if flg == False:
                            if sp < 0:
                                spread_trade_real["spread0"] = spread_trade_real.get("spread0", 0) + 1
                                if win_trade_flg:
                                    spread_win_real["spread0"] = spread_win_real.get("spread0", 0) + 1

                                if max == 0:
                                    spread_trade_real_up["spread0"] = spread_trade_real_up.get("spread0", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_up["spread0"] = spread_win_real_up.get("spread0", 0) + 1
                                elif max == 2:
                                    spread_trade_real_dw["spread0"] = spread_trade_real_dw.get("spread0", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_dw["spread0"] = spread_win_real_dw.get("spread0", 0) + 1

                            else:
                                spread_trade_real["spread16Over"] = spread_trade_real.get("spread16Over", 0) + 1
                                if win_trade_flg:
                                    spread_win_real["spread16Over"] = spread_win_real.get("spread16Over", 0) + 1

                                if max == 0:
                                    spread_trade_real_up["spread16Over"] = spread_trade_real_up.get("spread16Over", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_up["spread16Over"] = spread_win_real_up.get("spread16Over", 0) + 1
                                elif max == 2:
                                    spread_trade_real_dw["spread16Over"] = spread_trade_real_dw.get("spread16Over", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real_dw["spread16Over"] = spread_win_real_dw.get("spread16Over", 0) + 1

                    # 理論上の秒毎の勝率
                    if per_sec_flg:
                        per_sec_dict[predict_t.second][0] += 1
                        if win_flg:
                            per_sec_dict[predict_t.second][1] += 1

                        # 実際の秒毎の勝率
                        if len(tradeReult) != 0:
                            per_sec_dict_real[predict_t.second][0] += 1
                            if win_trade_flg:
                                per_sec_dict_real[predict_t.second][1] += 1

                    # 分ごと及び、分秒ごとのbetした回数と勝ち数を保持
                    per_min_dict[predict_t.minute][0] += 1
                    # per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][0] += 1
                    if win_flg:
                        per_min_dict[predict_t.minute][1] += 1
                        # per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][1] += 1
                    if len(tradeReult) != 0:
                        per_min_dict_real[predict_t.minute][0] += 1
                        if win_trade_flg:
                            per_min_dict_real[predict_t.minute][1] += 1

                    # 時間ごとのbetした回数と勝ち数を保持
                    per_hour_dict[predict_t.hour][0] += 1
                    if win_flg:
                        per_hour_dict[predict_t.hour][1] += 1
                    if len(tradeReult) != 0:
                        per_hour_dict_real[predict_t.hour][0] += 1
                        if win_trade_flg:
                            per_hour_dict_real[predict_t.hour][1] += 1

                    # 理論上の直近変化率毎の勝率
                    flg = False
                    for k, v in DIVIDE_LIST.items():
                        if dp > v[0] and dp <= v[1]:
                            divide_trade[k] = divide_trade.get(k, 0) + 1
                            if win_flg:
                                divide_win[k] = divide_win.get(k, 0) + 1
                            flg = True
                            break

                    if flg == False:
                        #変化率が10000以上の場合
                        divide_trade["divide13over"] = divide_trade.get("divide13over", 0) + 1
                        if win_flg:
                            divide_win["divide13over"] = divide_win.get("divide13over", 0) + 1

                    # 実際の直近変化率毎の勝率
                    if len(tradeReult) != 0:
                        flg = False
                        for k, v in DIVIDE_LIST.items():
                            if dp > v[0] and dp <= v[1]:
                                divide_trade_real[k] = divide_trade_real.get(k, 0) + 1
                                if win_trade_flg:
                                    divide_win_real[k] = divide_win_real.get(k, 0) + 1
                                flg = True
                                break

                        if flg == False:
                            # 変化率が10000以上の場合
                            divide_trade_real["divide13over"] = divide_trade_real.get("divide13over", 0) + 1
                            if win_trade_flg:
                                divide_win_real["divide13over"] = divide_win_real.get("divide13over", 0) + 1


                    # 理論上の正解までの変化率毎の勝率
                    flg = False
                    for k, v in DIVIDE_AFT_LIST.items():
                        if da > v[0] and da <= v[1]:
                            divide_aft_trade[k] = divide_aft_trade.get(k, 0) + 1
                            if win_flg:
                                divide_aft_win[k] = divide_aft_win.get(k, 0) + 1
                            flg = True
                            break

                    if flg == False:
                        #変化率が9以上の場合
                        divide_aft_trade["divide13over"] = divide_aft_trade.get("divide13over", 0) + 1
                        if win_flg:
                            divide_aft_win["divide13over"] = divide_aft_win.get("divide13over", 0) + 1

                    # 実際の正解までの変化率毎の勝率
                    if len(tradeReult) != 0:
                        flg = False
                        for k, v in DIVIDE_AFT_LIST.items():
                            if da > v[0] and da <= v[1]:
                                divide_aft_trade_real[k] = divide_aft_trade_real.get(k, 0) + 1
                                if win_trade_flg:
                                    divide_aft_win_real[k] = divide_aft_win_real.get(k, 0) + 1
                                flg = True
                                break

                        if flg == False:
                            # 変化率が9以上の場合
                            divide_aft_trade_real["divide13over"] = divide_aft_trade_real.get("divide13over", 0) + 1
                            if win_trade_flg:
                                divide_aft_win_real["divide13over"] = divide_aft_win_real.get("divide13over", 0) + 1


                    tmp_prob_cnt_list = {}
                    tmp_prob_real_cnt_list = {}
                    # 確率ごとのトレード数および勝率を求めるためのリスト
                    if percent in prob_list.keys():
                        tmp_prob_list = prob_list[percent]
                        if win_flg:
                            tmp_prob_cnt_list["win_cnt"] = tmp_prob_list["win_cnt"] + 1
                            tmp_prob_cnt_list["lose_cnt"] = tmp_prob_list["lose_cnt"]
                        else:
                            tmp_prob_cnt_list["win_cnt"] = tmp_prob_list["win_cnt"]
                            tmp_prob_cnt_list["lose_cnt"] = tmp_prob_list["lose_cnt"] + 1
                        prob_list[percent] = tmp_prob_cnt_list
                    else:
                        if win_flg:
                            tmp_prob_cnt_list["win_cnt"] = 1
                            tmp_prob_cnt_list["lose_cnt"] = 0
                        else:
                            tmp_prob_cnt_list["win_cnt"] = 0
                            tmp_prob_cnt_list["lose_cnt"] = 1
                        prob_list[percent] = tmp_prob_cnt_list
                    # トレードした場合
                    if len(tradeReult) != 0:
                        # 確率ごとのトレード数および勝率を求めるためのリスト
                        if percent in prob_real_list.keys():
                            tmp_prob_real_list = prob_real_list[percent]
                            if win_trade_flg:
                                tmp_prob_real_cnt_list["win_cnt"] = tmp_prob_real_list["win_cnt"] + 1
                                tmp_prob_real_cnt_list["lose_cnt"] = tmp_prob_real_list["lose_cnt"]
                            else:
                                tmp_prob_real_cnt_list["win_cnt"] = tmp_prob_real_list["win_cnt"]
                                tmp_prob_real_cnt_list["lose_cnt"] = tmp_prob_real_list["lose_cnt"] + 1
                            prob_real_list[percent] = tmp_prob_real_cnt_list
                        else:
                            if win_flg:
                                tmp_prob_real_cnt_list["win_cnt"] = 1
                                tmp_prob_real_cnt_list["lose_cnt"] = 0
                            else:
                                tmp_prob_real_cnt_list["win_cnt"] = 0
                                tmp_prob_real_cnt_list["lose_cnt"] = 1
                            prob_real_list[percent] = tmp_prob_real_cnt_list

                if (LEARNING_TYPE == "CATEGORY" and max == 0) or (LEARNING_TYPE == "CATEGORY_BIN_BOTH" and max == 0) or (LEARNING_TYPE == "CATEGORY_BIN_UP" and max == 0) :
                    # Up predict

                    if FX:
                        profit = (c - float( Decimal("0.001") * Decimal(SPREAD) )) * FX_POSITION
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                    if max == y.argmax():
                        if FX == False:
                            money = money + PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                        cnt_up_cor = cnt_up_cor + 1

                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                #理論上の結果と実トレードの結果がおなじ
                                correct = "TRUE"
                                true_cnt = true_cnt + 1
                            else:
                                correct = "FALSE"
                                trade_wrong_lose_cnt += 1
                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(ec) + "," + "UP" + ","
                                          + probe + "," + "win" + "," + startVal + "," + endVal
                                          + "," + result + "," + correct)

                    else :
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOFF * -1)

                        cnt_up_wrong = cnt_up_wrong + 1

                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                correct = "FALSE"
                                trade_wrong_win_cnt += 1
                            else:
                                correct = "TRUE"
                                true_cnt = true_cnt + 1

                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(
                                ec) + "," + "UP" + ","
                                              + probe + "," + "lose" + "," + startVal + "," + endVal
                                              + "," + result + "," + correct)

                elif (LEARNING_TYPE == "CATEGORY" and max == 2) or (LEARNING_TYPE == "CATEGORY_BIN_BOTH" and max == 2)  or (LEARNING_TYPE == "CATEGORY_BIN_DW" and max == 0) :
                    #Down predict
                    if FX:
                        #cはaft-befなのでdown予想の場合の利益として-1を掛ける
                        profit = ((c * -1)  - float( Decimal("0.001") * Decimal(SPREAD) )) * FX_POSITION
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                    if max == y.argmax():
                        if FX == False:
                            money = money + PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                        cnt_down_cor = cnt_down_cor + 1


                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                #理論上の結果と実トレードの結果がおなじ
                                correct = "TRUE"
                                true_cnt = true_cnt + 1
                            else:
                                correct = "FALSE"
                                trade_wrong_lose_cnt += 1
                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(ec) + "," + "DOWN" + ","
                                          + probe + "," + "win" + "," + startVal + "," + endVal
                                          + "," + result + "," + correct)

                    else:
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOFF * -1)

                        cnt_down_wrong = cnt_down_wrong + 1

                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                correct = "FALSE"
                                trade_wrong_win_cnt += 1
                            else:
                                correct = "TRUE"
                                true_cnt = true_cnt + 1

                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(
                                ec) + "," + "DOWN" + ","
                                              + probe + "," + "lose" + "," + startVal + "," + endVal
                                              + "," + result + "," + correct)

                money_tmp[s] = money
                if TRADE_FLG:
                    money_trade_tmp[s] = money_trade

            prev_money = START_MONEY
            prev_trade_money = START_MONEY

            print("cnt:", cnt)

            for i, score in enumerate(score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)

            if TRADE_FLG:
                for i, score in enumerate(score_list):
                    if score in money_trade_tmp.keys():
                        prev_trade_money = money_trade_tmp[score]

                    money_trade_y.append(prev_trade_money)

            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")


            fig = plt.figure()
            #価格の遷移
            ax1 = fig.add_subplot(111)

            ax1.plot(close_list, 'g')

            ax2 = ax1.twinx()
            ax2.plot(money_y)
            if TRADE_FLG:
                ax2.plot(money_trade_y, "r")

            if FX == True:
                plt.title('border:' + str(border) + " position:" + str(FX_POSITION) + " spread:" + str(SPREAD) + " money:" + str(money))
            else:
                if LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                    plt.title(
                        'border_ind:' + str(border_ind) + " payout:" + str(PAYOUT) + " spread:" + str(SPREAD) + " money:" + str(
                            money))
                else:
                    plt.title('border:' + str(border) + " payout:" + str(PAYOUT) + " spread:" + str(SPREAD) + " money:" + str(money))

            """
            for txt in result_txt_trade:
                res = txt.find("FALSE")
                if res != -1:
                    print(txt)
            """
            # print('\n'.join(result_txt))

            if trade_cnt != 0:
                print("trade cnt: " + str(trade_cnt))
                print("trade correct: " + str(true_cnt / trade_cnt))
                print("trade wrong cnt: " + str(trade_cnt - true_cnt))
                print("trade wrong win cnt: " + str(trade_wrong_win_cnt))
                print("trade wrong lose cnt: " + str(trade_wrong_lose_cnt))
                print("trade accuracy: " + str(trade_win_cnt / trade_cnt))
                print("trade money: " + str(prev_trade_money))
                print("trade cnt rate: " + str(trade_cnt / total_num))

            print("predict money: " + str(prev_money))
            for i in result_txt:
                print(i)

            print("理論上のスプレッド毎の勝率(全体)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade.get(k, 0),
                          " profit:", spread_win.get(k, 0) * PAYOUT - (spread_trade.get(k) - spread_win.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win.get(k, 0) / spread_trade.get(k))
                else:
                    print(k, " cnt:", spread_trade.get(k, 0))

            print("理論上のスプレッド毎の勝率(UP)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_up.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_up.get(k, 0),
                          " profit:", spread_win_up.get(k, 0) * PAYOUT - (spread_trade_up.get(k) - spread_win_up.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_up.get(k, 0) / spread_trade_up.get(k))
                else:
                    print(k, " cnt:", spread_trade_up.get(k, 0))

            print("理論上のスプレッド毎の勝率(DW)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_dw.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_dw.get(k, 0),
                          " profit:", spread_win_dw.get(k, 0) * PAYOUT - (spread_trade_dw.get(k) - spread_win_dw.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_dw.get(k, 0) / spread_trade_dw.get(k))
                else:
                    print(k, " cnt:", spread_trade_dw.get(k, 0))

            if TRADE_FLG:
                print("実トレード上のスプレッド毎の勝率(全体)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real.get(k, 0),
                              " profit:", spread_win_real.get(k, 0) * PAYOUT - (spread_trade_real.get(k) - spread_win_real.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real.get(k, 0) / spread_trade_real.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(UP)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_up.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_up.get(k, 0),
                              " profit:", spread_win_real_up.get(k, 0) * PAYOUT - (spread_trade_real_up.get(k) - spread_win_real_up.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_up.get(k, 0) / spread_trade_real_up.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_up.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(DW)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_dw.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_dw.get(k, 0),
                              " profit:", spread_win_real_dw.get(k, 0) * PAYOUT - (spread_trade_real_dw.get(k) - spread_win_real_dw.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_dw.get(k, 0) / spread_trade_real_dw.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_dw.get(k, 0))

                print("スプレッド毎の約定率")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real.get(k, 0), " rate:",
                              spread_trade_real.get(k) / spread_trade.get(k))


            if per_sec_flg:
                # 理論上の秒ごとの勝率
                per_sec_winrate_dict = {}
                for i in per_sec_dict.keys():
                    if per_sec_dict[i][0] != 0:
                        win_rate = per_sec_dict[i][1] / per_sec_dict[i][0]
                        per_sec_winrate_dict[i] = (win_rate,per_sec_dict[i][0])
                    else:
                        per_sec_winrate_dict[i] = (0,0)

                print("理論上の秒毎の勝率悪い順:" )
                worst_sorted = sorted(per_sec_winrate_dict.items(), key=lambda x: x[1][0])
                for i in worst_sorted:
                    print(i[0], i[1][0], i[1][1])

                if TRADE_FLG:
                    # 実際の秒ごとの勝率
                    per_sec_winrate_dict_real = {}
                    for i in per_sec_dict_real.keys():
                        if per_sec_dict_real[i][0] != 0:
                            win_rate = per_sec_dict_real[i][1] / per_sec_dict_real[i][0]
                            per_sec_winrate_dict_real[i] = (win_rate, per_sec_dict_real[i][0])
                        else:
                            per_sec_winrate_dict_real[i] = (0,0)

                    print("実際の秒毎の勝率悪い順:")
                    worst_sorted = sorted(per_sec_winrate_dict_real.items(), key=lambda x: x[1][0])
                    for i in worst_sorted:
                        print(i[0], i[1][0], i[1][1])

            # 理論上の分ごとの勝率
            per_min_winrate_dict = {}
            for i in per_min_dict.keys():
                if per_min_dict[i][0] != 0:
                    win_rate = per_min_dict[i][1] / per_min_dict[i][0]
                    per_min_winrate_dict[i] = (win_rate,per_min_dict[i][0])
                else:
                    per_min_winrate_dict[i] = (0,0)

            print("理論上の分毎の勝率悪い順:")
            worst_sorted = sorted(per_min_winrate_dict.items(), key=lambda x: x[1][0])
            for i in worst_sorted:
                print(i[0], i[1][0], i[1][1])

            if TRADE_FLG:
                # 実際の分ごとの勝率
                per_min_winrate_dict_real = {}
                for i in per_min_dict_real.keys():
                    if per_min_dict_real[i][0] != 0:
                        win_rate = per_min_dict_real[i][1] / per_min_dict_real[i][0]
                        per_min_winrate_dict_real[i] = (win_rate, per_min_dict_real[i][0])
                    else:
                        per_min_winrate_dict_real[i] = (0, 0)

                print("実際の分毎の勝率悪い順:")
                worst_sorted = sorted(per_min_winrate_dict_real.items(), key=lambda x: x[1][0])
                for i in worst_sorted:
                    print(i[0], i[1][0], i[1][1])

            # 理論上の時間ごとの勝率
            per_hour_winrate_dict = {}
            for i in per_hour_dict.keys():
                if per_hour_dict[i][0] != 0:
                    win_rate = per_hour_dict[i][1] / per_hour_dict[i][0]
                    per_hour_winrate_dict[i] = (win_rate,per_hour_dict[i][0])
                else:
                    per_hour_winrate_dict[i] = (0,0)

            print("理論上の時間毎の勝率悪い順:")
            worst_sorted = sorted(per_hour_winrate_dict.items(), key=lambda x: x[1][0])
            for i in worst_sorted:
                print(i[0], i[1][0], i[1][1])

            if TRADE_FLG:
                # 実際の分ごとの勝率
                per_hour_winrate_dict_real = {}
                for i in per_hour_dict_real.keys():
                    if per_hour_dict_real[i][0] != 0:
                        win_rate = per_hour_dict_real[i][1] / per_hour_dict_real[i][0]
                        per_hour_winrate_dict_real[i] = (win_rate, per_hour_dict_real[i][0])
                    else:
                        per_hour_winrate_dict_real[i] = (0, 0)

                print("実際の時間毎の勝率悪い順:")
                worst_sorted = sorted(per_hour_winrate_dict_real.items(), key=lambda x: x[1][0])
                for i in worst_sorted:
                    print(i[0], i[1][0], i[1][1])

            print("理論上の直近変化率毎の勝率")
            for k, v in sorted(DIVIDE_LIST.items()):
                if divide_trade.get(k, 0) != 0:
                    print(k, " cnt:", divide_trade.get(k, 0), " win rate:", divide_win.get(k, 0) / divide_trade.get(k))
                else:
                    print(k, " cnt:", divide_trade.get(k, 0))

            if TRADE_FLG:
                print("実際の直近変化率毎の勝率")
                for k, v in sorted(DIVIDE_LIST.items()):
                    if divide_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", divide_trade_real.get(k, 0), " win rate:", divide_win_real.get(k, 0) / divide_trade_real.get(k))
                    else:
                        print(k, " cnt:", divide_trade_real.get(k, 0))


            print("理論上の正解までの変化率毎の勝率")
            for k, v in sorted(DIVIDE_AFT_LIST.items()):
                if divide_aft_trade.get(k, 0) != 0:
                    print(k, " cnt:", divide_aft_trade.get(k, 0), " win rate:", divide_aft_win.get(k, 0) / divide_aft_trade.get(k))
                else:
                    print(k, " cnt:", divide_aft_trade.get(k, 0))

            if TRADE_FLG:
                print("実際の正解までの変化率毎の勝率")
                for k, v in sorted(DIVIDE_AFT_LIST.items()):
                    if divide_aft_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", divide_aft_trade_real.get(k, 0), " win rate:", divide_aft_win_real.get(k, 0) / divide_aft_trade_real.get(k))
                    else:
                        print(k, " cnt:", divide_aft_trade_real.get(k, 0))



            print("MAX DrawDowns(理論上のドローダウン)")
            max_drawdowns.sort()
            print(max_drawdowns[0:10])

            drawdown_cnt = {}
            for i in max_drawdowns:
                for k, v in DRAWDOWN_LIST.items():
                    if i < v[0] and i >= v[1]:
                        drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                        break
            for k, v in sorted(DRAWDOWN_LIST.items()):
                print(k, drawdown_cnt.get(k,0))

            if TRADE_FLG:
                print("MAX DrawDowns(実トレードのドローダウン)")
                max_drawdowns_trade.sort()
                print(max_drawdowns_trade[0:10])
                drawdown_cnt = {}
                for i in max_drawdowns_trade:
                    for k, v in DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                            break
                for k, v in sorted(DRAWDOWN_LIST.items()):
                    print(k, drawdown_cnt.get(k,0))


            for k, v in sorted(prob_list.items()):
                # 勝率
                win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                print("理論上の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))


            if TRADE_FLG:
                for k, v in sorted(prob_real_list.items()):
                    # トレードできた確率
                    trade_rate = (v["win_cnt"] + v["lose_cnt"]) / (prob_list[k]["win_cnt"] + prob_list[k]["lose_cnt"])
                    # 勝率
                    win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                    print("実トレード上の確率:" + k + " トレードできた割合:" + str(trade_rate) + " 勝率:" + str(win_rate))

            """
            if per_sec_flg:
                # 理論上の秒ごとの勝率
                for i in per_sec_dict.keys():
                    if per_sec_dict[i][0] != 0:
                        win_rate = per_sec_dict[i][1] / per_sec_dict[i][0]
                        print("理論上の秒毎の確率:" + str(i) + " トレード数:" + str(per_sec_dict[i][0]) + " 勝率:" + str(win_rate))

                if TRADE_FLG:
                    # 実際の秒ごとの勝率
                    for i in per_sec_dict_real.keys():
                        if per_sec_dict_real[i][0] != 0:
                            win_rate = per_sec_dict_real[i][1] / per_sec_dict_real[i][0]
                            print("実際の秒毎の確率:" + str(i) + " トレード数:" + str(per_sec_dict_real[i][0]) + " 勝率:" + str(win_rate))
            """
            #plt.show()
        if LEARNING_TYPE != "CATEGORY_BIN_BOTH":
            if "acc" in under_dict.keys() and "acc" in over_dict.keys():
                tmp_val = under_dict["money"] - (
                            (under_dict["money"] - over_dict["money"]) / (over_dict["acc"] - under_dict["acc"]) * (
                                line_val - under_dict["acc"]))
                print("Acc:", line_val, " Money:", tmp_val)
                total_money_txt.append(tmp_val)
                if max_val_suffix["val"] < tmp_val:
                    max_val_suffix["val"] = tmp_val
                    max_val_suffix["suffix"] = suffix
            else:
                print("Acc:", line_val, " Money:")
                total_money_txt.append("")

    if LEARNING_TYPE != "CATEGORY_BIN_BOTH":
        if "suffix" in max_val_suffix.keys():
            print("max_val_suffix:", max_val_suffix["suffix"])

        print("total_acc_txt")
        for i in total_acc_txt:
            print(i)

        print("total_money_txt")
        for i in total_money_txt:
            if i != "":
                print(int(i))
            else:
                print("")

if __name__ == "__main__":

    #print("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)
    do_predict()

    print("END!!!")