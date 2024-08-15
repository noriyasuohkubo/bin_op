import numpy as np
import redis
import json
from matplotlib import pyplot as plt
from datetime import datetime
import time

from readConf2 import *

from silence_tensorflow import silence_tensorflow
silence_tensorflow() #ログ抑制 import tensorflowの前におく

import tensorflow as tf

from DataSequence2 import DataSequence2
from operator import itemgetter

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


"""
#LEARNING_TYPE = "CATEGORY_BIN_FOUR" #2クラス分類 
UP,DOWNそれぞれを1分の前半、後半の秒を使って学習させたモデルを使ってテストする用　学習するためではない
"""

start = datetime(2021, 1, 1,)
end = datetime(2022, 3, 1, )

#2秒ごとの成績を計算する場合
per_sec_flg = True

#border以上の予想パーセントをしたものから正解率と予想数と正解数を返す

def getAcc(res, border, dataY, bin_both_ind_up, bin_both_ind_dw):

    up_ind = bin_both_ind_up
    down_ind = bin_both_ind_dw

    x5_up = res[up_ind,:]
    y5_up= dataY[up_ind,:]

    x5_down = res[down_ind,:]
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

    #target_spreadをkey,Valueは[ [UPのEXCSEC-32-0, DWのEXCSEC-32-0], [UPのEXCSEC-2-30, DWのEXCSEC-2-30] ]それぞれのモデル
    model_suffix = [
            {
            0:[
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-7",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-35"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-7",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.0001-35"
                ]
            ],
            1:[
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-16"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-13",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-16"
                ]
            ],
            2:[
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_EXCSEC-32-0_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-20",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-17"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_EXCSEC-32-0_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-34",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.0001_LOSS-C-ENTROPY_GU-GU-GU-3"
                ]
            ],
            3: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-20",
               ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-5",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-33",
                ]
            ],
            4: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-8",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-23",
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-10",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-23",
                ]
            ],
            5: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-40",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-24",
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-16",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37",
                ]
            ],
            6: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-21",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-32-0_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-36"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-24",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-34"
                ]
            ],
            7: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-27",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-7"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD8_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-22",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD8_UB1_202101_90_EXCSEC-2-30_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-8"
                ]
            ],
            8: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-2",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-23"
                ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-12",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-24"
                ]
            ],
            9: [
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-39",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32",               ],
                [
                    "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-39",
                    "GBPJPY_CATEGORY_BIN_DW_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY-32",                ]
            ],

            }
    ]
    # CATEGORY_BIN_BOTH用 スプレッドに対応する予想結果がない場合に対応させるtarget_spreadsのインデックス
    idx_other = 9 + 2  # リストの最初にtarget_spread_list, target_score_listが入っているのでプラス2する

    #CATEGORY_BIN_FOUR用
    #それぞれのスプレッド対応モデルごとのborder([formerのUP,DW], [latterのUP,DW]) -1スプレッドは対応するモデルがない場合用
    border_list = [ {-1:[ [0.50,0.50], [0.50,0.50] ],
                          0:[ [0.51, 0.51], [0.51, 0.51] ],
                          1:[ [0.5,0.49], [0.48,0.49] ],
                          2:[ [0.52,0.5], [0.52,0.49] ],
                          3:[ [0.5,0.52], [0.48,0.51] ],
                          4:[ [0.52,0.49], [0.49,0.52] ],
                          5:[ [0.5,0.5], [0.5,0.5] ],
                          6:[ [0.5,0.49], [0.48,0.5] ],
                          7:[ [0.49,0.5], [0.47,0.5] ],
                          8:[ [0.45,0.48], [0.45,0.46] ],
                          9:[ [0.52, 0.5], [0.52, 0.5] ],
                          }
                        ]

    border_list_show = border_list

    #FILE_PREFIX = "GBPJPY_CATEGORY_BIN_UP_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY"

    print("FILE_PREFIX:", FILE_PREFIX)


    total_acc_txt = []
    total_money_txt = []
    max_val_suffix = {"val":0,}

    for suffix in model_suffix:
        #print(suffix)

        #target_spreadのリスト
        target_spreads = []
        dic = sorted(suffix.items())

        #zip関数に渡すスプレッドと予想結果のリスト
        arg_lists = [target_spread_list]
        arg_lists.append(target_score_list)

        print("target_spread_list length:", len(target_spread_list))
        for k, v in dic:
            target_spreads.append(k)

            #CATEGORY_BIN_UP とDWで予想した結果を合わせる
            load_dir_up_former = "/app/model/bin_op/" + v[0][0]
            model_up_former = tf.keras.models.load_model(load_dir_up_former)

            load_dir_dw_former = "/app/model/bin_op/" + v[0][1]
            model_dw_former = tf.keras.models.load_model(load_dir_dw_former)

            load_dir_up_latter = "/app/model/bin_op/" + v[1][0]
            model_up_latter = tf.keras.models.load_model(load_dir_up_latter)

            load_dir_dw_latter = "/app/model/bin_op/" + v[1][1]
            model_dw_latter = tf.keras.models.load_model(load_dir_dw_latter)


            # ndarrayで返って来る
            predict_list_up_former = model_up_former.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=PROCESS_COUNT * 1,
                                                   use_multiprocessing=False,
                                                   verbose=0)
            predict_list_up_latter = model_up_latter.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=PROCESS_COUNT * 1,
                                                   use_multiprocessing=False,
                                                   verbose=0)
            predict_list_dw_former = model_dw_former.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=PROCESS_COUNT * 1,
                                                   use_multiprocessing=False,
                                                   verbose=0)
            predict_list_dw_latter = model_dw_latter.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=PROCESS_COUNT * 1,
                                                   use_multiprocessing=False,
                                                   verbose=0)



            #SAMEの予想結果は0とする
            predict_list_zero = np.zeros((len(predict_list_up_former), 2))

            #UP,SAME,DWの予想結果を合算する
            all_former = np.concatenate([predict_list_up_former, predict_list_zero, predict_list_dw_former], 1)
            predict_list_former_tmp = all_former[:, [0, 2, 4]]

            all_latter = np.concatenate([predict_list_up_latter, predict_list_zero, predict_list_dw_latter], 1)
            predict_list_latter_tmp = all_latter[:, [0, 2, 4]]

            #formerの予想(0,4, 0, 0,6)とlatterの予想を一緒にする(0,55, 0, 0,45) →[0,4, 0, 0,6, 0,55, 0, 0,45]
            predict_list_all_tmp = np.concatenate([predict_list_former_tmp, predict_list_latter_tmp], 1)

            arg_lists.append(predict_list_all_tmp)


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
            tmp_score = j[1]
            idx = idx_other

            if tmp_spr in target_spreads:
                #target_spreadsの何番目に予想結果が入っているか取得
                idx = target_spreads.index(tmp_spr) + 2 #リストの最初にtarget_spread_list,target_score_listが入っているのでプラス2する
            else:
                # スプレッドに対応する予想結果がない場合、スプレッドの一番大きいモデルの予想を取得する
                idx = idx_other

            target_sec = datetime.fromtimestamp(tmp_score).second
            # 予想するときの秒数によってFORMER,LATTERどちらから予想を取得するか決める
            if target_sec in FORMER_LIST:
                #predict_list.append(list(itemgetter(0, 1, 2)(j[idx])))
                predict_list.append(itemgetter(0, 1, 2)(j[idx]))
            else:
                #predict_list.append(list(itemgetter(3, 4, 5)(j[idx])))
                predict_list.append(itemgetter(3, 4, 5)(j[idx]))

        predict_list = np.array(predict_list) #listからnumpy.arrayにもどす
        #print(predict_list[:20])

        print("suffix:", suffix)

        under_dict = {}
        over_dict = {}
        line_val = 0.505
        #line_val = 0.582 #pyoutが950なので
        #line_val = 0.583

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

            for l, m in enumerate(zip(target_spread_list, target_score_list, predict_list)):
                spr_t = m[0]
                sec_t = m[1]
                pred_t = m[2]

                target_sec = datetime.fromtimestamp(sec_t).second
                # 予想するときの秒数によってFORMER,LATTERどちらから予想を取得するか決める
                if target_sec in FORMER_LIST:
                    if spr_t in border:
                        up_border = border[spr_t][0][0]
                        dw_border = border[spr_t][0][1]
                    else:
                        up_border = border[-1][0][0]
                        dw_border = border[-1][0][1]
                else:
                    if spr_t in border:
                        up_border = border[spr_t][1][0]
                        dw_border = border[spr_t][1][1]
                    else:
                        up_border = border[-1][1][0]
                        dw_border = border[-1][1][1]

                if pred_t[0] > pred_t[1] and pred_t[0] >= pred_t[2] and  pred_t[0] >= up_border:
                    bin_both_ind_up.append(l)
                    bin_both_ind.append(l)
                elif pred_t[2] > pred_t[0] and pred_t[2] >= pred_t[1] and  pred_t[2] >= dw_border:
                    bin_both_ind_dw.append(l)
                    bin_both_ind.append(l)


            Acc, total_num, correct_num = getAcc(predict_list, border, correct_list, bin_both_ind_up, bin_both_ind_dw)
            profit = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)

            #全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy border_ind " + str(border_ind) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
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

            ind = bin_both_ind

            x5 = predict_list[ind,:]
            y5 = correct_list[ind,:]
            s5 = target_score_list[ind]
            c5 = change_list[ind]
            sp5 = target_spread_list[ind]
            sc5 = pred_close_list[ind]
            ec5 = real_close_list[ind]
            dp5 = target_divide_prev_list[ind]
            da5 = target_divide_aft_list[ind]

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

            spread_trade_up_former = {}
            spread_win_up_former = {}
            spread_trade_up_latter = {}
            spread_win_up_latter = {}

            spread_trade_dw_former = {}
            spread_win_dw_former = {}
            spread_trade_dw_latter = {}
            spread_win_dw_latter = {}

            spread_trade_real = {}
            spread_win_real = {}

            spread_trade_real_up_former = {}
            spread_win_real_up_former = {}
            spread_trade_real_dw_former = {}
            spread_win_real_dw_former = {}

            spread_trade_real_up_latter = {}
            spread_win_real_up_latter = {}
            spread_trade_real_dw_latter = {}
            spread_win_real_dw_latter = {}

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

                former_flg = True #予想がformer,latterどちらか
                target_sec = datetime.fromtimestamp(s).second
                if not target_sec in FORMER_LIST:
                    former_flg = False


                if (max == 0 or max == 2):
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

                            if former_flg:
                                if max == 0:
                                    spread_trade_up_former[k] = spread_trade_up_former.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_up_former[k] = spread_win_up_former.get(k, 0) + 1
                                elif max == 2:
                                    spread_trade_dw_former[k] = spread_trade_dw_former.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_dw_former[k] = spread_win_dw_former.get(k, 0) + 1
                            else:
                                if max == 0:
                                    spread_trade_up_latter[k] = spread_trade_up_latter.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_up_latter[k] = spread_win_up_latter.get(k, 0) + 1
                                elif max == 2:
                                    spread_trade_dw_latter[k] = spread_trade_dw_latter.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_dw_latter[k] = spread_win_dw_latter.get(k, 0) + 1


                            flg = True
                            break

                    if flg == False:
                        if sp < 0:
                            spread_trade["spread0"] = spread_trade.get("spread0", 0) + 1
                            if win_flg:
                                spread_win["spread0"] = spread_win.get("spread0", 0) + 1

                            if former_flg:
                                if max == 0:
                                    spread_trade_up_former["spread0"] = spread_trade_up_former.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_up_former["spread0"] = spread_win_up_former.get("spread0", 0) + 1
                                elif max == 2:
                                    spread_trade_dw_former["spread0"] = spread_trade_dw_former.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_dw_former["spread0"] = spread_win_dw_former.get("spread0", 0) + 1
                            else:
                                if max == 0:
                                    spread_trade_up_latter["spread0"] = spread_trade_up_latter.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_up_latter["spread0"] = spread_win_up_latter.get("spread0", 0) + 1
                                elif max == 2:
                                    spread_trade_dw_latter["spread0"] = spread_trade_dw_latter.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_dw_latter["spread0"] = spread_win_dw_latter.get("spread0", 0) + 1

                        else:
                            spread_trade["spread16Over"] = spread_trade.get("spread16Over", 0) + 1
                            if win_flg:
                                spread_win["spread16Over"] = spread_win.get("spread16Over", 0) + 1

                            if former_flg:
                                if max == 0:
                                    spread_trade_up_former["spread16Over"] = spread_trade_up_former.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_up_former["spread16Over"] = spread_win_up_former.get("spread16Over", 0) + 1
                                elif max == 2:
                                    spread_trade_dw_former["spread16Over"] = spread_trade_dw_former.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_dw_former["spread16Over"] = spread_win_dw_former.get("spread16Over", 0) + 1
                            else:
                                if max == 0:
                                    spread_trade_up_latter["spread16Over"] = spread_trade_up_latter.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_up_latter["spread16Over"] = spread_win_up_latter.get("spread16Over", 0) + 1
                                elif max == 2:
                                    spread_trade_dw_latter["spread16Over"] = spread_trade_dw_latter.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_dw_latter["spread16Over"] = spread_win_dw_latter.get("spread16Over", 0) + 1

                    # 実際のスプレッド毎の勝率
                    if len(tradeReult) != 0:
                        flg = False
                        for k, v in SPREAD_LIST.items():
                            if sp > v[0] and sp <= v[1]:
                                spread_trade_real[k] = spread_trade_real.get(k, 0) + 1
                                if win_trade_flg:
                                    spread_win_real[k] = spread_win_real.get(k, 0) + 1

                                if former_flg:
                                    if max == 0:
                                        spread_trade_real_up_former[k] = spread_trade_real_up_former.get(k, 0) + 1
                                        if win_flg:
                                            spread_win_real_up_former[k] = spread_win_real_up_former.get(k, 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_former[k] = spread_trade_real_dw_former.get(k, 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_former[k] = spread_win_real_dw_former.get(k, 0) + 1
                                else:
                                    if max == 0:
                                        spread_trade_real_up_latter[k] = spread_trade_real_up_latter.get(k, 0) + 1
                                        if win_flg:
                                            spread_win_real_up_latter[k] = spread_win_real_up_latter.get(k, 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_latter[k] = spread_trade_real_dw_latter.get(k, 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_latter[k] = spread_win_real_dw_latter.get(k, 0) + 1
                                flg = True
                                break
                        if flg == False:
                            if sp < 0:
                                spread_trade_real["spread0"] = spread_trade_real.get("spread0", 0) + 1
                                if win_trade_flg:
                                    spread_win_real["spread0"] = spread_win_real.get("spread0", 0) + 1

                                if former_flg:
                                    if max == 0:
                                        spread_trade_real_up_former["spread0"] = spread_trade_real_up_former.get("spread0", 0) + 1
                                        if win_flg:
                                            spread_win_real_up_former["spread0"] = spread_win_real_up_former.get("spread0", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_former["spread0"] = spread_trade_real_dw_former.get("spread0", 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_former["spread0"] = spread_win_real_dw_former.get("spread0", 0) + 1
                                else:
                                    if max == 0:
                                        spread_trade_real_up_latter["spread0"] = spread_trade_real_up_latter.get("spread0", 0) + 1
                                        if win_flg:
                                            spread_win_real_up_latter["spread0"] = spread_win_real_up_latter.get("spread0", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_latter["spread0"] = spread_trade_real_dw_latter.get("spread0", 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_latter["spread0"] = spread_win_real_dw_latter.get("spread0", 0) + 1
                            else:
                                spread_trade_real["spread16Over"] = spread_trade_real.get("spread16Over", 0) + 1
                                if win_trade_flg:
                                    spread_win_real["spread16Over"] = spread_win_real.get("spread16Over", 0) + 1

                                if former_flg:
                                    if max == 0:
                                        spread_trade_real_up_former["spread16Over"] = spread_trade_real_up_former.get("spread16Over", 0) + 1
                                        if win_flg:
                                            spread_win_real_up_former["spread16Over"] = spread_win_real_up_former.get("spread16Over", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_former["spread16Over"] = spread_trade_real_dw_former.get("spread16Over", 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_former["spread16Over"] = spread_win_real_dw_former.get("spread16Over", 0) + 1
                                else:
                                    if max == 0:
                                        spread_trade_real_up_latter["spread16Over"] = spread_trade_real_up_latter.get("spread16Over", 0) + 1
                                        if win_flg:
                                            spread_win_real_up_latter["spread16Over"] = spread_win_real_up_latter.get("spread16Over", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw_latter["spread16Over"] = spread_trade_real_dw_latter.get("spread16Over", 0) + 1
                                        if win_flg:
                                            spread_win_real_dw_latter["spread16Over"] = spread_win_real_dw_latter.get("spread16Over", 0) + 1

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

                    """
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
                    """

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

                if (max == 0) :
                    # Up predict

                    if max == y.argmax():
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

                elif (max == 2):
                    #Down predict

                    if max == y.argmax():
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

            plt.title(
                'border_ind:' + str(border_ind) + " payout:" + str(PAYOUT) + " spread:" + str(SPREAD) + " money:" + str(
                    money))



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

            print("理論上のスプレッド毎の勝率(UP FORMER)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_up_former.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_up_former.get(k, 0),
                          " profit:", spread_win_up_former.get(k, 0) * PAYOUT - (spread_trade_up_former.get(k) - spread_win_up_former.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_up_former.get(k, 0) / spread_trade_up_former.get(k))
                else:
                    print(k, " cnt:", spread_trade_up_former.get(k, 0))

            print("理論上のスプレッド毎の勝率(UP LATTER)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_up_latter.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_up_latter.get(k, 0),
                          " profit:", spread_win_up_latter.get(k, 0) * PAYOUT - (spread_trade_up_latter.get(k) - spread_win_up_latter.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_up_latter.get(k, 0) / spread_trade_up_latter.get(k))
                else:
                    print(k, " cnt:", spread_trade_up_latter.get(k, 0))

            print("理論上のスプレッド毎の勝率(DW FORMER)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_dw_former.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_dw_former.get(k, 0),
                          " profit:", spread_win_dw_former.get(k, 0) * PAYOUT - (spread_trade_dw_former.get(k) - spread_win_dw_former.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_dw_former.get(k, 0) / spread_trade_dw_former.get(k))
                else:
                    print(k, " cnt:", spread_trade_dw_former.get(k, 0))

            print("理論上のスプレッド毎の勝率(DW LATTER)")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade_dw_latter.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade_dw_latter.get(k, 0),
                          " profit:", spread_win_dw_latter.get(k, 0) * PAYOUT - (spread_trade_dw_latter.get(k) - spread_win_dw_latter.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win_dw_latter.get(k, 0) / spread_trade_dw_latter.get(k))
                else:
                    print(k, " cnt:", spread_trade_dw_latter.get(k, 0))

            if TRADE_FLG:
                print("実トレード上のスプレッド毎の勝率(全体)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real.get(k, 0),
                              " profit:", spread_win_real.get(k, 0) * PAYOUT - (spread_trade_real.get(k) - spread_win_real.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real.get(k, 0) / spread_trade_real.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(UP FORMER)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_up_former.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_up_former.get(k, 0),
                              " profit:", spread_win_real_up_former.get(k, 0) * PAYOUT - (spread_trade_real_up_former.get(k) - spread_win_real_up_former.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_up_former.get(k, 0) / spread_trade_real_up_former.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_up_former.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(UP LATTER)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_up_latter.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_up_latter.get(k, 0),
                              " profit:", spread_win_real_up_latter.get(k, 0) * PAYOUT - (spread_trade_real_up_latter.get(k) - spread_win_real_up_latter.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_up_latter.get(k, 0) / spread_trade_real_up_latter.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_up_latter.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(DW FORMER)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_dw_former.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_dw_former.get(k, 0),
                              " profit:", spread_win_real_dw_former.get(k, 0) * PAYOUT - (spread_trade_real_dw_former.get(k) - spread_win_real_dw_former.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_dw_former.get(k, 0) / spread_trade_real_dw_former.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_dw_former.get(k, 0))

                print("実トレード上のスプレッド毎の勝率(DW LATTER)")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real_dw_latter.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real_dw_latter.get(k, 0),
                              " profit:", spread_win_real_dw_latter.get(k, 0) * PAYOUT - (spread_trade_real_dw_latter.get(k) - spread_win_real_dw_latter.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real_dw_latter.get(k, 0) / spread_trade_real_dw_latter.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real_dw_latter.get(k, 0))

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

            """
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
            """


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

            plt.show()

if __name__ == "__main__":
    start_time = time.perf_counter()
    #print("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)
    if LEARNING_TYPE != "CATEGORY_BIN_FOUR":
        print("ERROR !!! This file is For CATEGORY_BIN_FOUR!!!")
        exit(1)

    do_predict()
    print("Processing Time(Sec)", time.perf_counter() - start_time )
