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

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

start = datetime(2021, 1, 1,)
end = datetime(2022, 1, 1,)

# start = datetime(2020, 1, 1, 23)
# end = datetime(2020, 12, 31, 22)

# 2秒ごとの成績を計算する場合
per_sec_flg = True

def getAccPerBorder(res, dataY, border):
    shape = res[0].shape[0]  # CATEGORY_BIN_BOTHなら3列、CATEGORY_BIN_UPやCATEGORY_BIN_DWなら2列
    if shape == 2:
        up_down_ind = np.where(res[:, 0] > border)[0]
        x5_up_down = res[up_down_ind, :]
        y5_up_down = dataY[up_down_ind, :]

        up_down_eq = np.equal(x5_up_down.argmax(axis=1), y5_up_down.argmax(axis=1))
        up_down_cor_length = int(len(np.where(up_down_eq == True)[0]))

        total_num = len(up_down_eq)
        correct_num = up_down_cor_length

    elif shape == 3:
        up_ind = np.where((res[:, 0] > res[:, 2]) & (res[:, 0] > border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > border))[0]

        x5_up = res[up_ind, :]
        y5_up = dataY[up_ind, :]

        x5_down = res[down_ind, :]
        y5_down = dataY[down_ind, :]

        up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
        up_cor_length = int(len(np.where(up_eq == True)[0]))
        down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
        down_cor_length = int(len(np.where(down_eq == True)[0]))

        total_num = len(up_ind) + len(down_ind)
        correct_num = up_cor_length + down_cor_length

    acc = 0

    if total_num != 0:
        acc = correct_num / total_num

    return total_num, correct_num, acc

def get_ask_bid(close, spr):
    # sprはask,bidの差として入っているので、ask,bidを求めるために一旦半分にする
    tmp_spr = float(Decimal(str(spr)) / Decimal("2"))
    now_ask = close + (0.001 * tmp_spr)
    now_bid = close - (0.001 * tmp_spr)

    return [now_ask, now_bid]

def getAccFxCat(res, border, dataY, change):
    up_ind = np.where((res[:, 0] > res[:, 1]) & (res[:, 0] >= res[:, 2]) & (res[:, 0] >= border))[0]
    down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > res[:, 1]) & (res[:, 2] >= border))[0]

    x5_up = res[up_ind, :]
    y5_up = dataY[up_ind, :]
    c5_up = change[up_ind]

    # 儲けを合算
    c5_up_sum = np.sum(c5_up)

    x5_down = res[down_ind, :]
    y5_down = dataY[down_ind, :]
    c5_down = change[down_ind]

    # 儲けを合算(売りなので-1を掛ける)
    c5_down_sum = np.sum(c5_down) * -1

    # 儲けからスプレッド分をひく
    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float(Decimal("0.001") * Decimal(FX_SPREAD)))
    c5_sum = c5_sum * FX_POSITION

    up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
    down_cor_length = int(len(np.where(down_eq == True)[0]))

    total_num = len(up_ind) + len(down_ind)
    correct_num = up_cor_length + down_cor_length

    if total_num == 0:
        Acc = 0
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
    dataSequence2 = DataSequence2(start, end, True, False)

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

    # 全spread値のリスト
    spread_list = dataSequence2.get_spread_list()

    # 全tick値のリスト
    tick_list = dataSequence2.get_tick_list()

    # spread値のリスト
    target_spread_list = np.array(dataSequence2.get_target_spread_list())

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    train_list_index = dataSequence2.get_train_list_index()

    model_suffix = []
    for i in range(40):
        model_suffix.append(str(i + 1))

    #border_list = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64,  ]
    border_list = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64,  ]

    # border_list = [ 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,  ]  # spread3
    # border_list = [ 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43,    ]  # spread5
    # border_list = [ 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54,    ]  # 60 spread5
    # border_list = [0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52,  ]  # 60 spread7
    # border_list = [0.38, 0.39, 0.4, 0.41, 0.42,0.43, 0.44, 0.45, 0.46, 0.47,   ]  # 60 spread9
    # border_list = [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, ]  # 180 spread1
    # border_list = [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,  ]  # 180 spread3
    # border_list = [0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, ]  # 180 spread5
    # border_list = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51,   ]  # 180 spread7
    # border_list = [0.38, 0.39,0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,     ]  # 180 spread9
    #border_list = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64,   ]  # 300 spread1
    # border_list = [ 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54,]  # 300 spread7
    #border_list = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6  ]  # 300 spread9
    # border_list = [0.36, 0.37, 0.38, 0.39, 0.4,0.41, 0.42, 0.43, 0.44, 0.45,   ]  # 300 spread11
    # border_list = [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, ]  # 600 spread1
    #border_list = [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,]  # 600 spread7
    #border_list = [0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54,]  # 600 spread9

    # border_list = [-0.006, -0.004, -0.002, 0, 0.002, 0.004,   ]  # RGR用
    #border_list = [0.54,]
    #model_suffix = ["37","38","39"]

    # border_list_show = [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, ]
    # 基本的にFXはベット延長させる
    border_list_show = border_list

    # 詳細表示
    show_detail = False

    show_history = False #取引履歴を表示

    # ベット延長するモデルを別途設定する場合
    ext_flg = False
    EXT_LEARNING_TYPE = LEARNING_TYPE

    # ベット延長するか判断するTERM
    ext_term = 60

    # ベット延長しはじめる期間 例えば本来のモデルが120秒予想で、延長決定するモデルが30秒予想の場は4
    # 120秒経過してはじめて延長するか判断する　120秒経過するまではロスカットされない限りベットしつづける
    # ext_begin_len = int(Decimal(str(TERM)) / Decimal(str(ext_term)))
    ext_begin_len = 1

    print("ext_term", ext_term)
    print("ext_begin_len", ext_begin_len)

    FILE_PREFIX = "USDJPY_CATEGORY_BIN_DW_LSTM_BET10_TERM60_INPUT10-60-300_INPUT_LEN360-120-24_L-UNIT36-12-4_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_201601_202012_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU_md-msd-mcd-d_d_BS8192"

    if ext_flg:
        # ベット延長するかを決定するモデル
        EXT_FILE_PREFIX = "USDJPY_CATEGORY_LSTM_BET2_TERM10_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_CO7_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU_FX-15"

        if len(border_list_show) != 0 and (LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION"):
            # 詳細を表示する場合(ベット延長するかを決定するモデルを使用する場合)
            ext_load_dir = "/app/model/bin_op/" + EXT_FILE_PREFIX
            ext_model = tf.keras.models.load_model(ext_load_dir)
            ext_predict_list = ext_model.predict_generator(dataSequence2,
                                                           steps=None,
                                                           max_queue_size=PROCESS_COUNT * 1,
                                                           use_multiprocessing=False,
                                                           verbose=0)

    print("FILE_PREFIX:", FILE_PREFIX)

    # 利益を保持
    max_result_detail = {}
    max_result_per_suffix = {}

    for suffix in model_suffix:

        load_dir = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix
        if not os.path.isdir(load_dir):
            print("model not exists:" + load_dir )
            continue
        model = tf.keras.models.load_model(load_dir)
        # model.summary()

        # START tensorflow1で作成したモデル用
        """
        model_file = "/app/bin_op/model/GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5" + "." + suffix
        model = tf.keras.models.load_model(model_file)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        """
        # END tensorflow1で作成したモデル用
        predict_list = model.predict(dataSequence2,
                                               steps=None,
                                               max_queue_size=PROCESS_COUNT * 1,
                                               use_multiprocessing=False,
                                               verbose=0)

        if ext_flg == False:
            ext_predict_list = predict_list

        """
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("train_list_index", len(train_list_index), train_list_index[:10])
        """

        # print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")
        print("")
        print("suffix:", suffix)

        for border in border_list:

            # 予想結果表示用テキストを保持
            result_txt = []

            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            pips = []

            if border not in border_list_show:
                for i in result_txt:
                    myLogger(i)
                continue

            money_y = []
            money_tmp = {}

            money = START_MONEY  # 始めの所持金

            position_num = 0
            position_num_tmp = {} #保持しているポジション数の推移

            #bet_price = {}  # 注文した際のレートを保持
            #bet_type = {}  # BUY or SELL
            #bet_len = {}  # betしている期間

            bet_dicts = {} # bet情報をshiftごとに保持 type:buy or sell, price:bet時のレート, stime:bet時の時間(score), len:betしている期間, prev_price:前回の延長判定時のレート
            bet_len_dict = {} # bet期間ごとの件数を保持
            profit_per_shift = {}  # shiftごとの利益を保持

            j = 0
            bet_cnt = 0
            win_cnt = 0 #stop_lossより稼いだら勝ちとする

            for i in range(ext_term):
                if Decimal(str(i)) % Decimal(str(BET_TERM)) == 0:
                    profit_per_shift[i] = {"cnt":0, "profit":0}

            stop_loss_cnt = 0
            deal_hist = [] #決済履歴を保持

            for cnt, (sc, close, ind, spr, tick) in enumerate(zip(score_list, close_list, train_list_index, spread_list, tick_list)):

                shift = int(Decimal(str(sc)) % Decimal(str(ext_term)))

                j += 1

                # 今のレート
                now_price = close

                tick_spr_list = tick.split(",")#tickとsprが:区切り
                ask_list = []
                bid_list = []
                for tick_spr in tick_spr_list:
                    tc, tcsp = tick_spr.split(":")
                    tc = float(tc)
                    tcsp = int(tcsp)
                    tmp_ask, tmp_bid = get_ask_bid(tc, tcsp)
                    ask_list.append(tmp_ask)
                    bid_list.append(tmp_bid)

                now_ask, now_bid = get_ask_bid(close, spr)

                # 1周り前のShiftで決済したタイプを保持
                # 延長をしないなどの理由で決済したら以下にBUY,SELLをいれる 延長しないために決済したのに同じ方向にさらにベットしないようにする
                bet_type_prev_shift = ""

                if ind == -1:  # 予想がない場合
                    # ポジションがあり、予想がない場合はその日の最後の予想、もしくはデータが途切れた場合なのでベットなしとする
                    bet_dicts = {}

                    continue

                if LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION":
                    pred = predict_list[ind, :]
                    mid = float(Decimal(str(now_price)) * (Decimal(str(pred[0])) / Decimal("10000") + Decimal("1")))
                else:
                    # 予想取得
                    pred = predict_list[ind, :]
                    max = pred.argmax()
                    percent = pred[max]

                    if LEARNING_TYPE == "CATEGORY_BIN_UP":
                        max = 0
                        percent = pred[0]
                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                        max = 2
                        percent = pred[0]


                    # ベット延長用
                    ext_pred = ext_predict_list[ind, :]

                    if EXT_LEARNING_TYPE == "CATEGORY_BIN_UP":
                        ext_max = 0
                        up_percent = ext_pred[0]
                        dw_percent = 0
                    elif EXT_LEARNING_TYPE == "CATEGORY_BIN_DW":
                        ext_max = 2
                        up_percent = 0
                        dw_percent = ext_pred[0]
                    else:
                        ext_max = ext_pred.argmax()
                        up_percent = ext_pred[0]
                        dw_percent = ext_pred[2]

                # 今保持している建玉すべてでストップロス確認
                if FX_STOP_LOSS != 0:  # 0の場合はストップロスを設定しない想定
                    del_key = []
                    for dict_key in bet_dicts.keys():
                        stop_loss_flg = False
                        bet_dict = bet_dicts[dict_key]
                        stop_price = 0
                        if bet_dict["type"] == "BUY":
                            for tick_bid in bid_list:
                                c = tick_bid - bet_dict["price"]
                                if c <= FX_STOP_LOSS:
                                    stop_loss_flg = True
                                    stop_price = tick_bid
                                    profit = FX_STOP_LOSS * FX_POSITION
                                    break

                        elif bet_dict["type"] == "SELL":
                            for tick_ask in ask_list:
                                c = tick_ask - bet_dict["price"]
                                if (c * -1) <= FX_STOP_LOSS:
                                    stop_loss_flg = True
                                    stop_price = tick_ask
                                    profit = FX_STOP_LOSS * FX_POSITION
                                    break

                        else:
                            # 想定外エラー
                            print("ERROR2")
                            sys.exit()

                        if stop_loss_flg:
                            pips.append(FX_STOP_LOSS)
                            stop_loss_cnt += 1
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            position_num = position_num -1
                            position_num_tmp[sc] = position_num

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(bet_dict["len"], 0) != 0 else 1
                            profit_per_shift[dict_key]["profit"] = profit_per_shift[dict_key]["profit"] + profit

                            hist_child = {"type":bet_dict["type"],
                                          "stime":datetime.fromtimestamp(bet_dict["stime"]).strftime('%Y/%m/%d %H:%M:%S'),
                                          "etime":datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice":bet_dict["price"],
                                          "eprice":stop_price,
                                          "profit":profit }
                            deal_hist.append(hist_child)
                            del_key.append(dict_key)

                    for dkey in del_key:
                        del bet_dicts[dkey]

                if shift in bet_dicts.keys():
                    finish_flg = False  # 決済するかどうか

                    # 既にポジションがある場合
                    # if len(FX_TARGET_SPREAD_LIST) != 0 and not(spr in FX_TARGET_SPREAD_LIST):
                    ## 指定スプレッド以外のトレードは決済する
                    #    finish_flg = True
                    # elif ext_begin_len > bet_len[shift]:
                    bet_dict = bet_dicts[shift]

                    if FX_NOT_EXT_FLG:
                        finish_flg = True
                    else:
                        if ext_begin_len > bet_dict["len"]:
                            # ベット延長するか判断できる秒数が経過していない場合
                            bet_dicts[shift]["len"] += 1
                        else:
                            # ベット延長するか判断できる秒数が経過している場合
                            if bet_dict["type"] == "BUY":
                                # 買いポジションがある場合
                                if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and now_price + FX_MORE_BORDER_RGR < mid) or \
                                        ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and
                                         FX_MORE_BORDER_CAT <= up_percent ):
                                         #not(ext_max == 2 and FX_MORE_BORDER_CAT < ext_percent) and (now_ask - bet_dicts[shift]["prev_price"]) > 0 ):
                                    # 更に上がると予想されている場合、決済しないままとする
                                    bet_dicts[shift]["prev_price"] = now_ask
                                    bet_dicts[shift]["len"] += 1
                                else:
                                    finish_flg = True

                            elif bet_dict["type"] == "SELL":
                                # 売りポジションがある場合
                                if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and now_price - FX_MORE_BORDER_RGR > mid) or \
                                        ((LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and
                                         FX_MORE_BORDER_CAT <= dw_percent):
                                         #not(ext_max == 0 and FX_MORE_BORDER_CAT < ext_percent) and (bet_dicts[shift]["prev_price"] - now_bid) > 0):
                                    # 更に上がると予想されている場合、決済しないままとする
                                    bet_dicts[shift]["prev_price"] = now_bid
                                    bet_dicts[shift]["len"] += 1
                                else:
                                    finish_flg = True

                    if finish_flg:
                        if bet_dict["type"] == "BUY":
                            # 更に上がらなければ決済する
                            profit_pips = now_bid - bet_dict["price"]
                            pips.append(profit_pips)
                            if profit_pips > FX_STOP_LOSS * -1:
                                win_cnt += 1

                            profit = (profit_pips) * FX_POSITION
                            stop_price = now_bid
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            position_num = position_num -1
                            position_num_tmp[sc] = position_num

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(bet_dict["len"], 0) != 0 else 1
                            profit_per_shift[shift]["profit"] = profit_per_shift[shift]["profit"] + profit

                            if FX_NOT_EXT_FLG == False:
                                bet_type_prev_shift = "BUY"

                            hist_child = {"type":bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime('%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice":bet_dict["price"],
                                          "eprice":stop_price,
                                          "profit":profit }

                            deal_hist.append(hist_child)
                            del bet_dicts[shift]

                        elif bet_dict["type"] == "SELL":
                            profit_pips = (now_ask - bet_dict["price"]) * -1
                            pips.append(profit_pips)
                            if profit_pips > FX_STOP_LOSS * -1:
                                win_cnt += 1
                            profit = (profit_pips) * FX_POSITION
                            stop_price = now_ask
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            position_num = position_num -1
                            position_num_tmp[sc] = position_num

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(bet_dict["len"], 0) != 0 else 1
                            profit_per_shift[shift]["profit"] = profit_per_shift[shift]["profit"] + profit

                            if FX_NOT_EXT_FLG == False:
                                bet_type_prev_shift = "SELL"

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime('%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["price"],
                                          "eprice": stop_price,
                                          "profit": profit}

                            deal_hist.append(hist_child)
                            del bet_dicts[shift]
                        else:
                            # 想定外エラー
                            print("ERROR3")
                            sys.exit()
                if (FX_SINGLE_FLG == False and (shift in bet_dicts.keys()) == False and len(bet_dicts) < FX_MAX_POSITION_CNT) or \
                        (FX_SINGLE_FLG == True and len(bet_dicts) == 0):

                    # ポジションがない場合
                    # 指定シフト以外トレードしない
                    if len(FX_TARGET_SHIFT) == 0 or (
                            len(FX_TARGET_SHIFT) != 0 and shift in FX_TARGET_SHIFT):

                        # 指定スプレッド以外のトレードは無視する
                        if len(FX_TARGET_SPREAD_LIST) == 0 or (
                                len(FX_TARGET_SPREAD_LIST) != 0 and spr in FX_TARGET_SPREAD_LIST):

                            if ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and \
                                now_price + float(
                                        Decimal("0.001") * Decimal(str(FX_SPREAD)) + Decimal(str(border))) <= mid) or \
                                    ((
                                             LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and max == 0 and percent >= border):

                                if bet_type_prev_shift != "BUY":
                                    # すでに決済済みでない場合のみ新規ポジションをもつ
                                    bet_dicts[shift] = {
                                        "type":"BUY",
                                        "price": now_ask,
                                        "stime": sc,
                                        "len":1,
                                        "prev_price": now_ask,
                                    }
                                    bet_cnt += 1
                                    profit_per_shift[shift]["cnt"] = profit_per_shift[shift]["cnt"] + 1
                                    position_num = position_num +1
                                    position_num_tmp[sc] = position_num

                            elif ((LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION") and \
                                  now_price - float(
                                        Decimal("0.001") * Decimal(str(FX_SPREAD)) + Decimal(str(border))) >= mid) or \
                                    ((
                                             LEARNING_TYPE != "REGRESSION_SIGMA" and LEARNING_TYPE != "REGRESSION") and max == 2 and percent >= border):

                                if bet_type_prev_shift != "SELL":
                                    # すでに決済済みでない場合のみ新規ポジションをもつ
                                    bet_dicts[shift] = {
                                        "type":"SELL",
                                        "price": now_bid,
                                        "stime": sc,
                                        "len":1,
                                        "prev_price": now_bid,
                                    }
                                    bet_cnt += 1
                                    profit_per_shift[shift]["cnt"] = profit_per_shift[shift]["cnt"] + 1
                                    position_num = position_num +1
                                    position_num_tmp[sc] = position_num

            prev_money = START_MONEY

            for i, score in enumerate(score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)

            prev_position_num = 0
            position_num_y = []
            for i, score in enumerate(score_list):
                if score in position_num_tmp.keys():
                    prev_position_num = position_num_tmp[score]

                position_num_y.append(prev_position_num)



            detail_profit = prev_money - START_MONEY
            print("border ", border)
            print("bet cnt: ", bet_cnt)
            print("win cnt: ", win_cnt)
            win_rate = 0
            if bet_cnt !=0:
                win_rate = win_cnt/bet_cnt
            print("win_rate:", win_rate)

            print("Detail Earned Money: ", detail_profit)

            if bet_cnt != 0:
                d_np = np.array(pips)
                d_np = np.sort(d_np)

                print("pips data length:", len(d_np))
                print("pips avg:", np.average(d_np))
                print("pips std:", np.std(d_np))
                print("pips mid:", d_np[int(len(d_np) / 2) - 1])  # 中央値
                print("pips max:", np.max(d_np))
                print("pips min:", np.min(d_np))

            for i in result_txt:
                myLogger(i)

            profit_per_drawdown = 0
            tmp_drawdown = 0
            if len(max_drawdowns) != 0:
                max_drawdowns.sort()
                profit_per_drawdown = int(detail_profit) / int(max_drawdowns[0]) * -1
                tmp_drawdown = int(max_drawdowns[0])

            print("profit_per_dd: ", profit_per_drawdown, tmp_drawdown)

            sl_bet_cnt = 0
            if bet_cnt != 0:
                sl_bet_cnt = stop_loss_cnt / bet_cnt

            max_result_detail[str(suffix) + " " + str(border)] = {"profit_per_dd": profit_per_drawdown,
                                                                  "profit": int(detail_profit), "dd": tmp_drawdown, "sl/bet":sl_bet_cnt,
                                                                  "sl_cnt": stop_loss_cnt, "bet_cnt":bet_cnt,}
            if ((suffix in max_result_per_suffix.keys()) and max_result_per_suffix[suffix]["profit"] < int(detail_profit)) or \
                (suffix in max_result_per_suffix.keys()) == False:
                    #最高利益を更新した場合
                    max_result_per_suffix[suffix] = {"border": str(border),"profit_per_dd": profit_per_drawdown,
                                                                  "profit": int(detail_profit), "dd": tmp_drawdown, "sl/bet":sl_bet_cnt,
                                                                  "sl_cnt": stop_loss_cnt, "bet_cnt":bet_cnt,}

            # print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")

            print("stop loss cnt:", stop_loss_cnt)
            print("sl/bet cnt:", sl_bet_cnt)
            """
            print("bet期間 件数")
            for k, v in sorted(bet_len_dict.items()):
                print(k, bet_len_dict.get(k, 0))
            """
            if show_history:
                print("決済履歴")
                for h in deal_hist:
                    print(h)

            if show_detail:
                print("MAX DrawDowns(理論上のドローダウン)")
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
                    print(k, "cnt:", v["cnt"], "profit:", v["profit"])

                fig = plt.figure()
                # 価格の遷移
                ax1 = fig.add_subplot(111)

                ax1.plot(close_list, 'g')

                ax2 = ax1.twinx()
                ax2.plot(money_y,'b')

                ax3 = ax1.twinx()
                ax3.plot(position_num_y,'r')
                
                plt.title(
                    'border:' + str(border) + " spread:" + str(FX_SPREAD) + " money:" + str(
                        money))
                plt.show()
    """
    print("max_suffix:", max_result["suffix"])
    print("max_border:", max_result["border"])
    print("max_profit:", max_result["profit"])
    """

    max_result_detail_sorted = sorted(max_result_detail.items(), key=lambda x: x[1]["profit"], reverse=True)
    cnt_t = 0
    print("利益の多い順")
    for i in max_result_detail_sorted:
        if cnt_t > 10:
            break
        print("suffix border:", i[0], i[1]["profit"], i[1]["profit_per_dd"], i[1]["dd"], i[1]["sl/bet"], i[1]["sl_cnt"], i[1]["bet_cnt"] )
        cnt_t += 1

    max_result_drawdown_sorted = sorted(max_result_detail.items(), key=lambda x: x[1]["profit_per_dd"], reverse=True)
    cnt_t = 0
    print("")
    print("利益/ドローダウンの多い順")
    for i in max_result_drawdown_sorted:
        if cnt_t > 20:
            break
        print("suffix border:", i[0], i[1]["profit"], i[1]["profit_per_dd"], i[1]["dd"], i[1]["sl/bet"], i[1]["sl_cnt"], i[1]["bet_cnt"] )
        cnt_t += 1

    """
    min_slbet_sorted = sorted(max_result_detail.items(), key=lambda x: x[1]["sl/bet"], reverse=False)
    cnt_t = 0
    print("")
    print("ストップロス/Bet数の少ない順")
    for i in min_slbet_sorted:
        if cnt_t > 10:
            break
        print("suffix border:", i[0], i[1]["profit"], i[1]["profit_per_dd"], i[1]["dd"], i[1]["sl/bet"], i[1]["sl_cnt"], i[1]["bet_cnt"] )
        cnt_t += 1
    """
    
    max_result_per_suffix_sorted = sorted(max_result_per_suffix.items(), key=lambda x: int(x[0]), reverse=False)
    print("suffixごとの最高利益")
    for k,v in max_result_per_suffix_sorted:
        print("suffix :", k, v["border"], v["profit"], v["profit_per_dd"], v["dd"], v["sl/bet"], v["sl_cnt"],v["bet_cnt"])

if __name__ == "__main__":
    if FX_REAL_SPREAD_FLG == False:
        # test時はFX_REAL_SPREAD_FLGはTrueであるべき
        print("FX_REAL_SPREAD_FLG is False! turn to True!!")
        exit(1)
    start_time = time.perf_counter()
    # print("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)
    do_predict()
    print("Processing Time(Sec)", time.perf_counter() - start_time)

    print("END!!!")