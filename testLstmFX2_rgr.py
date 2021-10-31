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
from decimal import Decimal
from DataSequence2 import DataSequence2
from readConf2 import *
import pandas as pd
import sys

start = datetime(2020, 1, 1, 23)

start = datetime(2020, 5, 1, 23)

end = datetime(2021, 3, 31, 22)

# 2秒ごとの成績を計算する場合
per_sec_flg = True

def getAccFx(res, pred_close_list, real_close_list, change, border):
    mid_tmp = res[:, 0]

    # print(mid_tmp)
    # 現実のレートに換算する
    mid = pred_close_list * ((mid_tmp / 10000) + 1)

    # 上昇予想 レートより予想が上の場合 ベット対象
    if EXCEPT_DIVIDE_MAX != 0:
        mid_max = pred_close_list * ((EXCEPT_DIVIDE_MAX / 10000) + 1)
        up_ind = np.where(
            (mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(SPREAD))) + border) & (mid <= mid_max))
    else:
        up_ind = np.where(mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(SPREAD))) + border)

    up_win = np.where(
        real_close_list[up_ind] >= pred_close_list[up_ind] + float(Decimal("0.001") * Decimal(str(SPREAD))))

    # 下降予想 レートより予想が下の場合 ベット対象
    if EXCEPT_DIVIDE_MAX != 0:
        mid_min = pred_close_list * ((EXCEPT_DIVIDE_MAX * -1 / 10000) + 1)
        down_ind = np.where(
            (mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(SPREAD))) - border) & (mid >= mid_min))
    else:
        down_ind = np.where(mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(SPREAD))) - border)
    down_win = np.where(
        real_close_list[down_ind] <= pred_close_list[down_ind] - float(Decimal("0.001") * Decimal(str(SPREAD))))

    c5_up = change[up_ind]
    # 儲けを合算
    c5_up_sum = np.sum(c5_up)

    c5_down = change[down_ind]
    # 儲けを合算(売りなので-1を掛ける)
    c5_down_sum = np.sum(c5_down) * -1

    # 儲けからスプレッドをひく
    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float(Decimal("0.001") * Decimal(SPREAD)))
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

    train_list_index = dataSequence2.get_train_list_index()

    model_suffix = [
        "90*1",
        "90*2",
        "90*3",
        "90*4",
        "90*5",
        "90*6",
        "90*7",
        "90*8",
        "90*9",
        "90*10",
        "90*11",
        "90*12",
        "90*13",
        "90*14",
        "90*15",
        "90*16",
        "90*17",
        "90*18",
        "90*19",
        "90*20",
        "90*21",
        "90*22",
        "90*23",
        "90*24",
        "90*25",
        "90*26",
        "90*27",
        "90*28",
        "90*29",
        "90*30",
        "90*31",
        "90*32",
        "90*33",
        "90*34",
        "90*35",
        "90*36",
        "90*37",
        "90*38",
        "90*39",
        "90*40",
    ]

    model_suffix = ["90*41", ]

    border_list = [-0.006, -0.004, -0.002, 0, 0.002, 0.004, 0.006, ]  # 正解率と獲得金額のみ表示

    border_list_show = [-0.006, -0.004, -0.002, 0, 0.002, 0.004, 0.006, ]  # グラフも表示

    show_detail = True

    for suffix in model_suffix:
        print(suffix)

        load_dir = "/app/model/fx/" + FILE_PREFIX + "-" + suffix

        """
        load_dir = "/app/model/fx/" \
                   + "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT10-10-8-4-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX10_SPREAD2_UB1_LOSS-HUBER-90*23"
        """

        """
        load_dir = "/app/model/fx/" \
                    + "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT10-10-8-4-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_DIVIDEMIN0.2_SPREAD2_UB1_LOSS-HUBER-90*41"
        """

        model = tf.keras.models.load_model(load_dir)

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

        # print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

        for border in border_list:

            # 予想結果表示用テキストを保持
            result_txt = []

            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            """
            Acc, total_num, correct_num, profit = getAccFx(predict_list, pred_close_list, real_close_list, change_list, border)

            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
            result_txt.append("Earned Money:" + str(profit))

            """

            money_y = []
            money_tmp = {}

            money = START_MONEY  # 始めの所持金

            bet_price = {}  # 注文した際のレートを保持
            bet_type = {}  # BUY or SELL

            now_price = 0

            j = 0
            target_idx = -1
            bet_cnt = 0
            bet_len = {}
            # bet期間ごとの数
            bet_len_dict = {}

            for i in range(TERM):
                if i % BET_TERM == 0:
                    bet_price[i] = 0
                    bet_type[i] = ""
                    bet_len[i] = 0

            stop_loss_cnt = 0

            for cnt, (sc, close, ind) in enumerate(zip(score_list, close_list, train_list_index)):

                shift = sc % TERM

                j += 1

                # 今のレート
                now_price = close

                # 今のスコアで予想をした時のインデックスを取得
                # target_idx = np.where(target_score_list == sc)[0]

                # 予想がない場合
                if ind == -1:
                    if bet_price[shift] != 0:
                        # ポジションがあり、予想がない場合はその日の最後の予想だったので決済する
                        # print("predict not exist! score:", sc)
                        c = now_price - bet_price[shift]
                        if bet_type[shift] == "BUY":
                            profit = (c - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                        elif bet_type[shift] == "SELL":
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                        else:
                            # 想定外エラー
                            print("ERROR1")
                            sys.exit()

                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                        money_tmp[sc] = money
                        bet_price[shift] = 0
                        bet_type[shift] = ""
                        bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(
                            bet_len[shift],
                            0) != 0 else 1
                        bet_len[shift] = 0
                    continue

                # 予想取得
                pred = predict_list[ind, :]

                div = 0
                if pred[0] < 0:
                    # マイナスの場合
                    div = str(pred[0])[0:4]  # マイナス符号が入るため1文字増やす
                else:
                    div = str(pred[0])[0:3]

                mid = float(Decimal(str(now_price)) * (Decimal(str(pred[0])) / Decimal("10000") + Decimal("1")))

                # 今保持している建玉すべてでストップロス確認
                for sh in bet_price:

                    if bet_price[sh] != 0:
                        # 既にポジションがある場合
                        c = now_price - bet_price[sh]
                        if bet_type[sh] == "BUY":
                            if c <= FX_STOP_LOSS:
                                stop_loss_cnt += 1
                                # print("stop loss!", sc)
                                # ストップロス以上の損失を出している場合は決済する
                                profit = (FX_STOP_LOSS - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[sh] = 0
                                bet_type[sh] = ""
                                bet_len_dict[bet_len[sh]] = bet_len_dict[bet_len[sh]] + 1 if bet_len_dict.get(
                                    bet_len[sh],
                                    0) != 0 else 1
                                bet_len[sh] = 0

                        elif bet_type[sh] == "SELL":
                            if (c * -1) <= FX_STOP_LOSS:
                                stop_loss_cnt += 1
                                # print("stop loss!", sc)
                                # ストップロス以上の損失を出している場合は決済する
                                profit = (FX_STOP_LOSS - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                bet_price[sh] = 0
                                bet_type[sh] = ""
                                bet_len_dict[bet_len[sh]] = bet_len_dict[bet_len[sh]] + 1 if bet_len_dict.get(
                                    bet_len[sh],
                                    0) != 0 else 1
                                bet_len[sh] = 0
                        else:
                            # 想定外エラー
                            print("ERROR2")
                            sys.exit()

                if bet_price[shift] != 0:
                    # 既にポジションがある場合
                    c = now_price - bet_price[shift]

                    if bet_type[shift] == "BUY":
                        # 買いポジションがある場合
                        if now_price + FX_MORE_BORDER < mid:
                            # 更に上がると予想されている場合、決済しないままとする
                            bet_len[shift] += 1
                        else:
                            # 更に上がらなければ決済する
                            profit = (c - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(
                                bet_len[shift],
                                0) != 0 else 1
                            bet_len[shift] = 0

                    elif bet_type[shift] == "SELL":
                        # 売りポジションがある場合
                        if now_price - FX_MORE_BORDER > mid:
                            # 更に下がると予想されている場合、決済しないままとする
                            bet_len[shift] += 1
                        else:
                            # 更に下がらなければ決済する
                            profit = ((c * -1) - float(Decimal("0.001") * Decimal(SPREAD))) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            bet_price[shift] = 0
                            bet_type[shift] = ""
                            bet_len_dict[bet_len[shift]] = bet_len_dict[bet_len[shift]] + 1 if bet_len_dict.get(
                                bet_len[shift],
                                0) != 0 else 1
                            bet_len[shift] = 0
                    else:
                        # 想定外エラー
                        print("ERROR3")
                        sys.exit()

                if bet_price[shift] == 0:
                    # ポジションがない場合

                    # 上る予想
                    if now_price + float(Decimal("0.001") * Decimal(str(SPREAD)) + Decimal(str(border))) <= mid:
                        bet_price[shift] = now_price
                        bet_type[shift] = "BUY"
                        bet_cnt += 1
                        bet_len[shift] = 1
                    # 下る予想
                    elif now_price - float(Decimal("0.001") * Decimal(str(SPREAD)) + Decimal(str(border))) >= mid:
                        bet_price[shift] = now_price
                        bet_type[shift] = "SELL"
                        bet_cnt += 1
                        bet_len[shift] = 1

            # print("loop cnt:", j)

            print("border:", border)
            print("Earned money: " + str(money - START_MONEY))

            # for i in result_txt:
            #    myLogger(i)

            print("bet cnt:", bet_cnt)

            if show_detail:

                print("stop loss cnt:", stop_loss_cnt)

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
                        prev_money))

                print("bet期間 件数")
                for k, v in sorted(bet_len_dict.items()):
                    print(k, bet_len_dict.get(k, 0))

            print("MAX DrawDowns(理論上のドローダウン)")
            max_drawdowns.sort()
            myLogger(max_drawdowns[0:10])

            if show_detail:
                drawdown_cnt = {}
                for i in max_drawdowns:
                    for k, v in DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                            break
                for k, v in sorted(DRAWDOWN_LIST.items()):
                    print(k, drawdown_cnt.get(k, 0))

                # plt.show()

                file_path = "/home/reicou/" + str(suffix) + "-" + str(border) + ".png"
                plt.savefig(file_path)


if __name__ == "__main__":
    do_predict()

    print("END!!!")