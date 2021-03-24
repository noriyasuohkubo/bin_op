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
import pandas as pd
from lstm_generator2 import create_model, loss


"""
回帰regression用の検証用

"""

start = datetime(2018, 1, 1)
#end = datetime(2018, 1, 11)
end = datetime(2019, 12, 31)

# 2秒ごとの成績を計算する場合
per_sec_flg = False
# 2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
# {key:[trade_num,win_num]}
per_sec_dict = {}
per_sec = 2
if per_sec_flg:
    for i in range(60):
        if i % per_sec == 0:
            per_sec_dict[i] = [0, 0]

def getAcc(res, pred_close_list, real_close_list, target_spread_list, border):

    mid_tmp = res[:,0]

    #print(mid_tmp)
    # 現実のレートに換算する
    mid = pred_close_list * ((mid_tmp / 10000) + 1)
    #上昇予想 スプレッドを加味したレートより予想が上の場合 ベット対象
    if REAL_SPREAD_FLG:
        up_ind = np.where(mid >= pred_close_list + float( Decimal("0.001") * Decimal(str(target_spread_list + 1))) + border)
        up_win = np.where(real_close_list[up_ind] >= pred_close_list[up_ind] + float( Decimal("0.001") * Decimal(str(target_spread_list + 1))))

        # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
        down_ind = np.where(mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(target_spread_list + 1))) - border)
        down_win = np.where(real_close_list[down_ind] <= pred_close_list[down_ind] - float(Decimal("0.001") * Decimal(str(target_spread_list + 1))))
    else:
        up_ind = np.where(
            mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(SPREAD))) + border)
        up_win = np.where(real_close_list[up_ind] >= pred_close_list[up_ind] + float(
            Decimal("0.001") * Decimal(str(SPREAD))))

        # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
        down_ind = np.where(
            mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(SPREAD))) - border)
        down_win = np.where(real_close_list[down_ind] <= pred_close_list[down_ind] - float(
            Decimal("0.001") * Decimal(str(SPREAD))))

    total_num = len(down_ind[0]) + len(up_ind[0])
    correct_num = len(down_win[0]) + len(up_win[0])

    if total_num > 0:
        Acc = correct_num / total_num
    else:
        Acc = 0

    return Acc, total_num, correct_num

def getAccFx(res, pred_close_list, dataY, change, border):

    mid_tmp = res[:,0]

    #print(mid_tmp)
    # 現実のレートに換算する
    mid = pred_close_list * ((mid_tmp / 10000) + 1)

    #上昇予想 レートより予想が上の場合 ベット対象
    up_ind = np.where(mid >= pred_close_list + float( Decimal("0.001") * Decimal(str(SPREAD))) + border)
    # 下降予想 レートより予想が下の場合 ベット対象
    down_ind = np.where(mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(SPREAD))) - border)

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

    # 直近の変化率リスト
    target_divide_prev_list = np.array(dataSequence2.get_target_divide_prev_list())

    # 正解までのの変化率リスト
    target_divide_aft_list = np.array(dataSequence2.get_target_divide_aft_list())

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

    # model_suffix = ["90*17",]

    border_list = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, ]
    # border_list = []  # 正解率と獲得金額のみ表示
    border_list_show = []  # グラフも表示
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
        print("pred_close", len(pred_close_list), pred_close_list[:10])
        """

        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

        r = redis.Redis(host='localhost', port=6379, db=DB_TRADE_NO)

        for b in border_list:

            # 予想結果表示用テキストを保持
            result_txt = []

            border = b
            if FX == False:
                Acc, total_num, correct_num = getAcc(predict_list, pred_close_list, real_close_list, target_spread_list, border)

            else:
                Acc, total_num, correct_num, profit = getAccFx(predict_list, pred_close_list, correct_list, change_list, border)

            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

            if FX == False:
                win_money = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)
                result_txt.append("Earned Money:" + str(win_money))

            else:
                result_txt.append("Earned Money:" + str(profit))

            if border not in border_list_show:
                for i in result_txt:
                    myLogger(i)
                continue

                money_y = []
                money_tmp = {}

                money = START_MONEY  # 始めの所持金

                cnt_up_cor = 0  # upと予想して正解した数
                cnt_up_wrong = 0  # upと予想して不正解だった数

                cnt_down_cor = 0  # downと予想して正解した数
                cnt_down_wrong = 0  # downと予想して不正解だった数

                for pred, bef, aft, s, c, real, sp in zip(predict_list, pred_close_list, real_close_list, target_score_list,
                                                      change_list, correct_list, target_spread_list):

                    mid = bef * ((pred / 10000) + 1)  # 予測レート平均

                    # 予想した時間
                    predict_t = datetime.fromtimestamp(s)
                    win_flg = False

                    spread = SPREAD

                    if REAL_SPREAD_FLG:
                        spread = spread_tmp[i - 1] + 1

                    # 上昇予想 スプレッドを加味したレートより下限の予想が上の場合 ベット対象
                    if bef + float(Decimal("0.001") * Decimal(str(spread))) + border <= mid:

                        if FX:
                            profit = (c - (0.001 * spread)) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                        if bef + float(Decimal("0.001") * Decimal(str(spread))) <= aft:
                            # 正解の場合
                            if FX == False:
                                money = money + PAYOUT
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                            cnt_up_cor = cnt_up_cor + 1
                            win_flg = True

                        else:
                            if FX == False:
                                money = money - PAYOFF
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,
                                                                       PAYOFF * -1)

                            cnt_up_wrong = cnt_up_wrong + 1

                    elif bef - float(Decimal("0.001") * Decimal(str(spread))) - border >= mid:

                        if FX:
                            # cはaft-befなのでdown予想の場合の利益として-1を掛ける
                            profit = ((c * -1) - (0.001 * spread)) * FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                        if bef - float(Decimal("0.001") * Decimal(str(spread))) >= aft:
                            # 正解の場合
                            if FX == False:
                                money = money + PAYOUT
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                            cnt_down_cor = cnt_down_cor + 1
                            win_flg = True

                        else:
                            if FX == False:
                                money = money - PAYOFF
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,
                                                                       PAYOFF * -1)

                            cnt_down_wrong = cnt_down_wrong + 1

                    money_tmp[s] = money

                    # 秒ごとのbetした回数と勝ち数を保持
                    if per_sec_flg:
                        if max == 0 or max == 2:
                            per_sec_dict[predict_t.second][0] += 1
                            if win_flg:
                                per_sec_dict[predict_t.second][1] += 1

                prev_money = 1000000

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

                if FX == True:
                    plt.title('border:' + str(border) + " position:" + str(FX_POSITION) + " spread:" + str(
                        SPREAD) + " money:" + str(money))
                else:
                    plt.title('border:' + str(border) + " payout:" + str(FX_POSITION) + " spread:" + str(
                        SPREAD) + " money:" + str(money))
                plt.show()

                if per_sec_flg:
                    # 理論上の秒ごとの勝率
                    for i in per_sec_dict.keys():
                        if per_sec_dict[i][0] != 0:
                            win_rate = per_sec_dict[i][1] / per_sec_dict[i][0]
                            print("理論上の秒毎の確率:" + str(i) + " トレード数:" + str(per_sec_dict[i][0]) + " 勝率:" + str(win_rate))

                max_drawdowns.sort()
                myLogger(max_drawdowns[0:10])

                drawdown_cnt = {}
                for i in max_drawdowns:
                    for k, v in DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                            break
                for k, v in sorted(DRAWDOWN_LIST.items()):
                    myLogger(k, drawdown_cnt.get(k, 0))

                """
                max_drawdowns_np = np.array(max_drawdowns)
                df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
                myLogger(df)
                """

            for i in result_txt:
                myLogger(i)


if __name__ == "__main__":
    do_predict()

    print("END!!!")