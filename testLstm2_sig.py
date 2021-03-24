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
import math

"""
回帰regression用の検証用

"""

start = datetime(2018, 1, 1)
#end = datetime(2018, 1, 11)
end = datetime(2019, 12, 31)

# 2秒ごとの成績を計算する場合
per_sec_flg = True
# 2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
# {key:[trade_num,win_num]}
per_sec_dict = {}
per_sec = 2
if per_sec_flg:
    for i in range(60):
        if i % per_sec == 0:
            per_sec_dict[i] = [0, 0]


# σに指定した数値を掛けた上限値および下限値が現在レートを跨がず 且つ
# 指定スプレッドより上もしくは下と予想したものから正解率と予想数と正解数を返す
def getAcc(res, pred_close_list, real_close_list, border):

    mu_pred = res[:, 0]
    sigma_pred = np.exp(res[:, 1])

    upper_tmp = mu_pred + border * sigma_pred #予測上限値
    downer_tmp = mu_pred - border * sigma_pred #予測下限値

    # 現実のレートに換算する
    upper = pred_close_list * ((upper_tmp / 10000) + 1)
    downer = pred_close_list * ((downer_tmp / 10000) + 1)

    #上昇予想 スプレッドを加味したレートより下限の予想が上の場合 ベット対象
    up_ind = np.where(downer >= pred_close_list + float( Decimal("0.001") * Decimal(str(SPREAD)) ))
    up_win = np.where(real_close_list[up_ind] >= pred_close_list[up_ind] + float( Decimal("0.001") * Decimal(str(SPREAD)) ))

    # 下降予想 スプレッドを加味したレートより上限の予想が下の場合 ベット対象
    down_ind = np.where(upper <= pred_close_list - float( Decimal("0.001") * Decimal(str(SPREAD)) ))
    down_win = np.where(real_close_list[down_ind] <= pred_close_list[down_ind] - float( Decimal("0.001") * Decimal(str(SPREAD)) ))

    total_num = len(down_ind[0]) + len(up_ind[0])
    correct_num = len(down_win[0]) + len(up_win[0])

    if total_num > 0:
        Acc = correct_num / total_num
    else:
        Acc = 0

    return Acc, total_num, correct_num

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

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    model_suffix = [
        #"90*1",
        #"90*2",
        #"90*3",
        #"90*4",
        #"90*5",
        #"90*6",
        #"90*7",
        #"90*8",
        #"90*9",
        "90*30",
    ]

    # 回帰予測の検証時にσにかける値
    border_list = [0, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,] # 正解率と獲得金額のみ表示
    # border_list_show=[1, 2, 3, ] #グラフも表示
    border_list_show = [ ]  # グラフも表示

    for suffix in model_suffix:
        print(suffix)

        load_dir = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix

        if LEARNING_TYPE == "REGRESSION_SIGMA":
            model = tf.keras.models.load_model(load_dir, custom_objects={'loss': loss})
        else:
            model = tf.keras.models.load_model(load_dir)
        # model.summary()

        # ndarrayで返って来る
        predict_list = model.predict_generator(dataSequence2,
                                               steps=None,
                                               max_queue_size=PROCESS_COUNT * 1,
                                               use_multiprocessing=False,
                                               verbose=0)
        """
        print("correct", len(correct_list), correct_list[:10])
        print("bef", len(pred_close_list), pred_close_list[:10])
        print("aft", len(real_close_list), real_close_list[:10])
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("predict", len(predict_list), predict_list[:10])
        """

        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

        # 予想結果表示用テキストを保持
        result_txt = []

        for b in border_list:
            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            border = b
            Acc, total_num, correct_num = getAcc(predict_list, pred_close_list, real_close_list, border)

            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
            result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
            result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

            if FX == False:
                win_money = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)
                result_txt.append("Earned Money:" + str(win_money))

            if border not in border_list_show:
                continue

            money_y = []
            money_tmp = {}

            money = 1000000  # 始めの所持金

            cnt_up_cor = 0  # upと予想して正解した数
            cnt_up_wrong = 0  # upと予想して不正解だった数

            cnt_down_cor = 0  # downと予想して正解した数
            cnt_down_wrong = 0  # downと予想して不正解だった数

            err_list = [] #正解との絶対差
            sigma_list = [] #分散

            for pred, bef, aft, s, c, real in zip(predict_list, pred_close_list, real_close_list, target_score_list, change_list, correct_list):

                mu_pred = pred[0]
                sigma_pred = np.exp(pred[1])

                upper_tmp = mu_pred + border * sigma_pred  # 予測上限値
                downer_tmp = mu_pred - border * sigma_pred  # 予測下限値

                # 現実のレートに換算する
                upper = bef * ((upper_tmp / 10000) + 1)
                downer = bef * ((downer_tmp / 10000) + 1)

                mid = bef * ((mu_pred / 10000) + 1) # 予測レート平均
                err = abs(aft - mid) #予測レートと実際の差
                sigma_rate_tmp = mu_pred + 1 * sigma_pred  # 予測レートの分散
                sigma_rate = bef * ((sigma_rate_tmp / 10000) + 1)
                sigma_rate_err = abs(sigma_rate - mid)

                #print(mid,bef,aft,sigma_rate)

                # 予想した時間
                predict_t = datetime.fromtimestamp(s)
                win_flg = False

                # 上昇予想 スプレッドを加味したレートより下限の予想が上の場合 ベット対象
                if bef + float(Decimal("0.001") * Decimal(str(SPREAD))) <= downer:
                    err_list.append(err)
                    sigma_list.append(sigma_rate_err)

                    if FX:
                        profit = (c - (0.001 * SPREAD)) * FX_POSITION
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                    if bef + float(Decimal("0.001") * Decimal(str(SPREAD))) <= aft:
                        # 正解の場合
                        if FX == False:
                            money = money + PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                        cnt_up_cor = cnt_up_cor + 1
                        win_flg = True

                    else:
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOFF * -1)

                        cnt_up_wrong = cnt_up_wrong + 1

                elif bef - float( Decimal("0.001") * Decimal(str(SPREAD)) ) >= upper:
                    err_list.append(err)
                    sigma_list.append(sigma_rate_err)

                    if FX:
                        # cはaft-befなのでdown予想の場合の利益として-1を掛ける
                        profit = ((c * -1) - (0.001 * SPREAD)) * FX_POSITION
                        money = money + profit
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                    if bef - float( Decimal("0.001") * Decimal(str(SPREAD)) ) >= aft:
                        # 正解の場合
                        if FX == False:
                            money = money + PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                        cnt_down_cor = cnt_down_cor + 1
                        win_flg = True

                    else:
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOFF * -1)

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

            # 予想と実際の差とその時の分散を表示
            # 実際との差が小さければ分散も少ないはず
            plt.scatter(err_list, sigma_list)
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