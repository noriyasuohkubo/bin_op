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
from lstm_generator2 import create_model

start = datetime(2018, 1, 1)
end = datetime(2019, 12, 31)

#2秒ごとの成績を計算する場合
per_sec_flg = True
#2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
#{key:[trade_num,win_num]}
per_sec_dict = {}
per_sec = 2
if per_sec_flg:
    for i in range(60):
        if i % per_sec == 0:
            per_sec_dict[i] = [0,0]

#border以上の予想パーセントをしたものから正解率と予想数と正解数を返す
def getAcc(res, border, dataY):
    up = res[:, 0]
    down = res[:, 2]

    #閾値以上の予想パーセンテージのみ抽出
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

    total_num = len(up_ind5) + len(down_ind5)
    correct_num = up_cor_length + down_cor_length

    if total_num ==0:
        Acc =0
    else:
        Acc = correct_num / total_num

    return Acc, total_num, correct_num

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

def get_model(single_flg):

    model = None
    if single_flg:
        # 複数GPUを使用しない CPU用
        if LOAD_TYPE == 1:
            model = tf.keras.models.load_model(MODEL_DIR_LOAD)
        elif LOAD_TYPE == 2:
            #重さのみロード
            model = create_model()
            model.load_weights(LOAD_CHK_PATH)
        else:
            #新規作成
            print("ERROR LOAD_TYPE = 0 !!!")
            exit(1)
    else:
        # モデル作成
        with tf.distribute.MirroredStrategy().scope():
            # 複数GPU使用する
            # https://qiita.com/ytkj/items/18b2910c3363b938cde4
            if LOAD_TYPE == 1:
                model = tf.keras.models.load_model(MODEL_DIR_LOAD)
            elif LOAD_TYPE == 2:
                # 重さのみロード
                model = create_model()
                model.load_weights(LOAD_CHK_PATH)
            else:
                # 新規作成
                print("ERRO LOAD_TYPE = 0 !!!")
                exit(1)

    model.summary()

    return model

def do_predict():
    model = get_model(False)

    dataSequence2 = DataSequence2(0, start, end, True)

    # 正解ラベル(ndarray)
    correct_list = dataSequence2.get_correct_list()

    # レートの変化幅(FX用)
    change_list = np.array(dataSequence2.get_change_list())

    # 全close値のリスト
    close_list = dataSequence2.get_close_list()

    # 全score値のリスト
    score_list = dataSequence2.get_score_list()

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    # ndarrayで返って来る
    predict_list = model.predict_generator(dataSequence2,
                                  steps=None,
                                  max_queue_size=PROCESS_COUNT * 1,
                                  use_multiprocessing=False,
                                  verbose=0)
    """
    print("correct", len(correct_list), correct_list[:10])
    print("change", len(change_list), change_list[:10])
    print("close", len(close_list), close_list[:10])
    print("score", len(score_list), score_list[:10])
    print("target_score", len(target_score_list), target_score_list[:10])
    print("predict", len(predict_list), predict_list[:10])
    """

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

    border_list=[0.55,0.552,0.554,0.556,0.558,0.56,] #正解率と獲得金額のみ表示
    border_list_show=[0.55,0.554,0.558,] #グラフも表示

    #予想結果表示用テキストを保持
    result_txt = []

    for b in border_list:
        max_drawdown = 0
        drawdown = 0
        max_drawdowns = []

        border = b
        Acc, total_num, correct_num = getAcc(predict_list, border, correct_list)

        #全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
        result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
        result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

        if FX == False:
            win_money = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)
            result_txt.append("Earned Money:" + str(win_money))

        if border not in border_list_show:
            continue


        ind = np.where(predict_list >=border)[0]
        #up,same,downどれかが閾値以上の予想パーセントであるもののみ抽出
        x5 = predict_list[ind,:]
        y5 = correct_list[ind,:]
        s5 = target_score_list[ind]
        c5 = change_list[ind]

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
        money_tmp = {}

        money = 1000000 #始めの所持金

        cnt_up_cor = 0 #upと予想して正解した数
        cnt_up_wrong = 0 #upと予想して不正解だった数

        cnt_down_cor = 0 #downと予想して正解した数
        cnt_down_wrong = 0 #downと予想して不正解だった数

        for x, y, s, c in zip(x5, y5, s5, c5, ):
            max = x.argmax()

            #予想した時間
            predict_t = datetime.fromtimestamp(s)
            win_flg = False

            if max == 0:
                # Up predict

                if FX:
                    profit = (c - (0.001 * SPREAD)) * FX_POSITION
                    money = money + profit
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                if max == y.argmax():
                    if FX == False:
                        money = money + PAYOUT
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOUT)

                    cnt_up_cor = cnt_up_cor + 1
                    win_flg = True

                else :
                    if FX == False:
                        money = money - PAYOFF
                        max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, PAYOFF * -1)

                    cnt_up_wrong = cnt_up_wrong + 1

            elif max == 2:
                #Down predict
                if FX:
                    #cはaft-befなのでdown予想の場合の利益として-1を掛ける
                    profit = ((c * -1)  - (0.001 * SPREAD)) * FX_POSITION
                    money = money + profit
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                if max == y.argmax():
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

            #秒ごとのbetした回数と勝ち数を保持
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
        #価格の遷移
        ax1 = fig.add_subplot(111)

        ax1.plot(close_list, 'g')

        ax2 = ax1.twinx()
        ax2.plot(money_y)

        if FX == True:
            plt.title('border:' + str(border) + " position:" + str(FX_POSITION) + " spread:" + str(SPREAD) + " money:" + str(money))
        else:
            plt.title('border:' + str(border) + " payout:" + str(FX_POSITION) + " spread:" + str(SPREAD) + " money:" + str(money))
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
                    drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                    break
        for k, v in sorted(DRAWDOWN_LIST.items()):
            myLogger(k, drawdown_cnt.get(k,0))

        """
        max_drawdowns_np = np.array(max_drawdowns)
        df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
        myLogger(df)
        """

    for i in result_txt:
        myLogger(i)


if __name__ == "__main__":

    do_predict()