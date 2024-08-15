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
from decimal import Decimal
from DataSequence2 import DataSequence2

import socket
import conf_class
from util import *


host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)


"""
回帰regression用の検証用

"""
c = None

start = datetime(2021, 1, 1)
end = datetime(2022, 7, 1)

# 2秒ごとの成績を計算する場合
per_sec_flg = True

def tmp_create_float(x, y):
    return float(Decimal(str(y)) * (Decimal(str(x)) / Decimal("10000") + Decimal("1")))


create_float = np.vectorize(tmp_create_float, otypes=[np.ndarray])

def getAcc(res, pred_close_list, real_close_list, target_spread_list, border):
    global c
    mid_tmp = res[:,0]

    #output(mid_tmp)
    # 現実のレートに換算する
    #mid = create_float(mid_tmp, pred_close_list)
    #mid = pred_close_list * ((mid_tmp / 10000) + 1)
    mid = get_rate(mid_tmp, pred_close_list, multi=10000)

    #上昇予想 スプレッドを加味したレートより予想が上の場合 ベット対象
    if c.REAL_SPREAD_FLG:
        target_spread_list = target_spread_list + 1
        target_spread_list = np.array(target_spread_list, dtype=np.dtype(Decimal))

        up_ind = np.where(mid >= pred_close_list + np.array(list(map(float,(Decimal("0.001") * target_spread_list + Decimal(str(border)) )))))
        up_win = np.where(real_close_list[up_ind] >= pred_close_list[up_ind] + np.array(list(map(float,( Decimal("0.001")  * target_spread_list[up_ind])))))

        # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
        down_ind = np.where(mid <= pred_close_list - np.array(list(map(float,(Decimal("0.001") * target_spread_list + Decimal(str(border)) )))))
        down_win = np.where(real_close_list[down_ind] <= pred_close_list[down_ind] - np.array(list(map(float,( Decimal("0.001") * target_spread_list[down_ind])))))

        """
        up_ind = np.where(mid >= pred_close_list + (0.001 * target_spread_list) + border)
        up_win = np.where(
            real_close_list[up_ind] >= pred_close_list[up_ind] + (0.001 * target_spread_list[up_ind]))

        # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
        down_ind = np.where(mid <= pred_close_list - (0.001 * target_spread_list) - border)
        down_win = np.where(
            real_close_list[down_ind] <= pred_close_list[down_ind] - (0.001 * target_spread_list[down_ind]))
        """
    else:
        up_ind = np.where(
            mid >= pred_close_list + float(Decimal("0.001") * Decimal(str(c.SPREAD))) + border)
        up_win = np.where(real_close_list[up_ind] >= pred_close_list[up_ind] + float(
            Decimal("0.001") * Decimal(str(c.SPREAD))))

        # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
        down_ind = np.where(
            mid <= pred_close_list - float(Decimal("0.001") * Decimal(str(c.SPREAD))) - border)
        down_win = np.where(real_close_list[down_ind] <= pred_close_list[down_ind] - float(
            Decimal("0.001") * Decimal(str(c.SPREAD))))

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

def make_data(conf, target_spreads):
    global c
    c = conf

    dataSequence2 = DataSequence2(conf, start, end, True, False, target_spreads)
    return dataSequence2

def do_predict(conf, dataSequence2, target_spreads):
    global c
    c = conf

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

    #予想対象の場合-1が入っている
    train_list_index = np.array(dataSequence2.get_train_list_index())

    # 全atr値のリスト(Noneあり)
    atr_list = np.array(dataSequence2.get_atr_list())

    # 予想対象のatrのリスト
    target_atr_list = atr_list[np.where(train_list_index != -1)[0]]


    model_suffix = []
    for i in range(c.EPOCH):
        model_suffix.append(str(i + 1))

    #model_suffix = ["90*23",]

    border_list = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02] # 正解率と獲得金額のみ表示
    #border_list = []

    border_list_show = []  # グラフも表示

    """
    FILE_PREFIXS = [
        "GBPJPY_CATEGORY_BIN_UP_LSTM_BET6_TERM30_INPUT30_INPUT_LEN240_L-UNIT24_D-UNIT_SPREAD4_201601_202012_L-RATE0.01_LOSS-C-ENTROPY_ADAM_d1_IDL1_BS10240_SEED0_SHUFFLE",
    ]
    """
    #model_suffix = ["32",]
    #border_list = [ 0.5, ]
    #border_list_show = border_list

    # BIN_BOTH用以外用
    FILE_PREFIXS = [c.FILE_PREFIX]

    # output("model:", FILE_PREFIX)
    total_acc_txt = []
    total_money_txt = []
    max_val_suffix = {"val":0,}

    line_val = 0.54

    for file in FILE_PREFIXS:
        output("FILE_PREFIX:", file)
        output("target_spreads", target_spreads)

        for suffix in model_suffix:
            output("suffix:", suffix)

            load_dir = "/app/model/fx/" + file + "-" + suffix

            model = tf.keras.models.load_model(load_dir)

            # ndarrayで返って来る
            predict_list = model.predict_generator(dataSequence2,
                                                   steps=None,
                                                   max_queue_size=c.PROCESS_COUNT * 1,
                                                   use_multiprocessing=True,
                                                   verbose=0)
            under_dict = {}
            over_dict = {}

            r = redis.Redis(host='localhost', port=6379, db=c.DB_TRADE_NO)

            for b in border_list:

                # 予想結果表示用テキストを保持
                result_txt = []
                result_txt_trade = []

                max_drawdown = 0
                drawdown = 0
                max_drawdowns = []

                max_drawdown_trade = 0
                drawdown_trade = 0
                max_drawdowns_trade = []

                border = b

                Acc, total_num, correct_num = getAcc(predict_list, pred_close_list, real_close_list, target_spread_list, border)

                # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
                result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

                profit = (c.PAYOUT * correct_num) - ((total_num - correct_num) * c.PAYOFF)
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
                        output(i)
                    continue

                money_y = []
                money_trade_y = []
                money_tmp = {}
                money_trade_tmp = {}

                spread_trade = {}
                spread_win = {}
                spread_trade_real = {}
                spread_win_real = {}

                money = c.START_MONEY  # 始めの所持金
                money_trade = c.START_MONEY

                cnt_up_cor = 0  # upと予想して正解した数
                cnt_up_wrong = 0  # upと予想して不正解だった数

                cnt_down_cor = 0  # downと予想して正解した数
                cnt_down_wrong = 0  # downと予想して不正解だった数

                # 2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
                # {key:[trade_num,win_num]}
                per_sec_dict = {}
                per_sec_dict_real = {}

                per_sec = 2
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
                """
                # 分秒ごとの成績
                per_minsec_dict = {}
                for i in range(60):
                    for j in range(60):
                        if j % per_sec == 0:
                            per_minsec_dict[str(i) + "-" + str(j)] = [0, 0]
                """

                # 理論上の予測ごとの勝率 key:確率 val:{win_cnt:勝った数, lose_cnt:負けた数}
                div_list = {}
                div_list_real = {}

                bet_cnt = 0
                win_cnt = 0

                trade_cnt = 0
                true_cnt = 0

                trade_win_cnt = 0
                trade_wrong_win_cnt = 0
                trade_wrong_lose_cnt = 0

                atr_win_list = []
                atr_lose_list = []
                atr_total_list = []

                for pred, bef, aft, s, c, real, sp, atr in zip(predict_list, pred_close_list, real_close_list, target_score_list,
                                                      change_list, correct_list, target_spread_list,target_atr_list):
                    spread = sp
                    if c.REAL_SPREAD_FLG:
                        spread = sp + 1

                    div = 0
                    if pred[0] < 0:
                        #マイナスの場合
                        div = str(pred[0])[0:4]#マイナス符号が入るため1文字増やす

                        if c.EXCEPT_DIVIDE_MAX !=0 and pred[0] < c.EXCEPT_DIVIDE_MAX * -1:
                            #大きすぎるdivideを予想した場合は取引しない
                            money_tmp[s] = money
                            continue
                    else:
                        div = str(pred[0])[0:3]

                        if c.EXCEPT_DIVIDE_MAX !=0 and pred[0] > c.EXCEPT_DIVIDE_MAX:
                            #大きすぎるdivideを予想した場合は取引しない
                            money_tmp[s] = money
                            continue

                    mid = float(Decimal(str(bef)) * (Decimal(str(pred[0])) / Decimal("10000") + Decimal("1")))  # 予測レート

                    # 予想した時間
                    predict_t = datetime.fromtimestamp(s)
                    win_flg = False
                    bet_flg = False

                    win_trade_flg = False
                    bet_trade_flg = False

                    startVal = "NULL"
                    endVal = "NULL"
                    result = "NULL"

                    tradeReult = []

                    if c.TRADE_FLG:
                        tradeReult = r.zrangebyscore(c.DB_TRADE_NAME, s, s)
                        if len(tradeReult) == 0:
                            # 取引履歴がない場合1秒後の履歴として残っているかもしれないので取得
                            tradeReult = r.zrangebyscore(c.DB_TRADE_NAME, s + 1, s + 1)

                        if len(tradeReult) != 0:
                            bet_trade_flg = True
                            trade_cnt = trade_cnt + 1
                            tmps = json.loads(tradeReult[0].decode('utf-8'))
                            startVal = tmps.get("startVal")
                            endVal = tmps.get("endVal")
                            result = tmps.get("result")
                            if result == "win":
                                win_trade_flg = True
                                trade_win_cnt = trade_win_cnt + 1
                                money_trade = money_trade + c.PAYOUT
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade,
                                                                                   drawdown_trade, c.PAYOUT)
                            else:
                                win_trade_flg = False
                                money_trade = money_trade - c.PAYOFF
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade,
                                                                                   drawdown_trade, c.PAYOFF * -1)

                    # 上昇予想 スプレッドを加味したレートより下限の予想が上の場合 ベット対象
                    if bef + float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) <= mid:
                        bet_flg = True

                        atr_total_list.append(atr)

                        if bef + float(Decimal("0.001") * Decimal(str(spread))) <= aft:
                            # 正解の場合
                            atr_win_list.append(atr)

                            money = money + c.PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOUT)

                            cnt_up_cor = cnt_up_cor + 1
                            win_flg = True

                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    #理論上の結果と実トレードの結果がおなじ
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1
                                else:
                                    correct = "FALSE"
                                    trade_wrong_lose_cnt += 1
                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(aft) + "," + "UP" + ","
                                              + div + "," + "win" + "," + startVal + "," + endVal
                                              + "," + result + "," + correct)

                        else:
                            atr_lose_list.append(atr)

                            money = money - c.PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOFF * -1)

                            cnt_up_wrong = cnt_up_wrong + 1

                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    correct = "FALSE"
                                    trade_wrong_win_cnt += 1
                                else:
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1

                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(
                                    aft) + "," + "UP" + ","
                                                        + div + "," + "lose" + "," + startVal + "," + endVal
                                                        + "," + result + "," + correct)

                    elif bef - float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) >= mid:
                        bet_flg = True
                        atr_total_list.append(atr)

                        if bef - float(Decimal("0.001") * Decimal(str(spread))) >= aft:
                            # 正解の場合
                            atr_win_list.append(atr)

                            money = money + c.PAYOUT
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOUT)

                            cnt_down_cor = cnt_down_cor + 1
                            win_flg = True

                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    #理論上の結果と実トレードの結果がおなじ
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1
                                else:
                                    correct = "FALSE"
                                    trade_wrong_lose_cnt += 1
                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(aft) + "," + "DOWN" + ","
                                              + div + "," + "win" + "," + startVal + "," + endVal
                                              + "," + result + "," + correct)

                        else:
                            atr_lose_list.append(atr)

                            money = money - c.PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOFF * -1)

                            cnt_down_wrong = cnt_down_wrong + 1

                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    correct = "FALSE"
                                    trade_wrong_win_cnt += 1
                                else:
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1

                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(
                                    aft) + "," + "DOWN" + ","
                                                  + div + "," + "lose" + "," + startVal + "," + endVal
                                                  + "," + result + "," + correct)
                    money_tmp[s] = money
                    if c.TRADE_FLG:
                        money_trade_tmp[s] = money_trade

                    # 秒ごとのbetした回数と勝ち数を保持
                    if per_sec_flg:
                        if bet_flg:
                            per_sec_dict[predict_t.second][0] += 1
                            if win_flg:
                                per_sec_dict[predict_t.second][1] += 1
                        if c.TRADE_FLG:
                            if bet_trade_flg:
                                per_sec_dict_real[predict_t.second][0] += 1
                                if win_trade_flg:
                                    per_sec_dict_real[predict_t.second][1] += 1

                    # 分ごと及び、分秒ごとのbetした回数と勝ち数を保持
                    if bet_flg:
                        per_min_dict[predict_t.minute][0] += 1
                        #per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][0] += 1
                        if win_flg:
                            per_min_dict[predict_t.minute][1] += 1
                            #per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][1] += 1
                    if c.TRADE_FLG:
                        if bet_trade_flg:
                            per_min_dict_real[predict_t.minute][0] += 1
                            if win_trade_flg:
                                per_min_dict_real[predict_t.minute][1] += 1

                    # 理論上のスプレッド毎の勝率
                    if bet_flg:
                        flg = False
                        for k, v in c.SPREAD_LIST.items():
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

                    if c.TRADE_FLG:
                        if bet_trade_flg:
                            flg = False
                            for k, v in c.SPREAD_LIST.items():
                                if sp > v[0] and sp <= v[1]:
                                    spread_trade_real[k] = spread_trade_real.get(k, 0) + 1
                                    if win_trade_flg:
                                        spread_win_real[k] = spread_win_real.get(k, 0) + 1
                                    flg = True
                                    break

                            if flg == False:
                                if sp < 0:
                                    spread_trade_real["spread0"] = spread_trade_real.get("spread0", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real["spread0"] = spread_win_real.get("spread0", 0) + 1
                                else:
                                    spread_trade_real["spread16Over"] = spread_trade_real.get("spread16Over", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real["spread16Over"] = spread_win_real.get("spread16Over", 0) + 1

                    tmp_div_cnt_list = {}
                    tmp_div_cnt_list_real = {}
                    # 確率ごとのトレード数および勝率を求めるためのリスト
                    if bet_flg:
                        if div in div_list.keys():
                            tmp_div_list = div_list[div]
                            if win_flg:
                                tmp_div_cnt_list["win_cnt"] = tmp_div_list["win_cnt"] + 1
                                tmp_div_cnt_list["lose_cnt"] = tmp_div_list["lose_cnt"]
                            else:
                                tmp_div_cnt_list["win_cnt"] = tmp_div_list["win_cnt"]
                                tmp_div_cnt_list["lose_cnt"] = tmp_div_list["lose_cnt"] + 1
                            div_list[div] = tmp_div_cnt_list
                        else:
                            if win_flg:
                                tmp_div_cnt_list["win_cnt"] = 1
                                tmp_div_cnt_list["lose_cnt"] = 0
                            else:
                                tmp_div_cnt_list["win_cnt"] = 0
                                tmp_div_cnt_list["lose_cnt"] = 1
                            div_list[div] = tmp_div_cnt_list

                    if c.TRADE_FLG:
                        if bet_trade_flg:
                            if div in div_list_real.keys():
                                tmp_div_list_real = div_list_real[div]
                                if win_trade_flg:
                                    tmp_div_cnt_list_real["win_cnt"] = tmp_div_list_real["win_cnt"] + 1
                                    tmp_div_cnt_list_real["lose_cnt"] = tmp_div_list_real["lose_cnt"]
                                else:
                                    tmp_div_cnt_list_real["win_cnt"] = tmp_div_list_real["win_cnt"]
                                    tmp_div_cnt_list_real["lose_cnt"] = tmp_div_list_real["lose_cnt"] + 1
                                div_list_real[div] = tmp_div_cnt_list_real
                            else:
                                if win_trade_flg:
                                    tmp_div_cnt_list_real["win_cnt"] = 1
                                    tmp_div_cnt_list_real["lose_cnt"] = 0
                                else:
                                    tmp_div_cnt_list_real["win_cnt"] = 0
                                    tmp_div_cnt_list_real["lose_cnt"] = 1
                                div_list_real[div] = tmp_div_cnt_list_real

                    if bet_flg:
                        bet_cnt = bet_cnt + 1
                    if win_flg:
                        win_cnt = win_cnt + 1

                output("bet_cnt", bet_cnt)
                output("win_cnt", win_cnt)

                prev_money = c.START_MONEY
                prev_trade_money = c.START_MONEY

                for i, score in enumerate(score_list):
                    if score in money_tmp.keys():
                        prev_money = money_tmp[score]

                    money_y.append(prev_money)

                if c.TRADE_FLG:
                    for i, score in enumerate(score_list):
                        if score in money_trade_tmp.keys():
                            prev_trade_money = money_trade_tmp[score]

                        money_trade_y.append(prev_trade_money)

                output(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")

                fig = plt.figure()
                # 価格の遷移
                ax1 = fig.add_subplot(111)

                ax1.plot(close_list, 'g')

                ax2 = ax1.twinx()
                ax2.plot(money_y)
                if c.TRADE_FLG:
                    ax2.plot(money_trade_y, "r")

                plt.title('border:' + str(border) + " payout:" + str(c.PAYOUT) + " spread:" + str(c.SPREAD) + " money:" + str(money))

                for txt in result_txt_trade:
                    res = txt.find("FALSE")
                    if res != -1:
                        output(txt)

                output("predict money: " + str(prev_money))
                max_drawdowns.sort()
                result_txt.append("Earned Money / MaxDrawDown:" + str((prev_money - c.START_MONEY) / max_drawdowns[0]))

                for i in result_txt:
                    output(i)


                output("理論上のスプレッド毎の勝率")
                for k, v in sorted(c.SPREAD_LIST.items()):
                    if spread_trade.get(k, 0) != 0:
                        output(k, " cnt:", spread_trade.get(k, 0), " win rate:", spread_win.get(k, 0) / spread_trade.get(k))
                    else:
                        output(k, " cnt:", spread_trade.get(k, 0))
                if c.TRADE_FLG:
                    output("実際のスプレッド毎の勝率")
                    for k, v in sorted(c.SPREAD_LIST.items()):
                        if spread_trade_real.get(k, 0) != 0:
                            output(k, " cnt:", spread_trade_real.get(k, 0), " win rate:", spread_win_real.get(k, 0) / spread_trade_real.get(k))
                        else:
                            output(k, " cnt:", spread_trade_real.get(k, 0))

                for k, v in sorted(div_list.items()):
                    # 勝率
                    win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                    output("理論上の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))

                if c.TRADE_FLG:
                    for k, v in sorted(div_list_real.items()):
                        # 勝率
                        win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                        output("実際の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))


                if c.ATR_COL != "":

                    target_atr_list_sorted = np.sort(np.array(atr_total_list))
                    atr_win_arr = np.array(atr_win_list)
                    #atr_lose_arr = np.array(atr_lose_list)
                    divide = 20
                    output("全atrの値を小さい方から1/20ごとに勝率を算出")
                    for i in range(divide):
                        #全atrの値を小さい方から1/10ごとに勝率を算出
                        tmp_target = int(len(target_atr_list_sorted)/divide) * i
                        tmp_target2 = int(len(target_atr_list_sorted) / divide) * (i+1) if i != divide -1 else len(target_atr_list_sorted)
                        under_val = target_atr_list_sorted[tmp_target]
                        over_val = target_atr_list_sorted[tmp_target2 - 1]
                        w_cnt = len(np.where((atr_win_arr >= under_val) & (atr_win_arr < over_val))[0])
                        #l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                        total_cnt = len(np.where((target_atr_list_sorted >= under_val) & (target_atr_list_sorted < over_val))[0])

                        output(under_val, "~" ,over_val, " rate:", w_cnt/total_cnt, " total_cnt:", total_cnt)
                    output("ATR毎の勝率")
                    for i in range(20):
                        if i !=19 :
                            w_cnt = len(np.where((atr_win_arr >= i) & (atr_win_arr < (i+1)))[0])
                            #l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                            total_cnt = len(np.where((target_atr_list_sorted >= i) & (target_atr_list_sorted < (i+1)))[0])
                            if total_cnt != 0:
                                output(i, "~", i + 1, " rate:", w_cnt / total_cnt, " total_cnt:", total_cnt)
                        else:
                            w_cnt = len(np.where(atr_win_arr >= i)[0])
                            # l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                            total_cnt = len(np.where(target_atr_list_sorted >= i)[0])
                            if total_cnt != 0:
                                output(i, "~", " rate:", w_cnt / total_cnt, " total_cnt:", total_cnt)

                if trade_cnt != 0:
                    output("trade cnt: " + str(trade_cnt))
                    output("trade correct: " + str(true_cnt / trade_cnt))
                    output("trade wrong cnt: " + str(trade_cnt - true_cnt))
                    output("trade wrong win cnt: " + str(trade_wrong_win_cnt))
                    output("trade wrong lose cnt: " + str(trade_wrong_lose_cnt))
                    output("trade accuracy: " + str(trade_win_cnt / trade_cnt))
                    output("trade money: " + str(prev_trade_money))
                    output("trade cnt rate: " + str(trade_cnt / total_num))

                if per_sec_flg:
                    # 理論上の秒ごとの勝率
                    per_sec_winrate_dict = {}
                    for i in per_sec_dict.keys():
                        if per_sec_dict[i][0] != 0:
                            win_rate = per_sec_dict[i][1] / per_sec_dict[i][0]
                            per_sec_winrate_dict[i] = (win_rate,per_sec_dict[i][0])
                        else:
                            per_sec_winrate_dict[i] = (0,0)

                    output("理論上の秒毎の勝率悪い順:" )
                    worst_sorted = sorted(per_sec_winrate_dict.items(), key=lambda x: x[1][0])
                    for i in worst_sorted:
                        output(i[0], i[1][0], i[1][1])

                    if c.TRADE_FLG:
                        # 実際の秒ごとの勝率
                        per_sec_winrate_dict_real = {}
                        for i in per_sec_dict_real.keys():
                            if per_sec_dict_real[i][0] != 0:
                                win_rate = per_sec_dict_real[i][1] / per_sec_dict_real[i][0]
                                per_sec_winrate_dict_real[i] = (win_rate, per_sec_dict_real[i][0])
                            else:
                                per_sec_winrate_dict_real[i] = (0,0)

                        output("実際の秒毎の勝率悪い順:")
                        worst_sorted = sorted(per_sec_winrate_dict_real.items(), key=lambda x: x[1][0])
                        for i in worst_sorted:
                            output(i[0], i[1][0], i[1][1])

                # 理論上の分ごとの勝率
                per_min_winrate_dict = {}
                for i in per_min_dict.keys():
                    if per_min_dict[i][0] != 0:
                        win_rate = per_min_dict[i][1] / per_min_dict[i][0]
                        per_min_winrate_dict[i] = (win_rate,per_min_dict[i][0])
                    else:
                        per_min_winrate_dict[i] = (0,0)

                output("理論上の分毎の勝率悪い順:")
                worst_sorted = sorted(per_min_winrate_dict.items(), key=lambda x: x[1][0])
                for i in worst_sorted:
                    output(i[0], i[1][0], i[1][1])

                if c.TRADE_FLG:
                    # 実際の分ごとの勝率
                    per_min_winrate_dict_real = {}
                    for i in per_min_dict_real.keys():
                        if per_min_dict_real[i][0] != 0:
                            win_rate = per_min_dict_real[i][1] / per_min_dict_real[i][0]
                            per_min_winrate_dict_real[i] = (win_rate, per_min_dict_real[i][0])
                        else:
                            per_min_winrate_dict_real[i] = (0, 0)

                    output("実際の分毎の勝率悪い順:")
                    worst_sorted = sorted(per_min_winrate_dict_real.items(), key=lambda x: x[1][0])
                    for i in worst_sorted:
                        output(i[0], i[1][0], i[1][1])

                """
                # 理論上の分秒ごとの勝率
                per_minsec_winrate_dict = {}
                for i in per_minsec_dict.keys():
                    if per_minsec_dict[i][0] != 0:
                        win_rate = per_minsec_dict[i][1] / per_minsec_dict[i][0]
                        per_minsec_winrate_dict[i] = win_rate
                    else:
                        per_minsec_winrate_dict[i] = 0
    
                worst_sorted = sorted(per_minsec_winrate_dict.items(), key=lambda x: x[1])
                output("理論上の分秒毎の勝率悪い順:" )
                for i in worst_sorted[:10]:
                    output(i[0], i[1])
    
                best_sorted = sorted(per_minsec_winrate_dict.items(), key=lambda x: -x[1])
                output("理論上の分秒毎の勝率良い順:" )
                for i in best_sorted[:10]:
                    output(i[0], i[1])
                """

                output("MAX DrawDowns(理論上のドローダウン)")
                output(max_drawdowns[0:10])

                drawdown_cnt = {}
                for i in max_drawdowns:
                    for k, v in c.DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                            break
                for k, v in sorted(c.DRAWDOWN_LIST.items()):
                    output(k, drawdown_cnt.get(k, 0))

                if c.TRADE_FLG:
                    output("MAX DrawDowns(実トレードのドローダウン)")
                    max_drawdowns_trade.sort()
                    output(max_drawdowns_trade[0:10])
                    drawdown_cnt = {}
                    for i in max_drawdowns_trade:
                        for k, v in c.DRAWDOWN_LIST.items():
                            if i < v[0] and i >= v[1]:
                                drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                                break
                    for k, v in sorted(c.DRAWDOWN_LIST.items()):
                        output(k, drawdown_cnt.get(k,0))

                """
                max_drawdowns_np = np.array(max_drawdowns)
                df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
                output(df)
                """

                plt.show()

            if "acc" in under_dict.keys() and "acc" in over_dict.keys():
                tmp_val = under_dict["money"] - (
                        (under_dict["money"] - over_dict["money"]) / (over_dict["acc"] - under_dict["acc"]) * (
                        line_val - under_dict["acc"]))
                output("Acc:", line_val, " Money:", tmp_val)
                total_money_txt.append(tmp_val)
                if max_val_suffix["val"] < tmp_val:
                    max_val_suffix["val"] = tmp_val
                    max_val_suffix["suffix"] = suffix
            else:
                output("Acc:", line_val, " Money:")
                total_money_txt.append("")

        if "suffix" in max_val_suffix.keys():
            output("max_val_suffix:", max_val_suffix["suffix"])

        output("total_money_txt")
        for i in total_money_txt:
            if i != "":
                output(int(i))
            else:
                output("")

if __name__ == "__main__":
    conf = conf_class.ConfClass()
    conf.change_real_spread_flg(True)

    spread_lists = [[1,2,3,4,5,6,7,8]] #confからではなくことなったスプレッドでテストしたい場合
    #spread_lists = [[conf.SPREAD]]
    #spread_lists = [[1,]]

    for list in spread_lists:
        dataSequence2 = make_data(conf, list)
        do_predict(conf, dataSequence2, list)

    output("END!!!")