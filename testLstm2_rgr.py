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
import math
"""
回帰regression用の検証用

"""

start = datetime(2020, 1, 1)
end = datetime(2021, 9, 30, 22)


def tmp_create_float(x, y):
    return y * math.pow(math.e * 0.1, (x / 10000))


create_float = np.vectorize(tmp_create_float, otypes=[np.ndarray])

def getAcc(res, pred_close_list, real_close_list, target_spread_list, border, target_predict_list):

    if METHOD == "LSTM2":
        mid_tmp = res[:, 0] + target_predict_list
    else:
        mid_tmp = res[:,0]

    #print(mid_tmp)
    # 現実のレートに換算する
    #mid = create_float(mid_tmp, pred_close_list)

    if DIRECT_FLG:
        mid = mid_tmp

    else:
        mid = pred_close_list * ((mid_tmp / 10000) + 1)

    #上昇予想 スプレッドを加味したレートより予想が上の場合 ベット対象
    if REAL_SPREAD_FLG:
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
    #dataSequence2_eval = DataSequence2(0, start, end, True, True)
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

    # 予想対象のpredict値のリスト
    target_predict_list = np.array(dataSequence2.get_target_predict_list())

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

    model_suffix = ["90*21",]

    #border_list = [ -0.001, -0.0008, -0.0006, -0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, 0.0008, ] #30秒
    #border_list = [ 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075, 0.0080, ] #30秒 Turbo

    border_list = [ -0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006,]

    border_list_show = [-0.0004, -0.0002, 0, 0.0002, 0.0004, 0.0006, ]  # グラフも表示

    # print("model:", FILE_PREFIX)
    total_acc_txt = []

    max_val_suffix = {"val":0,}

    for suffix in model_suffix:
        print(suffix)

        #FILE_PREFIX = "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-6-3-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_LOSS-HUBER"
        FILE_PREFIX = "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-4-2-1_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_LOSS-HUBER"

        load_dir = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix

        #load_dir = "/app/model/bin_op/" + "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-6-3-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_LOSS-HUBER-90*18"

        #load_dir = "/app/model/bin_op/" + "GBPJPY_REGRESSION_LSTM_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT8-8-6-3-2_D-UNIT_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202007_90_LOSS-HUBER-90*36"

        model = tf.keras.models.load_model(load_dir)

        """
        res = model.evaluate(dataSequence2_eval,
                                               steps=None,
                                               max_queue_size=PROCESS_COUNT * 1,
                                               use_multiprocessing=False,
                                               verbose=0)

        total_acc_txt.append(res)
        """

        under_dict = {}
        over_dict = {}
        line_val = 0.505
        #line_val = 0.583

        # ndarrayで返って来る
        predict_list = model.predict_generator(dataSequence2,
                                               steps=None,
                                               max_queue_size=PROCESS_COUNT * 1,
                                               use_multiprocessing=True,
                                               verbose=0)
        """
        print("close", len(close_list), close_list[:10])
        print("score", len(score_list), score_list[:10])
        print("target_score", len(target_score_list), target_score_list[:10])
        print("pred_close", len(pred_close_list), pred_close_list[:10])
        """

        #print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")
        print("")

        r = redis.Redis(host='localhost', port=6379, db=DB_TRADE_NO)

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

            if FX == False:
                Acc, total_num, correct_num = getAcc(predict_list, pred_close_list, real_close_list, target_spread_list, border, target_predict_list)
                profit = (PAYOUT * correct_num) - ((total_num - correct_num) * PAYOFF)
            else:
                Acc, total_num, correct_num, profit = getAccFx(predict_list, pred_close_list, correct_list, change_list, border)

            # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
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
                    myLogger(i)
                continue

            money_y = []
            money_trade_y = []
            money_tmp = {}
            money_trade_tmp = {}

            spread_trade = {}
            spread_win = {}
            spread_trade_real = {}
            spread_win_real = {}

            divide_bef_trade = {}
            divide_bef_win = {}

            money = START_MONEY  # 始めの所持金
            money_trade = START_MONEY

            cnt_up_cor = 0  # upと予想して正解した数
            cnt_up_wrong = 0  # upと予想して不正解だった数

            cnt_down_cor = 0  # downと予想して正解した数
            cnt_down_wrong = 0  # downと予想して不正解だった数

            # 2秒ごとの成績を計算する場合
            per_sec_flg = True
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

            # 理論上の曜日と時間ごとの勝率 key:曜日と時間 val:{win_cnt:勝った数, lose_cnt:負けた数}
            week_time_list = {}

            bet_cnt = 0
            win_cnt = 0

            trade_cnt = 0
            true_cnt = 0

            trade_win_cnt = 0
            trade_wrong_win_cnt = 0
            trade_wrong_lose_cnt = 0

            #このフラグが立っている場合、betしない
            except_flg = False

            for pred, bef, aft, s, c, real, sp, divide_bef, predict in zip(predict_list, pred_close_list, real_close_list, target_score_list,
                                                  change_list, correct_list, target_spread_list, target_divide_prev_list, target_predict_list):
                spread = sp

                if METHOD == "LSTM2":
                    real_pred = pred[0] + predict
                else:
                    real_pred = pred[0]

                if REAL_SPREAD_FLG:
                    spread = sp + 1

                div = 0
                if DIRECT_FLG:
                    mid = real_pred
                else:
                    if real_pred < 0:
                        #マイナスの場合
                        div = str(real_pred)[0:4]#マイナス符号が入るため1文字増やす
                    else:
                        div = str(real_pred)[0:3]

                    if div in EXCEPT_DIV_LIST:
                        except_flg = True
                    else:
                        except_flg = False

                    mid = float(Decimal(str(bef)) * (Decimal(str(real_pred)) / Decimal("10000") + Decimal("1")))  # 予測レート平均

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

                if TRADE_FLG:
                    if (bef + float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) <= mid  or
                        bef - float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) >= mid ) and except_flg == False:

                        tradeReult = r.zrangebyscore(DB_TRADE_NAME, s, s)
                        if len(tradeReult) == 0:
                            # 取引履歴がない場合1秒後の履歴として残っているかもしれないので取得
                            tradeReult = r.zrangebyscore(DB_TRADE_NAME, s + 1, s + 1)

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
                                money_trade = money_trade + PAYOUT
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade,
                                                                                   drawdown_trade, PAYOUT)
                            else:
                                win_trade_flg = False
                                money_trade = money_trade - PAYOFF
                                max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade,
                                                                                   drawdown_trade, PAYOFF * -1)

                # 上昇予想 スプレッドを加味したレートより下限の予想が上の場合 ベット対象
                if bef + float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) <= mid and except_flg == False:
                    bet_flg = True
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

                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                #理論上の結果と実トレードの結果がおなじ
                                correct = "TRUE"
                                true_cnt = true_cnt + 1
                            else:
                                correct = "FALSE"
                                trade_wrong_lose_cnt += 1
                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(aft) + "," + "UP" + ","
                                          + "win" + "," + startVal + "," + endVal
                                          + "," + result + "," + correct)

                    else:
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,
                                                                   PAYOFF * -1)
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
                                                    + "lose" + "," + startVal + "," + endVal
                                                    + "," + result + "," + correct)

                elif bef - float(Decimal("0.001") * Decimal(str(spread)) + Decimal(str(border))) >= mid and except_flg == False:
                    bet_flg = True

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

                        if len(tradeReult) != 0:
                            if win_trade_flg:
                                #理論上の結果と実トレードの結果がおなじ
                                correct = "TRUE"
                                true_cnt = true_cnt + 1
                            else:
                                correct = "FALSE"
                                trade_wrong_lose_cnt += 1
                            result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(bef) + "," + str(aft) + "," + "DOWN" + ","
                                          + "win" + "," + startVal + "," + endVal
                                          + "," + result + "," + correct)

                    else:
                        if FX == False:
                            money = money - PAYOFF
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,
                                                                   PAYOFF * -1)

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
                                              + "lose" + "," + startVal + "," + endVal
                                              + "," + result + "," + correct)
                money_tmp[s] = money
                if TRADE_FLG:
                    money_trade_tmp[s] = money_trade

                # 秒ごとのbetした回数と勝ち数を保持
                if per_sec_flg:
                    if bet_flg:
                        per_sec_dict[predict_t.second][0] += 1
                        if win_flg:
                            per_sec_dict[predict_t.second][1] += 1
                    if TRADE_FLG:
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
                if TRADE_FLG:
                    if bet_trade_flg:
                        per_min_dict_real[predict_t.minute][0] += 1
                        if win_trade_flg:
                            per_min_dict_real[predict_t.minute][1] += 1

                # 理論上のスプレッド毎の勝率
                if bet_flg:
                    flg = False
                    for k, v in SPREAD_LIST.items():
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

                if TRADE_FLG:
                    if bet_trade_flg:
                        flg = False
                        for k, v in SPREAD_LIST.items():
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

                # 理論上の直近の変化率毎の勝率
                if bet_flg:
                    flg = False
                    for k, v in DIVIDE_BEF_LIST.items():
                        if divide_bef > v[0] and divide_bef <= v[1]:
                            divide_bef_trade[k] = divide_bef_trade.get(k, 0) + 1
                            if win_flg:
                                divide_bef_win[k] = divide_bef_win.get(k, 0) + 1
                            flg = True
                            break

                    if flg == False:
                        if divide_bef < 0:
                            divide_bef_trade["divide0"] = divide_bef_trade.get("divide0", 0) + 1
                            if win_flg:
                                divide_bef_win["divide0"] = divide_bef_win.get("divide0", 0) + 1
                        else:
                            divide_bef_trade["divide0.5over"] = divide_bef_trade.get("divide0.5over", 0) + 1
                            if win_flg:
                                divide_bef_win["divide0.5over"] = divide_bef_win.get("divide0.5over", 0) + 1

                """
                # 曜日と時間ごとのトレード数および勝率を求めるためのリスト
                tmp_week_time_cnt_list = {}
                if bet_flg:
                    week_hour = str(predict_t.weekday()) + "-" + str(predict_t.hour)
                    if week_hour in week_time_list.keys():
                        tmp_week_time_list = week_time_list[week_hour]
                        if win_flg:
                            tmp_week_time_cnt_list["win_cnt"] = tmp_week_time_list["win_cnt"] + 1
                            tmp_week_time_cnt_list["lose_cnt"] = tmp_week_time_list["lose_cnt"]
                        else:
                            tmp_week_time_cnt_list["win_cnt"] = tmp_week_time_list["win_cnt"]
                            tmp_week_time_cnt_list["lose_cnt"] = tmp_week_time_list["lose_cnt"] + 1
                        week_time_list[week_hour] = tmp_week_time_cnt_list
                    else:
                        if win_flg:
                            tmp_week_time_cnt_list["win_cnt"] = 1
                            tmp_week_time_cnt_list["lose_cnt"] = 0
                        else:
                            tmp_week_time_cnt_list["win_cnt"] = 0
                            tmp_week_time_cnt_list["lose_cnt"] = 1
                        week_time_list[week_hour] = tmp_week_time_cnt_list

                """

                tmp_div_cnt_list = {}
                tmp_div_cnt_list_real = {}
                # 確率ごとのトレード数および勝率を求めるためのリスト
                if DIRECT_FLG == False:
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

                    if TRADE_FLG:
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

            print("bet_cnt", bet_cnt)
            print("win_cnt", win_cnt)

            #result_txtはEXCEPT_DIV_LISTを反映していない(getAccメソッドでEXCEPT_DIV_LISTを反映していない)ので上書き
            result_txt = []
            result_txt.append("Accuracy over " + str(border) + ":" + str(win_cnt/bet_cnt))
            result_txt.append("Total:" + str(bet_cnt) + " Correct:" + str(win_cnt))


            prev_money = START_MONEY
            prev_trade_money = START_MONEY

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
            # 価格の遷移
            ax1 = fig.add_subplot(111)

            ax1.plot(close_list, 'g')

            ax2 = ax1.twinx()
            ax2.plot(money_y)
            if TRADE_FLG:
                ax2.plot(money_trade_y, "r")

            if FX == True:
                plt.title('border:' + str(border) + " position:" + str(FX_POSITION) + " spread:" + str(
                    SPREAD) + " money:" + str(money))
            else:
                plt.title('border:' + str(border) + " payout:" + str(PAYOUT) + " spread:" + str(
                    SPREAD) + " money:" + str(money))

            """
            for txt in result_txt_trade:
                res = txt.find("FALSE")
                if res != -1:
                    print(txt)
            """

            print("predict money: " + str(prev_money))
            result_txt.append("Earned Money:" + str(prev_money - START_MONEY))
            for i in result_txt:
                myLogger(i)


            if trade_cnt != 0:
                print("trade cnt: " + str(trade_cnt))
                print("trade correct: " + str(true_cnt / trade_cnt))
                print("trade wrong cnt: " + str(trade_cnt - true_cnt))
                print("trade wrong win cnt: " + str(trade_wrong_win_cnt))
                print("trade wrong lose cnt: " + str(trade_wrong_lose_cnt))
                print("trade accuracy: " + str(trade_win_cnt / trade_cnt))
                print("trade money: " + str(prev_trade_money))
                print("trade cnt rate: " + str(trade_cnt / total_num))

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
            print("理論上の分秒毎の勝率悪い順:" )
            for i in worst_sorted[:10]:
                print(i[0], i[1])

            best_sorted = sorted(per_minsec_winrate_dict.items(), key=lambda x: -x[1])
            print("理論上の分秒毎の勝率良い順:" )
            for i in best_sorted[:10]:
                print(i[0], i[1])
            """

            print("理論上のスプレッド毎の勝率")
            for k, v in sorted(SPREAD_LIST.items()):
                if spread_trade.get(k, 0) != 0:
                    print(k, " cnt:", spread_trade.get(k, 0),
                          " profit:", spread_win.get(k, 0) * PAYOUT - (spread_trade.get(k) - spread_win.get(k, 0)) * PAYOFF,
                          " win rate:", spread_win.get(k, 0) / spread_trade.get(k))
                else:
                    print(k, " cnt:", spread_trade.get(k, 0))
            if TRADE_FLG:
                print("実際のスプレッド毎の勝率")
                for k, v in sorted(SPREAD_LIST.items()):
                    if spread_trade_real.get(k, 0) != 0:
                        print(k, " cnt:", spread_trade_real.get(k, 0),
                              " profit:", spread_win_real.get(k, 0) * PAYOUT - (spread_trade_real.get(k) - spread_win_real.get(k, 0)) * PAYOFF,
                              " win rate:", spread_win_real.get(k, 0) / spread_trade_real.get(k),)
                    else:
                        print(k, " cnt:", spread_trade_real.get(k, 0))

            print("理論上の直近の変化率毎の勝率")
            for k, v in sorted(DIVIDE_BEF_LIST.items()):
                if divide_bef_trade.get(k, 0) != 0:
                    print(k, " cnt:", divide_bef_trade.get(k, 0), " win rate:", divide_bef_win.get(k, 0) / divide_bef_trade.get(k))
                else:
                    print(k, " cnt:", divide_bef_trade.get(k, 0))

            for k, v in sorted(week_time_list.items()):
                # 勝率
                win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                print("曜日-時間:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))


            if DIRECT_FLG == False:
                for k, v in sorted(div_list.items()):
                    # 勝率
                    win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                    print("理論上の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))

                if TRADE_FLG:
                    for k, v in sorted(div_list_real.items()):
                        # 勝率
                        win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                        print("実際の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))

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
                myLogger(k, drawdown_cnt.get(k, 0))

            if TRADE_FLG:
                print("MAX DrawDowns(実トレードのドローダウン)")
                max_drawdowns_trade.sort()
                myLogger(max_drawdowns_trade[0:10])
                drawdown_cnt = {}
                for i in max_drawdowns_trade:
                    for k, v in DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                            break
                for k, v in sorted(DRAWDOWN_LIST.items()):
                    print(k, drawdown_cnt.get(k,0))

            """
            max_drawdowns_np = np.array(max_drawdowns)
            df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
            myLogger(df)
            """

            #plt.show()

        if "acc" in under_dict.keys() and "acc" in over_dict.keys():
            tmp_val = under_dict["money"] - ((under_dict["money"] - over_dict["money"]) /  (over_dict["acc"] - under_dict["acc"]) * (line_val - under_dict["acc"]))
            print("Acc:", line_val, " Money:", tmp_val)

            if max_val_suffix["val"] < tmp_val:
                max_val_suffix["val"] = tmp_val
                max_val_suffix["suffix"] = suffix

    if "suffix" in max_val_suffix.keys():
        print("max_val_suffix:", max_val_suffix["suffix"])

    #for txt in total_acc_txt:
    #    print(txt)

if __name__ == "__main__":
    do_predict()

    print("END!!!")