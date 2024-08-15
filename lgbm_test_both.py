import copy
import datetime
import pickle
import time
from matplotlib import pyplot as plt
import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import conf_class_lgbm
import numpy as np
import socket
from util import *

"""
LGBMの
LEARNING_TYPE == "CATEGORY_BIN_BOTH"
の場合専用
"""
host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)


def get_money(conf, win_cnt, bet_cnt):
    money = (win_cnt * conf.PAYOUT) - ((bet_cnt - win_cnt) * conf.PAYOFF)

    return money
# border以上の予想パーセントをしたものから正解率と予想数と正解数を返す

def getMoneyCat(conf, preds, y_eval):
    #up:0, same:1, dw:2

    #up予想
    ind_up =  np.where(preds == 0)[0]
    correct_bet_up = y_eval[ind_up]

    bet_cnt_up = len(ind_up)

    pred_bet_up = np.full(bet_cnt_up, 0)
    eq_up = np.equal(pred_bet_up, correct_bet_up)
    win_cnt_up = int(len(np.where(eq_up == True)[0]))

    #down予想
    ind_dw = np.where(preds == 2)[0]
    correct_bet_dw = y_eval[ind_dw]

    bet_cnt_dw = len(ind_dw)

    pred_bet_dw = np.full(bet_cnt_dw, 2)
    eq_dw = np.equal(pred_bet_dw, correct_bet_dw)
    win_cnt_dw = int(len(np.where(eq_dw == True)[0]))

    bet_cnt = bet_cnt_up + bet_cnt_dw
    win_cnt = win_cnt_up + win_cnt_dw

    if bet_cnt == 0:
        Acc = 0
    else:
        Acc = win_cnt / bet_cnt

    money = get_money(conf, win_cnt, bet_cnt)
    money_up = get_money(conf, win_cnt_up, bet_cnt_up)
    money_dw = get_money(conf, win_cnt_dw, bet_cnt_dw)

    return bet_cnt, win_cnt, Acc, money, bet_cnt_up, win_cnt_up, money_up, bet_cnt_dw, win_cnt_dw, money_dw

def getMoneyRgr(conf, pred, border, correct):

    up_ind = np.where(pred >= float(Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD)) + Decimal(str(border))) )
    up_win = np.where(correct[up_ind] >= float(Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD))) )

    # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
    dw_ind = np.where(pred <= float(Decimal("-1") * Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD)) - Decimal(str(border))) )
    dw_win = np.where(correct[dw_ind] <= float(Decimal("-1") * Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD))) )

    bet_cnt = len(dw_ind[0]) + len(up_ind[0])
    win_cnt = len(dw_win[0]) + len(up_win[0])

    if bet_cnt == 0:
        Acc = 0
    else:
        Acc = win_cnt / bet_cnt

    money = (win_cnt * conf.PAYOUT) - ((bet_cnt - win_cnt) * conf.PAYOFF)

    return bet_cnt, win_cnt, Acc, money

# 全体の正解率を返す(SAME予想も含める)
def getAccTotalSame(res, dataY):

    eq = np.equal(res.argmax(axis=1), dataY.argmax(axis=1))
    cor_length = int(len(np.where(eq == True)[0]))

    total_num = len(res)
    correct_num = cor_length

    if total_num == 0:
        Acc = 0
    else:
        Acc = correct_num / total_num

    return Acc

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


def do_test(conf, line_val, test_data_load_path, cat_up_model_file="", cat_dw_model_file="", up_iteration=0, dw_iteration=0, rgr_model_file="", rgr_iteration=0):

    #テスト用データロード
    with open(test_data_load_path, 'rb') as f:
        test_lmd = pickle.load(f)

    if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        tmp_x = test_lmd.get_x()
        tmp_col_up = copy.copy(conf.INPUT_DATA)
        tmp_col_dw = copy.copy(conf.INPUT_DATA)

        if conf.USE_PRED:
            if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                tmp_col_up.extend(["UP"])
                tmp_col_dw.extend(["DW"])

        x_eval_up = tmp_x.loc[:, tmp_col_up]
        x_eval_dw = tmp_x.loc[:, tmp_col_dw]

        y_eval = test_lmd.get_y()

    # レート(ndarray)
    open_rate_arr = test_lmd.get_open()

    # 正解レート(ndarray)
    answer_rate_arr = test_lmd.get_answer()

    # 対象データのインデックスを保持、対象外:-1(ndarray)
    target_idx_arr = test_lmd.get_train_idx()

    # スコア(ndarray)
    score_arr = test_lmd.get_score()

    # ATR値(ndarray)
    atr_arr = test_lmd.get_atr()

    result_txt = []
    preds = []

    #モデルロード
    if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        line_val_up, line_val_dw = line_val

        if up_iteration == 0 or dw_iteration == 0:
            output("iteration is not defined!!!")
            exit(1)

        if cat_up_model_file == "" or cat_dw_model_file == "":
            output("model_file is not defined!!!")
            exit(1)
        else:
            bst_up = lgb.Booster(model_file=conf.MODEL_DIR + cat_up_model_file)
            bst_dw = lgb.Booster(model_file=conf.MODEL_DIR + cat_dw_model_file)

            preds_up = bst_up.predict(x_eval_up, num_iteration=up_iteration)
            preds_dw = bst_dw.predict(x_eval_dw, num_iteration=dw_iteration)


            for pred_up, pred_dw in zip(preds_up, preds_dw):
                if pred_up >= pred_dw and pred_up >= line_val_up:
                    #UP予想
                    preds.append(0)
                elif pred_dw > pred_up and pred_dw >= line_val_dw:
                    #DW予想
                    preds.append(2)
                else:
                    #SAME予想
                    preds.append(1)
        preds = np.array(preds)
        """
        bet_cnt, win_cnt, Acc, money, bet_cnt_up, win_cnt_up, money_up, bet_cnt_dw, win_cnt_dw, money_dw = getMoneyCat(conf, preds, y_eval )

        if bet_cnt != 0:
            result_txt.append("Accuracy:" + str(Acc))
            result_txt.append("bet_cnt:" + str(bet_cnt) + " win_cnt:" + str(win_cnt))
            result_txt.append("Earned Money:" + str(money))
        if bet_cnt_up !=0 :
            result_txt.append("Accuracy UP:" + str(win_cnt_up/bet_cnt_up))
            result_txt.append("bet_cnt_up:" + str(bet_cnt_up) + " win_cnt_up:" + str(win_cnt_up))
            result_txt.append("Earned Money UP:" + str(money_up))
        if bet_cnt_dw !=0 :
            result_txt.append("Accuracy DW:" + str(win_cnt_dw/bet_cnt_dw))
            result_txt.append("bet_cnt_dw:" + str(bet_cnt_dw) + " win_cnt_dw:" + str(win_cnt_dw))
            result_txt.append("Earned Money DW:" + str(money_dw))

        for txt in result_txt:
            output(txt)
        """

    money_y = []
    money_tmp = {}
    money = conf.START_MONEY  # 始めの所持金

    bet_cnt = 0
    win_cnt = 0

    bet_cnt_up = 0
    win_cnt_up = 0

    bet_cnt_dw = 0
    win_cnt_dw = 0

    max_drawdown = 0
    drawdown = 0
    max_drawdowns = []

    prev_trade_time = 0

    target_len = np.where(target_idx_arr != -1)[0] #予想対象の数
    if len(target_len) != len(preds) or len(target_len) != len(y_eval) :
        #予想対象の数が合っているか確認
        output("target_len i not correct!!!", len(target_len) , len(preds), len(y_eval))
        exit(1)

    #予想対象外のときはNoneを入れる
    full_preds = []
    full_y_val = []
    tmp_cnt = 0
    for idx in target_idx_arr:
        if idx != -1:
            full_preds.append(preds[tmp_cnt])
            full_y_val.append(y_eval[tmp_cnt])
            tmp_cnt += 1
        else:
            full_preds.append(None)
            full_y_val.append(None)

    if len(target_idx_arr) != len(full_preds):
        output("full_preds length no correct!!", len(target_idx_arr), len(full_preds))


    for i, (pred, y_val, score, target_idx, open_rate, answer_rate)  in enumerate(zip(full_preds, full_y_val, score_arr, target_idx_arr, open_rate_arr, answer_rate_arr)):

        if pred == None:
            #予想対象外の場合はスキップ
            money_tmp[score] = money
            continue
        else:
            if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and pred == 1:
                #予想がsameの場合はスキップ
                money_tmp[score] = money
                continue

        # 取引間隔を開ける場合、指定時間が経過していなければスキップする
        if conf.RESTRICT_FLG:
            # 取引制限をかける
            if prev_trade_time != 0 and score - prev_trade_time < conf.RESTRICT_SEC:
                money_tmp[score] = money
                continue

        bet_cnt += 1
        prev_trade_time = score

        win_flg = True if pred == y_val else False

        if win_flg:
            win_cnt += 1

        if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and pred == 0:
            bet_cnt_up += 1
            # Up predict
            if win_flg:
                win_cnt_up += 1
                money = money + conf.PAYOUT
                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, conf.PAYOUT)
            else:
                money = money - conf.PAYOFF
                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,conf.PAYOFF * -1)

        elif conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and pred == 2:
            bet_cnt_dw += 1
            # Down predict
            if win_flg:
                win_cnt_dw += 1
                money = money + conf.PAYOUT
                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, conf.PAYOUT)
            else:
                money = money - conf.PAYOFF
                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown,conf.PAYOFF * -1)

        money_tmp[score] = money

    prev_money = conf.START_MONEY

    output("bet_cnt:", bet_cnt)

    for i, sc in enumerate(score_arr):
        if sc in money_tmp.keys():
            prev_money = money_tmp[sc]

        money_y.append(prev_money)

    #output("predict money: " + str(prev_money))
    max_drawdowns.sort()

    result_txt_detail = []

    Acc = 0 if bet_cnt == 0 else win_cnt / bet_cnt
    result_txt_detail.append("Accuracy:" + str(Acc))
    result_txt_detail.append("Total:" + str(bet_cnt) + " Correct:" + str(win_cnt))
    earned_money = prev_money - conf.START_MONEY

    result_txt_detail.append("Earned Money:" + str(earned_money))
    result_txt_detail.append("Earned Money / MaxDrawDown:" + str(earned_money / max_drawdowns[0]))

    if bet_cnt_up !=0 :
        result_txt_detail.append("Accuracy UP:" + str(win_cnt_up/bet_cnt_up))
        result_txt_detail.append("bet_cnt_up:" + str(bet_cnt_up) + " win_cnt_up:" + str(win_cnt_up))
        result_txt_detail.append("Earned Money UP:" + str(get_money(conf, win_cnt_up, bet_cnt_up)))
    if bet_cnt_dw !=0 :
        result_txt_detail.append("Accuracy DW:" + str(win_cnt_dw/bet_cnt_dw))
        result_txt_detail.append("bet_cnt_dw:" + str(bet_cnt_dw) + " win_cnt_dw:" + str(win_cnt_dw))
        result_txt_detail.append("Earned Money DW:" + str(get_money(conf, win_cnt_dw, bet_cnt_dw)))

    for i in result_txt_detail:
        output(i)


    output("MAX DrawDowns(理論上のドローダウン)")
    output(max_drawdowns[0:10])

    drawdown_cnt = {}
    for i in max_drawdowns:
        for k, v in conf.DRAWDOWN_LIST.items():
            if i < v[0] and i >= v[1]:
                drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                break
    for k, v in sorted(conf.DRAWDOWN_LIST.items()):
        output(k, drawdown_cnt.get(k, 0))

    """
    fig = plt.figure()
    # 価格の遷移
    ax1 = fig.add_subplot(111)

    ax1.plot(open_rate_arr, 'g')

    ax2 = ax1.twinx()
    ax2.plot(money_y)

    if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        plt.title("payout:" + str(conf.PAYOUT) + " earned money:" + str(earned_money))

    plt.show()
    """

if __name__ == '__main__':
    # 処理時間計測
    start = time.time()

    conf = conf_class_lgbm.ConfClassLgbm()
    if conf.USE_H:
        conf.INPUT_DATA.extend(["hour"])
    if conf.USE_M:
        conf.INPUT_DATA.extend(["min"])
    if conf.USE_S:
        conf.INPUT_DATA.extend(["sec"])
    if conf.USE_DAY:
        conf.INPUT_DATA.extend(["day"])

    test_data_load_path = "/db2/data/USDJPY_bt2_2sub100-hms_db1"

    cat_up_model_file = "USDJPY_CATEGORY_BIN_UP_BET2_TERM30_2sub100-30satr5_h_m_s_p_DT-cpu_ESR-0_LR-0.1_MB-255_MD--1_MDIL-20_NBR-500_NBRT-500_NL-31_NT-10_S-42_201601_202012_TUNER_PV1"
    cat_dw_model_file = "USDJPY_CATEGORY_BIN_DW_BET2_TERM30_2sub100-30satr5_h_m_s_p_DT-cpu_ESR-0_LR-0.1_MB-255_MD--1_MDIL-20_NBR-500_NBRT-500_NL-31_NT-10_S-42_201601_202012_TUNER_PV1"

    up_iteration = 97
    dw_iteration = 81

    line_val = [0.585,0.595] #up, dwの閾値

    output("RESTRICT_FLG:",conf.RESTRICT_FLG)
    if conf.RESTRICT_FLG:
        output("RESTRICT_SEC:",conf.RESTRICT_SEC)

    do_test(conf, line_val, test_data_load_path, cat_up_model_file, cat_dw_model_file, up_iteration, dw_iteration,)


    process_time = time.time() - start
    output("process_time:", process_time / 60, "分")
