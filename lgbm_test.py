import pickle
import time

import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import conf_class_lgbm
import numpy as np
import socket
from util import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

def get_money(conf, win_cnt, bet_cnt):
    money = (win_cnt * conf.PAYOUT) - ((bet_cnt - win_cnt) * conf.PAYOFF)

    return money

def getMoneyCat(conf, pred, border, correct):

    ind = np.where(pred >= border)[0]
    bet_cnt = len(ind)

    pred_bet = pred[ind]
    correct_bet = correct[ind]
    # pred_bet_round = np.round(pred_bet)
    pred_bet_round = np.full(bet_cnt, 1)  # border以上の予想は１にする 答えのラベルとするため

    eq = np.equal(pred_bet_round, correct_bet)
    win_cnt = int(len(np.where(eq == True)[0]))

    if bet_cnt == 0:
        Acc = 0
    else:
        Acc = win_cnt / bet_cnt

    money = (win_cnt * conf.PAYOUT) - ((bet_cnt - win_cnt) * conf.PAYOFF)

    """
    print("bet_cnt:", bet_cnt)
    print("win_cnt:", win_cnt)
    print("Acc:", Acc)
    print("money:",money)
    """

    return bet_cnt, win_cnt, Acc, money

def getMoneyRgr(conf, pred, border, correct):

    up_ind = np.where(pred >= float(Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD)) + Decimal(str(border))) )[0]
    up_win = np.where(correct[up_ind] >= float(Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD))) )[0]

    # 下降予想 スプレッドを加味したレートより予想が下の場合 ベット対象
    dw_ind = np.where(pred <= float(Decimal("-1") * Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD)) - Decimal(str(border))) )[0]
    dw_win = np.where(correct[dw_ind] <= float(Decimal("-1") * Decimal(str(conf.PIPS)) * Decimal(str(conf.SPREAD))) )[0]

    bet_cnt_up = len(up_ind)
    bet_cnt_dw = len(dw_ind)

    win_cnt_up = len(up_win)
    win_cnt_dw = len(dw_win)

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

def do_test(conf, line_val, iteration_range_s, iteration_range_e, test_data_load_path, skip=1, model_file=""):
    if model_file == "":
        bst = lgb.Booster(model_file=conf.MODEL_DIR + conf.FILE_PREFIX)
    else:
        bst = lgb.Booster(model_file=conf.MODEL_DIR + model_file)

    output("current_iteration", bst.current_iteration())

    with open(test_data_load_path, 'rb') as f:
        test_lmd = pickle.load(f)

    tmp_x = test_lmd.get_x()
    x_eval = tmp_x.loc[:, conf.INPUT_DATA]

    if conf.LEARNING_TYPE == "CATEGORY_BIN_UP":
        y_eval = test_lmd.get_y_up()
    elif conf.LEARNING_TYPE == "CATEGORY_BIN_DW":
        y_eval = test_lmd.get_y_dw()
    else:
        y_eval = test_lmd.get_y()

    if conf.LEARNING_TYPE == "REGRESSION":
        border_list = [0.001, 0.002, 0.003, 0.004,  ]
    elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP" or conf.LEARNING_TYPE == "CATEGORY_BIN_DW":
        border_list = [0.57,0.58,0.59, 0.6 ]

    total_money_txt = []
    max_val_iteration = {"val": 0, }

    output("")

    for i in range(bst.current_iteration()):
        iteration = i + 1
        if (iteration_range_s <= iteration and iteration <= iteration_range_e) == False:
            continue

        output("iteration:", iteration)

        if int(iteration % skip) != 0:
            total_money_txt.append("")
            continue

        under_dict = {}
        over_dict = {}
        for border_ind, border in enumerate(border_list):
            result_txt = []

            preds = bst.predict(x_eval, num_iteration=iteration)

            if conf.LEARNING_TYPE == "REGRESSION":
                bet_cnt, win_cnt, Acc, money, bet_cnt_up, win_cnt_up, money_up, bet_cnt_dw, win_cnt_dw, money_dw = getMoneyRgr(conf, preds, border, y_eval)
            elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP" or conf.LEARNING_TYPE == "CATEGORY_BIN_DW":
                bet_cnt, win_cnt, Acc, money = getMoneyCat(conf, preds, border, y_eval)

            if bet_cnt != 0:
                result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                result_txt.append("bet_cnt:" + str(bet_cnt) + " win_cnt:" + str(win_cnt))
                result_txt.append("Earned Money:" + str(money))

            if conf.LEARNING_TYPE == "REGRESSION":
                if bet_cnt_up != 0:
                    result_txt.append("Accuracy over UP" + str(border) + ":" + str(win_cnt_up/bet_cnt_up))
                    result_txt.append("bet_cnt_up:" + str(bet_cnt_up) + " win_cnt_up:" + str(win_cnt_up))
                    result_txt.append("Earned Money UP:" + str(money_up))
                if bet_cnt_dw != 0:
                    result_txt.append("Accuracy over DW" + str(border) + ":" + str(win_cnt_dw/bet_cnt_dw))
                    result_txt.append("bet_cnt_dw:" + str(bet_cnt_dw) + " win_cnt_dw:" + str(win_cnt_dw))
                    result_txt.append("Earned Money DW:" + str(money_dw))

            if line_val >= Acc:
                if not "acc" in over_dict.keys():
                    under_dict["acc"] = Acc
                    under_dict["money"] = money
            else:
                if not "acc" in over_dict.keys():
                    over_dict["acc"] = Acc
                    over_dict["money"] = money

            for i in result_txt:
                output(i)

        if "acc" in under_dict.keys() and "acc" in over_dict.keys():
            tmp_val = under_dict["money"] - (
                    (under_dict["money"] - over_dict["money"]) / (over_dict["acc"] - under_dict["acc"]) * (
                    line_val - under_dict["acc"]))
            output("Acc:", line_val, " Money:", tmp_val)
            total_money_txt.append(tmp_val)
            if max_val_iteration["val"] < tmp_val:
                max_val_iteration["val"] = tmp_val
                max_val_iteration["iteration"] = iteration
        else:
            output("Acc:", line_val, " Money:")
            total_money_txt.append("")

    if "iteration" in max_val_iteration.keys():
        output("max_val_iteration:", max_val_iteration["iteration"],max_val_iteration["val"] )

    output("total_money_txt")
    for i in total_money_txt:
        if i != "":
            output(int(i))
        else:
            output("")


if __name__ == '__main__':
    # 処理時間計測
    start = time.time()

    conf = conf_class_lgbm.ConfClassLgbm()

    line_val = 0.575
    iteration_range_s = 1
    iteration_range_e = 500
    skip = 10
    model_file = ""
    test_data_load_path = "/db2/data/USDJPY_bt5_5sub300" + conf.ANSWER_STR + "_test"

    do_test(conf, line_val, iteration_range_s, iteration_range_e, test_data_load_path, skip=skip, model_file=model_file)

    process_time = time.time() - start
    output("process_time:", process_time / 60, "分")
