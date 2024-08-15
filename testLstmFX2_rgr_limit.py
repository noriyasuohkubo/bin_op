import json
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import time
import send_mail as mail
from DataSequence2 import DataSequence2
import conf_class
import conf_class_lgbm
import numpy as np

from important_index import *
from util import *
import sys
import copy
import socket
from adabound_tf import AdaBound
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tcn import TCN  # keras-tcn
import redis
import lightgbm as lgb
from lgbm_make_data import LgbmMakeData

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + "-limit.txt"
output = output_log(output_log_name)

"""
指値注文の場合、延長しない

"""
# 設定ファイル読み込み
c = None
png_dir = "/app/fx/png/"


def get_category_tp_sl_by_pred(tp_sl_dict, pred):
    tp_sl = None
    if pred[0] >= pred[2]:
        # BUYの場合
        p = pred[0]

    elif pred[2] > pred[0]:
        # SELLの場合
        p = pred[2]

    for k, v in tp_sl_dict.items():
        start_k, end_k = k.split('-')
        start_k = float(start_k)
        end_k = float(end_k)
        if start_k <= p and p < end_k:
            tp_sl = v

    if tp_sl == None:
        print("get_category_tp_sl_by_pred cannot get tp_sl")
        exit(1)

    return tp_sl


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def fx_mean_squared_error(y_true, y_pred):
    # 予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(error ** 2 + conf_class.FX_LOSS_PNALTY * error ** 2 * not_trend_match)

    return loss


def fx_mean_squared_error2(y_true, y_pred):
    # 予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(
        error ** 2 - conf_class.FX_LOSS_PNALTY * error ** 2 + conf_class.FX_LOSS_PNALTY * 2 * error ** 2 * not_trend_match)
    return loss


def mean_squared_error_custome(y_true, y_pred):
    # 誤差の３乗を罰則とする
    error = abs(y_true - y_pred)
    loss = tf.math.reduce_mean(error ** conf_class.MSE_PENALTY)

    return loss


def fx_insensitive_error(y_true, y_pred):
    # ε-感度損失:細かい誤差は気にしない
    # 閾値以上の誤差がある場合だけ罰則
    error = abs(y_true - y_pred)
    not_trend_match = tf.cast(error >= conf_class.INSENSITIVE_BORDER, tf.float32)

    loss = tf.math.reduce_mean(error ** 2 * not_trend_match)

    return loss


def negative_log_likelihood(y_true, y_pred):
    return -1 * y_pred.log_prob(y_true)


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


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


def buy_cat_cond_1(pred, border, border_ceil=""):
    if border_ceil != "":
        if border <= pred[0] and pred[2] <= pred[0] and pred[0] < border_ceil:
            return True
        else:
            return False
    else:
        if border <= pred[0] and pred[2] <= pred[0]:
            return True
        else:
            return False


def sell_cat_cond_1(pred, border, border_ceil=""):
    if border_ceil != "":
        if border <= pred[2] and pred[0] < pred[2] and pred[2] < border_ceil:
            return True
        else:
            return False
    else:
        if border <= pred[2] and pred[0] < pred[2]:
            return True
        else:
            return False


def buy_cat_cond_ext(pred, border, border_ceil=""):
    if border_ceil != "":
        if border <= pred[0] and pred[2] < pred[0] and pred[0] < border_ceil:
            return True
        else:
            return False
    else:
        if border <= pred[0] and pred[2] < pred[0]:
            return True
        else:
            return False


def sell_cat_cond_ext(pred, border, border_ceil=""):
    if border_ceil != "":
        if border <= pred[2] and pred[0] < pred[2] and pred[2] < border_ceil:
            return True
        else:
            return False
    else:
        if border <= pred[2] and pred[0] < pred[2]:
            return True
        else:
            return False


def buy_cat_cond_2(pred, border):
    if border <= pred[0] and pred[1] <= pred[0]:
        return True
    else:
        return False


def sell_cat_cond_2(pred, border):
    if border <= pred[1] and pred[0] < pred[1]:
        return True
    else:
        return False


def buy_cat_cond_ext_2(pred, border):
    if border <= pred[0]:
        return True
    else:
        return False


def sell_cat_cond_ext_2(pred, border):
    if border <= pred[1]:
        return True
    else:
        return False


def buy_d_cond_1(pred, border, border_ceil=""):
    if c.LEARNING_TYPE == "REGRESSION":
        if border_ceil != "":
            if border <= pred and pred < border_ceil:
                return True
            else:
                return False
        else:
            if border <= pred:
                return True
            else:
                return False
    elif c.LEARNING_TYPE == "REGRESSION_OCOPS":
        if pred[0] > pred[1] and pred[0] > border:
            return True
        else:
            return False


def sell_d_cond_1(pred, border, border_ceil=""):
    if c.LEARNING_TYPE == "REGRESSION":
        if border_ceil != "":
            if border <= pred * -1 and pred * -1 < border_ceil:
                return True
            else:
                return False
        else:
            if border <= pred * -1:
                return True
            else:
                return False

    elif c.LEARNING_TYPE == "REGRESSION_OCOPS":
        if pred[1] > pred[0] and pred[1] > border:
            return True
        else:
            return False


def buy_limit_cond_1(md, border, now_price, x_std_buy_limit):
    if border[0] <= md and md <= border[1] and x_std_buy_limit - now_price < 0:
        return True
    else:
        return False


def sell_limit_cond_1(md, border, now_price, x_std_sell_limit):
    if border[0] <= md * -1 and md * -1 <= border[1] and x_std_sell_limit - now_price > 0:
        return True
    else:
        return False


def buy_cond_1(x_gap, border, x_std_buy_sl, now_price, bef_mean):
    if x_gap > border and x_std_buy_sl < now_price and (bef_mean - now_price > 0 or bef_mean - now_price < 0):
        # if x_gap > border and  (bef_mean - now_price > 0 or bef_mean - now_price < 0):
        return True
    else:
        return False


def sell_cond_1(x_gap, border, x_std_sell_sl, now_price, bef_mean):
    if (x_gap * -1) > border and x_std_sell_sl > now_price and (bef_mean - now_price > 0 or bef_mean - now_price < 0):
        # if (x_gap * -1) > border and (bef_mean - now_price > 0 or bef_mean - now_price < 0):
        return True
    else:
        return False


def get_predict(file_name, dataSequence2):
    load_dir = "/app/model/bin_op/" + file_name
    if not os.path.isdir(load_dir):
        print("model not exists:" + load_dir)
        sys.exit()

    model = tf.keras.models.load_model(load_dir, custom_objects={"root_mean_squared_error": root_mean_squared_error,
                                                                 'AdaBound': AdaBound,
                                                                 "fx_mean_squared_error": fx_mean_squared_error,
                                                                 "fx_mean_squared_error2": fx_mean_squared_error2,
                                                                 "mean_squared_error_custome": mean_squared_error_custome,
                                                                 "fx_insensitive_error": fx_insensitive_error,
                                                                 "negative_log_likelihood": negative_log_likelihood,
                                                                 })
    if conf.MIXTURE_NORMAL:
        # tensorで帰ってくるようにする
        # see https://atmarkit.itmedia.co.jp/ait/articles/2003/10/news016.html
        predict_list = None

        for i in range(dataSequence2.__len__()):
            tmp_arr = dataSequence2.__getitem__(idx=i)[0]
            x_tensor = tf.convert_to_tensor(tmp_arr, dtype=tf.float32)
            # x_tensor = tf.convert_to_tensor(np.array([tmp_arr[0]]), dtype=tf.float32)
            # print(x_tensor.shape)
            tmp_y = model(x_tensor)
            # print(tmp_y)
            tmp_y_mean = tmp_y.mean().numpy()  # 平均を取得 tensorflor probabilityをインポートしないとtfp.distributions.MixtureSameFamilyオブジェクトが帰ってこない
            if i == 0:
                predict_list = tmp_y_mean
            else:
                predict_list = np.concatenate([predict_list, tmp_y_mean])
            # pred_stddev = tmp_y.stddev().numpy()
            # print(predict_list)
            # print(pred_stddev)
    else:
        predict_list = model.predict(dataSequence2,
                                     steps=None,
                                     max_queue_size=0,
                                     use_multiprocessing=False,
                                     verbose=0)
    return predict_list


def print_result(dixt, target, max_cnt):
    tmp_sorted = sorted(dixt.items(), key=lambda x: x[1][target], reverse=True)
    cnt_t = 0
    for i in tmp_sorted:
        if cnt_t > max_cnt:
            break
        output("suffix:", i[0], i[1]["profit"], i[1]["win_cnt"], i[1]["win_rate"], i[1]["profit_per_dd"], i[1]["dd"],
               i[1]["sl/bet"], i[1]["sl_cnt"], i[1]["bet_cnt"])
        cnt_t += 1


def showProfitAtr(pips, atrs, col):
    prev_atr = 0
    pips = np.array(pips)
    atrs = np.array(atrs)

    col_val_list = []
    win_rate_list = []
    profit_list = []
    avg_list = []

    tpips = float(Decimal(str(c.PIPS)) * Decimal("10"))
    if ("satr" in col) == False:
        tpips = 1

    for a in range(20):

        tmp_atr = float(Decimal(str(tpips)) * Decimal(str((a + 1))))
        tmp_ind = np.where((prev_atr <= atrs) & (atrs < tmp_atr))[0]
        tmp_pips = pips[tmp_ind]
        if len(tmp_pips) != 0:
            avg_pips = np.average(tmp_pips)
            total_pips = np.sum(tmp_pips)
            win_ind = np.where(tmp_pips >= 0)[0]
            lose_ind = np.where(tmp_pips < 0)[0]
            tmp_acc = len(win_ind) / len(tmp_pips)
            output(prev_atr, "~", tmp_atr, "acc:", tmp_acc, "cnt:", len(tmp_ind), "total pips:", total_pips,
                   "avg pips:", avg_pips, "win pips:", np.sum(tmp_pips[win_ind]), "lose pips:",
                   np.sum(tmp_pips[lose_ind]))

            col_val_list.append(prev_atr)
            win_rate_list.append(tmp_acc)
            profit_list.append(total_pips)
            avg_list.append(avg_pips)

        prev_atr = tmp_atr

    tpips = float(Decimal(str(c.PIPS)) * Decimal("100"))
    if ("satr" in conf.ATR_COL) == False:
        tpips = 10

    prev_atr = 0
    for a in range(10):
        tmp_atr = float(Decimal(str(tpips)) * Decimal(str((a + 1))))
        tmp_ind = np.where((prev_atr <= atrs) & (atrs < tmp_atr))[0]
        tmp_pips = pips[tmp_ind]
        if len(tmp_pips) != 0:
            avg_pips = np.average(tmp_pips)
            total_pips = np.sum(tmp_pips)
            win_ind = np.where(tmp_pips >= 0)[0]
            lose_ind = np.where(tmp_pips < 0)[0]
            tmp_acc = len(win_ind) / len(tmp_pips)
            output(prev_atr, "~", tmp_atr, "acc:", tmp_acc, "cnt:", len(tmp_ind), "total pips:", total_pips,
                   "avg pips:", avg_pips, "win pips:", np.sum(tmp_pips[win_ind]), "lose pips:",
                   np.sum(tmp_pips[lose_ind]))
        prev_atr = tmp_atr

    # 1以上のatrの場合を表示
    tmp_ind = np.where(tpips * 10 <= atrs)[0]
    tmp_pips = pips[tmp_ind]
    if len(tmp_pips) != 0:
        avg_pips = np.average(tmp_pips)
        total_pips = np.sum(tmp_pips)
        win_ind = np.where(tmp_pips >= 0)[0]
        lose_ind = np.where(tmp_pips < 0)[0]
        tmp_acc = len(win_ind) / len(tmp_pips)
        output(tpips * 10, "以上", "acc:", tmp_acc, "cnt:", len(tmp_ind), "total pips:", total_pips, "avg pips:",
               avg_pips, "win pips:", np.sum(tmp_pips[win_ind]), "lose pips:", np.sum(tmp_pips[lose_ind]))

    return [col_val_list, win_rate_list, profit_list, avg_list]


def showProfitIND(border, conf, pips, inds, show_plot, save_dir):
    pips = np.array(pips)
    inds = np.array(inds)

    for i, col in enumerate(conf.IND_COLS):
        tmp_target_col_list = inds[:, i]
        if len(pips) != len(tmp_target_col_list):
            print("pips length is not same with tmp_target_col_list length", len(pips), len(tmp_target_col_list))

        tmp_target_col_list = np.array(tmp_target_col_list, dtype='float32')

        output(col + "ごとの平均PIPS")

        if "atr" in col:
            col_val_list, win_rate_list, profit_list, avg_list = showProfitAtr(pips, tmp_target_col_list, col)
        else:
            col_val_list = []
            win_rate_list = []
            profit_list = []
            avg_list = []

            avg = np.average(tmp_target_col_list)
            std = np.std(tmp_target_col_list)
            # max_ind = float('{:.4f}'.format(avg + std * 3))
            # min_ind = float('{:.4f}'.format(avg - std * 3))
            max_ind = float('{:.4f}'.format(max(tmp_target_col_list)))
            min_ind = float('{:.4f}'.format(min(tmp_target_col_list)))
            width = float('{:.4f}'.format((max_ind - min_ind) / 20))

            range_start = min_ind

            for j in range(20):
                range_next = float('{:.4f}'.format(range_start + width))

                tmp_ind = np.where((range_start <= tmp_target_col_list) & (tmp_target_col_list < range_next))[0]
                tmp_pips = pips[tmp_ind]
                if len(tmp_pips) != 0:
                    avg_pips = np.average(tmp_pips)
                    total_pips = np.sum(tmp_pips)
                    win_ind = np.where(tmp_pips >= 0)[0]
                    lose_ind = np.where(tmp_pips < 0)[0]
                    tmp_acc = len(win_ind) / len(tmp_pips)
                    output(range_start, "~", range_next, "acc:", tmp_acc, "cnt:", len(tmp_ind),
                           "total pips:", total_pips, "avg pips:", avg_pips, "win pips:", np.sum(tmp_pips[win_ind]),
                           "lose pips:", np.sum(tmp_pips[lose_ind]))

                    col_val_list.append(range_start)
                    win_rate_list.append(tmp_acc)
                    profit_list.append(total_pips)
                    avg_list.append(avg_pips)

                range_start = range_next

        if show_plot:
            fig, ax1 = plt.subplots(figsize=(6.4 * 0.45, 4.8 * 0.45))
            ax1.plot(col_val_list, win_rate_list, "b-")
            ax1.set_ylabel("win_rate")

            ax2 = ax1.twinx()

            ax2.plot(col_val_list, profit_list, "r-")
            ax2.set_ylabel("profit")

            ax3 = ax1.twinx()

            ax3.plot(col_val_list, avg_list, "g-")
            ax3.set_ylabel("avg")

            tmp_title = str(border) + '_show_win_ind_col:' + col
            plt.title(tmp_title)

            filename = save_dir + "/" + tmp_title + ".png"
            fig.savefig(filename)


def showProfitTime(conf, pips, times):
    per_sec_dict = {}
    for i in range(60):
        if get_decimal_mod(i, conf.BET_TERM) == 0:
            per_sec_dict[i] = []

    # 分ごとの成績
    per_min_dict = {}
    for i in range(60):
        per_min_dict[i] = []

    # 時間ごとの成績
    per_hour_dict = {}
    for i in range(24):
        per_hour_dict[i] = []

    for s, pip in zip(times, pips, ):
        # 予想した時間
        predict_t = datetime.fromtimestamp(s)
        per_sec_dict[predict_t.second].append(pip)
        per_min_dict[predict_t.minute].append(pip)
        per_hour_dict[predict_t.hour].append(pip)

    per_sec_winrate_dict = {}
    for i in per_sec_dict.keys():
        pips_tmp_arr = np.array(per_sec_dict[i])
        bet_cnt = len(pips_tmp_arr)
        if bet_cnt != 0:
            win_ind = np.where(pips_tmp_arr >= 0)[0]
            win_cnt = len(win_ind)
            per_sec_winrate_dict[i] = [win_cnt / bet_cnt, bet_cnt, np.average(pips_tmp_arr)]

    per_min_winrate_dict = {}
    for i in per_min_dict.keys():
        pips_tmp_arr = np.array(per_sec_dict[i])
        bet_cnt = len(per_min_dict[i])
        if bet_cnt != 0:
            win_cnt = len(np.where(pips_tmp_arr >= 0)[0])
            per_min_winrate_dict[i] = [win_cnt / bet_cnt, bet_cnt, np.average(pips_tmp_arr)]

    per_hour_winrate_dict = {}
    for i in per_hour_dict.keys():
        pips_tmp_arr = np.array(per_sec_dict[i])
        bet_cnt = len(per_hour_dict[i])
        if bet_cnt != 0:
            win_cnt = len(np.where(pips_tmp_arr >= 0)[0])
            per_hour_winrate_dict[i] = [win_cnt / bet_cnt, bet_cnt, np.average(pips_tmp_arr)]

    output("理論上の秒毎の勝率悪い順(勝率,賭数,平均PIPS):")
    worst_sorted = sorted(per_sec_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0], ":", i[1][0], i[1][1], i[1][2])

    output("理論上の分毎の勝率悪い順(勝率,賭数):")
    worst_sorted = sorted(per_min_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0], ":", i[1][0], i[1][1], i[1][2])

    output("理論上の時毎の勝率悪い順(勝率,賭数):")
    worst_sorted = sorted(per_hour_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0], ":", i[1][0], i[1][1], i[1][2])

    """
    # 理論上の秒ごとの勝率
    per_sec_winrate_dict = {}
    for i in per_sec_dict.keys():
        bet_cnt = len(per_sec_dict[i])
        if bet_cnt != 0:
            win_cnt = len(np.where(np.array(per_sec_dict[i]) >= 0)[0])
            per_sec_winrate_dict[i] = [win_cnt/bet_cnt, bet_cnt]

    output("理論上の秒毎の勝率悪い順(勝率,賭数):")
    worst_sorted = sorted(per_sec_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0],":", i[1][0], i[1][1])

    """


def showPipsPerSpread(spr_pred_pips_list):
    tmp_spr_list = spr_pred_pips_list[:, 0].astype(int)  # スプレッドだけ抽出
    spr_types = list(set(tmp_spr_list))  # スプレッドから重複をなくす
    spr_types.sort()  # 昇順ソート

    up_ind = np.where(spr_pred_pips_list[:, 1] == "BUY")[0]
    tmp_up_list = spr_pred_pips_list[up_ind]

    dw_ind = np.where(spr_pred_pips_list[:, 1] == "SELL")[0]
    tmp_dw_list = spr_pred_pips_list[dw_ind]

    # スプレッドごとに勝率と利益を表示
    for spr_type in spr_types:
        target_ind = np.where(tmp_spr_list == spr_type)[0]
        tmp_pips_list = spr_pred_pips_list[target_ind][:, 2].astype(float)
        tmp_profit_list = spr_pred_pips_list[target_ind][:, 3].astype(float)

        tmp_pips_list_correct_num = len(np.where(tmp_pips_list >= 0)[0])

        output("SPRREAD:", spr_type)
        output("TOTAL", "BET CNT:", len(tmp_pips_list), "CORRECT CNT:", tmp_pips_list_correct_num,
               "ACC:", tmp_pips_list_correct_num / len(tmp_pips_list), "AVG_PIPS:", np.average(tmp_pips_list),
               "PROFIT:", np.sum(tmp_profit_list))

        if len(tmp_up_list) != 0:
            tmp_up_spr_list = tmp_up_list[:, 0].astype(int)
            target_ind = np.where(tmp_up_spr_list == spr_type)[0]
            if len(target_ind) != 0:
                tmp_pips_list = tmp_up_list[target_ind][:, 2].astype(float)
                tmp_profit_list = tmp_up_list[target_ind][:, 3].astype(float)
                tmp_pips_list_correct_num = len(np.where(tmp_pips_list >= 0)[0])
                output("UP", "BET CNT:", len(tmp_pips_list), "CORRECT CNT:", tmp_pips_list_correct_num,
                       "ACC:", tmp_pips_list_correct_num / len(tmp_pips_list), "AVG_PIPS:", np.average(tmp_pips_list),
                       "PROFIT:", np.sum(tmp_profit_list))

        if len(tmp_dw_list) != 0:
            tmp_dw_spr_list = tmp_dw_list[:, 0].astype(int)
            target_ind = np.where(tmp_dw_spr_list == spr_type)[0]
            if len(target_ind) != 0:
                tmp_pips_list = tmp_dw_list[target_ind][:, 2].astype(float)
                tmp_profit_list = tmp_dw_list[target_ind][:, 3].astype(float)
                tmp_pips_list_correct_num = len(np.where(tmp_pips_list >= 0)[0])
                output("DW", "BET CNT:", len(tmp_pips_list), "CORRECT CNT:", tmp_pips_list_correct_num,
                       "ACC:", tmp_pips_list_correct_num / len(tmp_pips_list), "AVG_PIPS:", np.average(tmp_pips_list),
                       "PROFIT:", np.sum(tmp_profit_list))


def showPipsPerPred(deal_hist):
    start = 0
    while True:
        end = get_decimal_add(start, 0.01)
        if end > 1:
            break

        total = [d.get("profit_pips") for d in deal_hist if d.get("pred") >= start and d.get("pred") < end]

        if len(total) != 0:
            output("Predict:", start)
            win_cnt = len(np.where(np.array(total) >= 0)[0])
            output("Total BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                   " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

            total = [d.get("profit_pips") for d in deal_hist if
                     d.get("pred") >= start and d.get("pred") < end and d.get("type") == "BUY"]
            if len(total) != 0:
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                output("BUY BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

            total = [d.get("profit_pips") for d in deal_hist if
                     d.get("pred") >= start and d.get("pred") < end and d.get("type") == "SELL"]
            if len(total) != 0:
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                output("SELL BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

        start = get_decimal_add(start, 0.01)


def showPipsPerDiv(deal_hist, show_profit_per_div_list):
    for div in show_profit_per_div_list:
        output("")
        output("Div sec" + str(div))
        start = -100
        while True:
            end = get_decimal_add(start, 1)
            if end > 100:
                break
            # print("start:", start)
            total = [d.get("profit_pips") for d in deal_hist if
                     d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end]
            # print("len:", len(total))
            if len(total) != 0:
                output("Div:", start)
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                output("Total BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

                total = [d.get("profit_pips") for d in deal_hist if
                         d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end and d.get("type") == "BUY"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    output("BUY BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                           " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

                total = [d.get("profit_pips") for d in deal_hist if
                         d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end and d.get("type") == "SELL"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    output("SELL BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                           " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), )

            start = get_decimal_add(start, 1)

def showHighProfitDeal(deal_hist_dict):
    output("利益が多い取引")

    host ='win6'
    db_no = 8
    db_name = 'USDJPY_LION_HISTORY'
    r = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

    limit_cnt = 100
    sorted_d = sorted(deal_hist_dict.items(), key=lambda x: x[1]["profit_pips"], reverse=True)
    cnt_t = 0
    for k, v in sorted_d:
        if cnt_t > limit_cnt:
            break

        score = k
        result = r.zrangebyscore(db_name, score, score, withscores=True)  # 全件取得
        if len(result) != 0:
            line = result[0]
            body = line[0]
            score = float(line[1])
            tmps = json.loads(body)

            output("stime:", v["stime"], "etime:", v["etime"], "sprice:", v["sprice"], "eprice:", v["eprice"],
                    "type:", v["type"], " profit_pips:", v["profit_pips"],
                   )
            output("stime:",datetime.fromtimestamp(tmps.get("order_score")), "etime:",datetime.fromtimestamp(tmps.get("deal_score")), "sprice:",tmps.get("start_rate"), "eprice:", tmps.get("end_rate"),
                   "type:", tmps.get("sign"), " profit:", tmps.get("profit"),
                   )
            output("")
        cnt_t += 1

def get_list_lstm(conf, start, end, ):
    dataSequence2 = DataSequence2(conf, start, end, True, False, )

    # 予想時のレート
    pred_close_list = np.array(dataSequence2.get_pred_close_list())

    # 全close値のリスト
    close_list = dataSequence2.get_close_list()
    # 全score値のリスト
    score_list = dataSequence2.get_score_list()
    # 全spread値のリスト
    spread_list = dataSequence2.get_spread_list()
    # 全tick値のリスト
    tick_list = dataSequence2.get_tick_list()

    # 予想対象のscore値のリスト
    target_score_list = np.array(dataSequence2.get_train_score_list())

    # 予想対象リスト 予想対象の場合-1が入っている
    train_list_index = np.array(dataSequence2.get_train_list_index())

    # 全atr値のリスト(Noneあり)
    atr_list = np.array(dataSequence2.get_atr_list())

    # 全jpy値のリスト
    jpy_list = np.array(dataSequence2.get_jpy_list())

    # 全ind値のリスト(Noneあり)
    ind_list = np.array(dataSequence2.get_ind_list())

    # 全OUTPUT_DATA値のリスト(Noneあり)
    output_dict = dataSequence2.get_output_dict()
    for tmp_k in c.OUTPUT_LIST:
        # np.arrayにいれなおし
        output_dict[tmp_k] = np.array(output_dict[tmp_k])

    # 予想対象のOUTPUT_DATAのリスト
    target_output_dict = {}
    for tmp_k in c.OUTPUT_LIST:
        target_output_dict[tmp_k] = output_dict[tmp_k][np.where(train_list_index != -1)[0]]

    return dataSequence2, pred_close_list, close_list, score_list, spread_list, tick_list, target_score_list, train_list_index, atr_list, jpy_list, ind_list, target_output_dict


def get_list_lgbm(conf, start, end, conf_lgbm_dict):
    # テストデータロード
    with open(conf_lgbm_dict["test_data_load_path"], 'rb') as f:
        test_lmd = pickle.load(f)

    # テストデータ作成時のconfをロード
    with open(conf_lgbm_dict["conf_load_path"], 'rb') as cf:
        test_conf = pickle.load(cf)

    df = test_lmd.get_x()

    x = df.loc[:, conf.INPUT_DATA]

    # 全close値のリスト
    close_list = test_lmd.get_close_list()
    # 全score値のリスト
    score_list = test_lmd.get_score_list()
    # 全spread値のリスト
    spread_list = test_lmd.get_spread_list()
    # 全tick値のリスト
    tick_list = test_lmd.get_tick_list()

    # 予想対象のscore値のリスト
    target_score_list = np.array(df.index)

    # 予想対象リスト 予想対象の場合-1が入っている
    train_list_index = test_lmd.get_train_list_index()

    # 全atr値のリスト(Noneあり)
    atr_list = test_lmd.get_atr_list()
    # 全jpy値のリスト
    jpy_list = test_lmd.get_jpy_list()
    # 全ind値のリスト(Noneあり)
    ind_list = test_lmd.get_ind_list()

    # 全OUTPUT_DATA値のリスト(Noneあり)
    output_dict_list = test_lmd.get_output_dict_list()
    for tmp_k in c.OUTPUT_LIST:
        # np.arrayにいれなおし
        output_dict_list[tmp_k] = np.array(output_dict_list[tmp_k])

    # 予想時のレート
    pred_close_list = close_list[np.where(train_list_index != -1)[0]]

    # 予想対象のOUTPUT_DATAのリスト
    target_output_dict = {}
    for tmp_k in c.OUTPUT_LIST:
        target_output_dict[tmp_k] = output_dict_list[tmp_k][np.where(train_list_index != -1)[0]]

    return test_lmd, test_conf, pred_close_list, close_list, score_list, spread_list, tick_list, target_score_list, train_list_index, atr_list, jpy_list, ind_list, target_output_dict


def get_reg_convert_predict_list(reg_conf, predict_list_ext, pred_close_list_ext, target_output_dict_ext):
    OUTPUT_MULTI = reg_conf["OUTPUT_MULTI"]
    OUTPUT_LIST = reg_conf["OUTPUT_LIST"]
    OUTPUT_TYPE = reg_conf["OUTPUT_TYPE"]
    OUTPUT_DATA_BEF_C = reg_conf["OUTPUT_DATA_BEF_C"]

    if OUTPUT_TYPE == "d":
        # 現実のレートに換算する
        tmp_list = []
        if OUTPUT_DATA_BEF_C == True:
            for t_c, t_p in zip(pred_close_list_ext, predict_list_ext):
                if isinstance(t_p, list):
                    tmp_list.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                else:
                    tmp_list.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))
        else:
            if len(OUTPUT_LIST) == 1:
                for j, tmp_k in enumerate(OUTPUT_LIST):
                    for t_c, t_p in zip(target_output_dict_ext[tmp_k], predict_list_ext):
                        if isinstance(t_p, list):
                            tmp_list.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                        else:
                            tmp_list.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))

            else:
                for j, tmp_k in enumerate(OUTPUT_LIST):
                    t_l = []
                    for t_c, t_p in zip(target_output_dict_ext[tmp_k], predict_list_ext[:, j]):
                        if isinstance(t_p, list):
                            t_l.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                        else:
                            t_l.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))

                    tmp_list.append(t_l)

        predict_list = np.array(tmp_list)

    elif OUTPUT_TYPE == "sub":
        # 現実のレートに換算する
        tmp_list = []
        for t_p in predict_list_ext:
            if isinstance(t_p, list):
                tmp_list.append(get_decimal_divide(t_p[0], OUTPUT_MULTI))
            else:
                tmp_list.append(get_decimal_divide(t_p, OUTPUT_MULTI))
        predict_list = np.array(tmp_list)

    return predict_list


def get_score_pred_dict_ext_lstm(conf, conf_lstm_dict, dataSequence2):
    FILE_PREFIX_EXT = conf_lstm_dict["FILE_PREFIX_EXT"]

    output("FILE_PREFIX_EXT:", FILE_PREFIX_EXT)

    if conf_lstm_dict["USE_DATASEQ_EXT"] == True:
        # ext用のconf設定
        conf_ext = copy.deepcopy(conf)
        conf_ext.change_db_terms(db1_term=2, db2_term=10, db3_term=60, db4_term=0, db5_term=0, )
        conf_ext.INPUT_LEN = [300, 300, 120, ]
        conf_ext.METHOD = "LSTM7"
        conf_ext.LEARNING_TYPE = "CATEGORY_BIN_BOTH"
        conf_ext.INPUT_DATAS = ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ]
        dataSequence2_ext = DataSequence2(conf_ext, start, end, True, False, )
    else:
        dataSequence2_ext = dataSequence2

    if type(FILE_PREFIX_EXT) == list:
        # FILE_PREFIX_EXTがリストならCATEGORY_BIN_BOTH
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        # ndarrayで返って来る
        predict_list_ext_up = get_predict(FILE_PREFIX_EXT[0], dataSequence2_ext)

        predict_list_ext_dw = get_predict(FILE_PREFIX_EXT[1], dataSequence2_ext)
        # SAMEの予想結果は0とする
        predict_list_ext_zero = np.zeros((len(predict_list_ext_up), 2))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_ext_up, predict_list_ext_zero, predict_list_ext_dw], 1)
        predict_list_ext = all[:, [0, 2, 4]]

    else:
        predict_list_ext = get_predict(FILE_PREFIX_EXT, dataSequence2_ext)

    # 現実レートに変換する
    if conf["LEARNING_TYPE"] == "REGRESSION":
        reg_conf = conf["REG_CONF"]
        pred_close_list_ext = np.array(dataSequence2_ext.get_pred_close_list())
        output_dict_ext = dataSequence2_ext.get_output_dict()
        train_list_index_ext = np.array(dataSequence2_ext.get_train_list_index())

        OUTPUT_LIST = reg_conf["OUTPUT_LIST"]

        for tmp_k in OUTPUT_LIST:
            # np.arrayにいれなおし
            output_dict_ext[tmp_k] = np.array(output_dict_ext[tmp_k])

        # 予想対象のOUTPUT_DATAのリスト
        target_output_dict_ext = {}
        for tmp_k in OUTPUT_LIST:
            target_output_dict_ext[tmp_k] = output_dict_ext[tmp_k][np.where(train_list_index_ext != -1)[0]]

        # 現実レートに変換する
        predict_list_ext = get_reg_convert_predict_list(reg_conf, predict_list_ext, pred_close_list_ext,
                                                        target_output_dict_ext)

    target_score_list_ext = np.array(dataSequence2_ext.get_train_score_list())

    # 予想結果と予想時のスコアを辞書で保持
    if len(predict_list_ext) != len(target_score_list_ext):
        print("length of predict_list_ext and length of target_score_list_ext are not same:", len(predict_list_ext),
              len(target_score_list_ext))
        exit(1)

    score_pred_dict_ext = dict(zip(target_score_list_ext, predict_list_ext))

    return score_pred_dict_ext


def get_score_pred_dict_ext_lgbm(conf, conf_lgbm_dict, test_lmd, test_conf, ):
    FILE_PREFIX_EXT = conf_lgbm_dict["FILE_PREFIX_EXT"]
    FILE_PREFIX_EXT_SUFFIX = conf_lgbm_dict["FILE_PREFIX_EXT_SUFFIX"]

    if conf_lgbm_dict["USE_DATASEQ_EXT"] == True:
        # テストデータロード
        with open(conf_lgbm_dict["test_data_load_path_ext"], 'rb') as f:
            test_lmd_ext = pickle.load(f)

        # テストデータ作成時のconfをロード
        with open(conf_lgbm_dict["conf_load_path_ext"], 'rb') as cf:
            test_conf_ext = pickle.load(cf)

        df = test_lmd_ext.get_x()
        x = df.loc[:, conf_lgbm_dict["INPUT_DATA"]]
    else:
        test_lmd_ext = test_lmd
        test_conf_ext = test_conf

        df = test_lmd_ext.get_x()
        x = df.loc[:, conf.INPUT_DATA]

    if type(FILE_PREFIX_EXT) == list:
        # FILE_PREFIX_EXTがリストならCATEGORY_BIN_BOTH
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        bst_up = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX_EXT[0])
        bst_dw = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX_EXT[1])

        # ndarrayで返って来る
        predict_list_up = bst_up.predict(x, num_iteration=int(FILE_PREFIX_EXT_SUFFIX[0]))
        predict_list_dw = bst_dw.predict(x, num_iteration=int(FILE_PREFIX_EXT_SUFFIX[1]))

        # SAMEの予想結果は0とする
        predict_list_zero = np.zeros((len(predict_list_up), 1))

        predict_list_up = predict_list_up.reshape(len(predict_list_up), -1)  # 一次元から二次元配列に反感
        predict_list_dw = predict_list_dw.reshape(len(predict_list_dw), -1)  # 一次元から二次元配列に反感

        # UP,SAME,DWの予想結果を合算する
        predict_list_ext = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
    else:
        bst = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX_EXT)

        # 予想取得
        predict_list_ext = bst.predict(x, num_iteration=int(FILE_PREFIX_EXT_SUFFIX))

    # 現実レートに変換する
    if conf_lgbm_dict["LEARNING_TYPE"] == "REGRESSION":
        reg_conf = conf_lgbm_dict["REG_CONF"]

        # 全close値のリスト
        close_list_ext = test_lmd.get_close_list()
        # 予想対象リスト 予想対象の場合-1が入っている
        train_list_index_ext = test_lmd.get_train_list_index()

        OUTPUT_LIST = reg_conf["OUTPUT_LIST"]

        # 全OUTPUT_DATA値のリスト(Noneあり)
        output_dict_list = test_lmd.get_output_dict_list()
        for tmp_k in OUTPUT_LIST:
            # np.arrayにいれなおし
            output_dict_list[tmp_k] = np.array(output_dict_list[tmp_k])

        # 予想対象のOUTPUT_DATAのリスト
        target_output_dict_ext = {}
        for tmp_k in OUTPUT_LIST:
            target_output_dict_ext[tmp_k] = output_dict_list[tmp_k][np.where(train_list_index_ext != -1)[0]]

        pred_close_list_ext = close_list_ext[np.where(train_list_index_ext != -1)[0]]

        # 現実レートに変換する
        predict_list_ext = get_reg_convert_predict_list(reg_conf, predict_list_ext, pred_close_list_ext,
                                                        target_output_dict_ext)

    # 予想対象のscore値のリスト
    target_score_list_ext = np.array(df.index)
    # 予想結果と予想時のスコアを辞書で保持
    if len(predict_list_ext) != len(target_score_list_ext):
        print("length of predict_list_ext and length of target_score_list_ext are not same:", len(predict_list_ext),
              len(target_score_list_ext))
        exit(1)

    score_pred_dict_ext = dict(zip(target_score_list_ext, predict_list_ext))

    return score_pred_dict_ext


def get_predict_list_lstm(conf, FILE_PREFIX, suffix, dataSequence2, pred_close_list, target_output_dict):
    # print(suffix, border_list, ext_border)
    if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        # ndarrayで返って来る

        predict_list_up = get_predict(FILE_PREFIX[0] + "-" + suffix[0], dataSequence2)
        predict_list_dw = get_predict(FILE_PREFIX[1] + "-" + suffix[1], dataSequence2)

        # SAMEの予想結果は0とする
        predict_list_zero = np.zeros((len(predict_list_up), 2))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
        predict_list = all[:, [0, 2, 4]]

    else:
        predict_list = get_predict(FILE_PREFIX + "-" + suffix, dataSequence2)

    if conf.LEARNING_TYPE == "REGRESSION":
        # 現実レートに変換する
        reg_conf = {
            "OUTPUT_DATA_BEF_C": conf.OUTPUT_DATA_BEF_C,
            "OUTPUT_TYPE": conf.OUTPUT_TYPE,
            "OUTPUT_MULTI": conf.OUTPUT_MULTI,
            "OUTPUT_LIST": conf.OUTPUT_LIST,
        }
        predict_list = get_reg_convert_predict_list(reg_conf, predict_list, pred_close_list, target_output_dict)

    return predict_list


def get_predict_list_lgbm(conf, FILE_PREFIX, suffix, test_lmd, pred_close_list, target_output_dict, ):
    df = test_lmd.get_x()
    x = df.loc[:, conf.INPUT_DATA]

    if type(FILE_PREFIX) == list:
        # FILE_PREFIXがリストならCATEGORY_BIN_BOTH
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        # CATEGORY_BIN_UP とDWで予想した結果を合わせる
        bst_up = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX[0])
        bst_dw = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX[1])

        # ndarrayで返って来る
        predict_list_up = bst_up.predict(x, num_iteration=int(suffix[0]))
        predict_list_dw = bst_dw.predict(x, num_iteration=int(suffix[1]))

        # SAMEの予想結果は0とする
        predict_list_zero = np.zeros((len(predict_list_up), 1))

        predict_list_up = predict_list_up.reshape(len(predict_list_up), -1)  # 一次元から二次元配列に反感
        predict_list_dw = predict_list_dw.reshape(len(predict_list_dw), -1)  # 一次元から二次元配列に反感

        # UP,SAME,DWの予想結果を合算する
        predict_list = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
    else:
        bst = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX)

        # 予想取得
        predict_list = bst.predict(x, num_iteration=int(suffix))

    if conf.LEARNING_TYPE == "REGRESSION":
        # 現実レートに変換する
        reg_conf = {
            "OUTPUT_DATA_BEF_C": conf.OUTPUT_DATA_BEF_C,
            "OUTPUT_TYPE": conf.OUTPUT_TYPE,
            "OUTPUT_MULTI": conf.OUTPUT_MULTI,
            "OUTPUT_LIST": conf.OUTPUT_LIST,
        }
        predict_list = get_reg_convert_predict_list(reg_conf, predict_list, pred_close_list, target_output_dict)

    return predict_list


def do_predict(conf, start, end, spread_conf):
    start_min_spread, start_max_spread, end_min_spread, end_max_spread, cannot_deal_cnt_max = spread_conf
    global c
    c = conf

    stoploss_per_sec_dict = {}
    if c.FX_STOPLOSS_PER_SEC_FLG:
        r = redis.Redis(host='localhost', port=6379, db=c.FX_STOPLOSS_PER_SEC_DB_NO, decode_responses=True)

        for db in c.FX_STOPLOSS_PER_SEC_DB_LIST:
            tmp_db_name = c.FX_STOPLOSS_PER_SEC_DB_PREFIX + str(db)
            tmp_dict = {}

            result = r.zrangebyscore(tmp_db_name, -0.1, 1.0, withscores=True)

            for j, line in enumerate(result):
                body = line[0]
                score = float(line[1])
                tmps = json.loads(body)
                tmp_dict[score] = tmps["avg_pips"]

            stoploss_per_sec_dict[db] = tmp_dict

    # FILE_PREFIX = "USDJPY_LT1_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1-M1_OT-d_OD-c_IDL1_BS5120_SD0_SHU1_EL20-21-22_ub1_MN556"
    # FILE_PREFIX = "EURUSD_LT1_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1-M1_OT-d_OD-c_IDL1_BS5120_SD0_SHU1_EL20-21-22_ub1_MN580"
    FILE_PREFIX = "MN808"

    """
    # CATEGORY_BIN_BOTHの場合はupとdwのモデルをリストにする
    FILE_PREFIX = [
        "USDJPY_LT3_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN194",
        "USDJPY_LT4_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN196",
      ]

    # 延長判定用モデル:CATEGORY_BIN_BOTHの場合はupとdwのモデルをリストにする
    FILE_PREFIX_EXT = [
        "EURUSD_LT3_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN202",
        "EURUSD_LT4_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN203",
   ]
    """

    # [suffix,[borderのlist],ext_borderの値, border上限(設定なければ""),]

    model_suffix = [
        ["352", [0.6, ], 0.05, "", ],


    ]

    # 特定のsuffixとborder_list, ext_borderを組みでテストする
    # model_suffix = [ ["32", [ 0.52, 0.54, 0.56, 0.58 ], 0.5],] #category_bin_both 以外
    """
    #category_bin_both 用
    model_suffix = [
        [["10", "31"], [[0.53, 0.53], ], [0.53, 0.53]], 
    ]
    """

    # True:延長判定用モデルを使用する
    USE_EXT = False

    # lstm用の設定
    conf_lstm_dict = {
        # 延長判定用モデル suffixまで含める USE_EXT=Trueの場合設定
        "FILE_PREFIX_EXT": "",

        # 延長判定用モデルのLEARNING_TYPE
        "LEARNING_TYPE": "REGRESSION",
        # regressionモデルの場合に予想を現実のレートに換算するための設定
        "REG_CONF": {
            "OUTPUT_DATA_BEF_C": False,
            "OUTPUT_TYPE": "d",
            "OUTPUT_MULTI": 1,
            "OUTPUT_LIST": "c",
        },

        # ext独自用のDataseqを使う場合:True Dataseqの設定自体はget_score_pred_dict_ext_lstm()内で記述する
        "USE_DATASEQ_EXT": False,

    }

    # lgbm用の設定
    conf_lgbm_dict = {
        # 予想元テストデータファイル
        "test_data_load_path": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF306.pickle",
        "conf_load_path": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF306-conf.pickle",

        # 延長判定用モデル USE_EXT=Trueの場合設定 category_bin_bothの場合リストにする
        "FILE_PREFIX_EXT": "MN920",
        # 延長判定用モデルのsuffix category_bin_bothの場合リストにする
        "FILE_PREFIX_EXT_SUFFIX": "3697",

        # 延長判定用モデルのINPUT_DATA
        # "INPUT_DATA": [],
        "INPUT_DATA": "714-36-DW@714-36-DW-12@714-36-DW-4@714-36-DW-8@714-36-SAME@714-36-SAME-12@714-36-SAME-4@714-36-SAME-8@714-36-UP@714-36-UP-12@714-36-UP-4@714-36-UP-8@715-40-DW@715-40-DW-12@715-40-DW-4@715-40-DW-8@715-40-SAME@715-40-SAME-12@715-40-SAME-4@715-40-SAME-8@715-40-UP@715-40-UP-12@715-40-UP-4@715-40-UP-8@885-6-REG@885-6-REG-12@885-6-REG-4@885-6-REG-8@887-39-DW@887-39-DW-12@887-39-DW-4@887-39-DW-8@887-39-SAME@887-39-SAME-12@887-39-SAME-4@887-39-SAME-8@887-39-UP@887-39-UP-12@887-39-UP-4@887-39-UP-8".split(
            "@"),
        # 延長判定用モデルのLEARNING_TYPE
        "LEARNING_TYPE": "CATEGORY",
        # regressionモデルの場合に予想を現実のレートに換算するための設定
        "REG_CONF": {
            "OUTPUT_DATA_BEF_C": False,
            "OUTPUT_TYPE": "d",
            "OUTPUT_MULTI": 1,
            "OUTPUT_LIST": "c",
        },

        # ext独自用の予想元テストデータファイルを使う場合:True
        "USE_DATASEQ_EXT": True,
        # 延長用予想元テストデータファイル
        "test_data_load_path_ext": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF299.pickle",
        "conf_load_path_ext": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF299-conf.pickle",
    }

    # close値やスコアのリストを取得
    if conf.CONF_TYPE == "LSTM":
        dataSequence2, pred_close_list, close_list, score_list, spread_list, tick_list, \
        target_score_list, train_list_index, atr_list, jpy_list, ind_list, \
        target_output_dict = get_list_lstm(conf, start, end, )

    elif conf.CONF_TYPE == "LGBM":
        test_lmd, test_conf, pred_close_list, close_list, score_list, spread_list, tick_list, \
        target_score_list, train_list_index, atr_list, jpy_list, ind_list, \
        target_output_dict = get_list_lgbm(conf, start, end, conf_lgbm_dict)

    # 長さチェック
    if len(score_list) != len(close_list) or len(score_list) != len(train_list_index) or len(score_list) != len(
            spread_list) or \
            len(score_list) != len(tick_list) or len(score_list) != len(atr_list) or len(score_list) != len(
        jpy_list) or len(jpy_list) != len(ind_list):
        print("list length is wrong!!!", len(score_list), len(close_list), len(train_list_index), len(spread_list),
              len(tick_list), len(atr_list), len(jpy_list), len(ind_list))
        exit(1)

    # market:成行き limit:指値
    mode = "market"

    # suffixを変えて検証する対象 d(regression) or sub or category or category_bin or category_bin_both 検証対象外は固定のモデルを使用する
    target = "category"
    target_ext = "category"
    # target = "d"
    # target_ext = "d"

    output(start, end)
    if conf.CONF_TYPE == "LGBM":
        output("test_data_load_path:", conf_lgbm_dict["test_data_load_path"])
    output("TERM:", c.TERM)
    output("START_SEC:", c.START_TERM * c.BET_TERM)
    output("END_SEC:", c.END_TERM * c.BET_TERM)

    # ベット延長するか判断するTERM
    ext_term = 2
    output("ext_term", ext_term)

    # ポジション数がこの数以下の場合に延長判断するtermをext_term_shortにする
    ext_term_short_position_num = None  # None:設定なし
    # ext_term_short_position_num = 6

    # ポジション数がext_term_short_position_num以下の場合に延長判断するterm
    ext_term_short = 2

    if ext_term_short_position_num != None:
        output("ext_term_short_position_num", ext_term_short_position_num)
        output("ext_term_short", ext_term_short)

    # ベット延長判断開始秒数(ベットしてから何秒経過すれば延長判断するか)
    # 決済期間が30秒でも30未満を設定する場合は損切り判定の役割となる
    # 基本的に予想期間を設定
    ext_start_sec = 122
    output("ext_start_sec", ext_start_sec)

    ext_start_sec_short_position_num = None  # None:設定なし
    # ext_start_sec_short_position_num = 6

    # ポジション数がext_term_short_position_num以下の場合に延長判断するterm
    ext_start_sec_short = 2

    if ext_start_sec_short_position_num != None:
        output("ext_start_sec_short_position_num", ext_start_sec_short_position_num)
        output("ext_start_sec_short", ext_start_sec_short)

    # 急激なレート変動の場合取引しない
    max_div = None  # None:設定なし
    # max_div = 10
    max_div_sec = 60

    if max_div != None:
        output("max_div:", max_div)
        output("max_div_sec:", max_div_sec)

    # 全建玉の許容最大損失を下回ったら全建玉を決済する
    TOTAL_STOPLOSS = None  # None:設定なし
    # TOTAL_STOPLOSS = -0.001
    output("TOTAL_STOPLOSS", TOTAL_STOPLOSS)

    # 全建玉の許容最大損失を下回ったら新規発注しない
    ORDER_TOTAL_STOPLOSS = None  # None:設定なし
    # ORDER_TOTAL_STOPLOSS = -0.7
    output("ORDER_TOTAL_STOPLOSS", ORDER_TOTAL_STOPLOSS)

    # 延長時に前回予想より下がったら決済する
    OVER_BEF_PREDICT = False
    output("OVER_BEF_PREDICT", OVER_BEF_PREDICT)

    # 損切りするポジションがひとつでもあれば全部決済する
    ALL_DEAL_FLG = False
    output("ALL_DEAL_FLG", ALL_DEAL_FLG)

    # 逆張りのみ行う
    AGAINST_FLG = False
    output("AGAINST_FLG", AGAINST_FLG)

    AGAINST_SEC = 60  # 基準となるDIV算出秒
    AGAINST_DIV = 3  # 基準となるDIV
    if AGAINST_FLG:
        output("AGAINST_SEC", AGAINST_SEC)
        output("AGAINST_DIV", AGAINST_DIV)

    atr_range_base = []  # 絞り込むATR幅
    atr_range_ext = []  # 絞り込むATR幅(延長判断用)

    ind_range_bases = [

    ]
    ind_range_exts = [

    ]

    TK_SL_TYPE = "pips"  # どのようにtakeprofit,stoplossを決めるか satr or atr or pred or pips
    if target in ["category", "category_bin", "category_bin_both"] and TK_SL_TYPE == "pred":
        # categoryの場合はpredを設定できない
        output("cannot set TK_SL_TYPE on pred")
        exit(1)

    show_detail = True  # 詳細表示
    show_history = False  # 取引履歴を表示
    show_plot = True
    show_profit_atr = False
    show_profit_ind = False
    show_profit_time = False
    show_position = False
    show_profit_per_spread = False
    show_profit_per_pred = False
    show_profit_per_div = False
    show_profit_per_div_list = [10, 60, 180]
    show_high_profit_deal = False

    # 重要指標の時間帯を除外してテストする
    # important_index_list = []
    # important_index_list = ["雇用統計", "CPI", "ISM製造業景況指数","GDP", "ADP雇用統計", "ISM非製造業景況指数", "小売売上高", "新築住宅販売件数", "個人消費支出", "FOMC金利発表", "日銀政策金利発表","日銀記者会見"]
    important_index_list = ["雇用統計", "CPI", "ISM製造業景況指数", "GDP", "ADP雇用統計", "ISM非製造業景況指数", "小売売上高", "新築住宅販売件数", "個人消費支出",
                            "FOMC金利発表", "日銀政策金利発表", ]
    important_index_range = 300 #除外する前後の時間秒
    #important_index_range = 60  # 除外する前後の時間秒

    result_per_suffix_border = {}

    if show_plot:
        # png保存用のディレクトリ作成
        save_dir = png_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
        makedirs(save_dir)
        output("PNG SAVE DIR:", save_dir)

    output("FX_TAKE_PROFIT_FLG:", c.FX_TAKE_PROFIT_FLG)
    output("FX_STOP_LOSS_FLG:", c.FX_STOP_LOSS_FLG)

    if c.FX_TAKE_PROFIT_FLG or c.FX_STOP_LOSS_FLG:
        output("TP_SL_MODE:", c.TP_SL_MODE)
        if c.TP_SL_MODE == "manual":
            output("TP_SL_MANUAL_TERM:", c.TP_SL_MANUAL_TERM)

        output("TK_SL_TYPE:", TK_SL_TYPE)
        if TK_SL_TYPE == "atr" or TK_SL_TYPE == "satr":
            if c.FX_TAKE_PROFIT_FLG:
                output("FX_TP_ATR:", c.FX_TP_ATR)
            if c.FX_STOP_LOSS_FLG:
                output("FX_SL_ATR:", c.FX_SL_ATR)

        elif TK_SL_TYPE == "pred":
            if c.FX_TAKE_PROFIT_FLG:
                output("FX_TP_PRED:", c.FX_TP_PRED)
            if c.FX_STOP_LOSS_FLG:
                output("FX_SL_PRED:", c.FX_SL_PRED)

        elif TK_SL_TYPE == "pips":
            if c.FX_TAKE_PROFIT_FLG:
                output("FX_TAKE_PROFIT:", c.FX_TAKE_PROFIT)
            if c.FX_STOP_LOSS_FLG:
                output("FX_STOP_LOSS:", c.FX_STOP_LOSS)

    if c.FX_STOP_LOSS_FLG:
        output("CHANGE_STOPLOSS_FLG:", c.CHANGE_STOPLOSS_FLG)
        if c.CHANGE_STOPLOSS_FLG:
            output("CHANGE_STOPLOGG_TERM:", c.CHANGE_STOPLOGG_TERM)

    output("IND_COLS:", c.IND_COLS)
    output("ind_range_bases:", ind_range_bases)
    output("ind_range_exts:", ind_range_exts)

    output("ATR_COL:", c.ATR_COL)
    output("atr_range_base:", atr_range_base)
    output("atr_range_ext:", atr_range_ext)

    output("start_min_spread:", start_min_spread)
    output("start_max_spread:", start_max_spread)
    output("end_min_spread:", end_min_spread)
    output("end_max_spread:", end_max_spread)
    output("cannot_deal_cnt_max:", cannot_deal_cnt_max)

    output("BUY_FLG:", c.BUY_FLG)
    output("SELL_FLG:", c.SELL_FLG)
    output("FX_SINGLE_FLG:", c.FX_SINGLE_FLG)
    output("TRADE_SHIFT:", conf.TRADE_SHIFT)
    output("FX_NOT_EXT_FLG:", c.FX_NOT_EXT_FLG)
    output("START_MONEY:", c.START_MONEY)
    output("FX_FUND:", c.FX_FUND)
    output("FX_LEVERAGE:", c.FX_LEVERAGE)
    output("FX_FIX_POSITION:", c.FX_FIX_POSITION)
    if conf.SYMBOL == "BTCUSD":
        output("BTCUSD_SPREAD_PERCENT:", conf.BTCUSD_SPREAD_PERCENT)
    else:
        output("ADJUST_PIPS:", conf.ADJUST_PIPS)
    output("FX_BORDER_ATR:", c.FX_BORDER_ATR)
    output("FX_MIN_TP_SL:", c.FX_MIN_TP_SL)
    output("FX_NOT_EXT_MINUS:", c.FX_NOT_EXT_MINUS)
    output("FX_MAX_TRADE_SEC:", c.FX_MAX_TRADE_SEC)

    output("EXCEPT_LIST_SEC_TEST:", conf.EXCEPT_LIST_SEC_TEST)
    output("EXCEPT_LIST_HOUR_TEST:", conf.EXCEPT_LIST_HOUR_TEST)

    output("RESTRICT_FLG:", c.RESTRICT_FLG)
    if c.RESTRICT_FLG:
        output("RESTRICT_SEC:", c.RESTRICT_SEC)
    output("FX_MAX_POSITION_CNT:", c.FX_MAX_POSITION_CNT)

    output("FX_STOPLOSS_PER_SEC_FLG:", c.FX_STOPLOSS_PER_SEC_FLG)
    if c.FX_STOPLOSS_PER_SEC_FLG:
        output("FX_STOPLOSS_PER_SEC:", c.FX_STOPLOSS_PER_SEC)
        output("FX_STOPLOSS_PER_SEC_DB_NO:", c.FX_STOPLOSS_PER_SEC_DB_NO)
        output("FX_STOPLOSS_PER_SEC_DB_LIST:", c.FX_STOPLOSS_PER_SEC_DB_LIST)
        output("FX_STOPLOSS_PER_SEC_DB_PREFIX:", c.FX_STOPLOSS_PER_SEC_DB_PREFIX)
        output("FX_STOPLOSS_PER_SEC_CHK_TIMES:", c.FX_STOPLOSS_PER_SEC_CHK_TIMES)

    output("SAME_SHIFT_NG_FLG:", c.SAME_SHIFT_NG_FLG)
    if c.SAME_SHIFT_NG_FLG:
        output("NG_SHIFT:", c.NG_SHIFT)

    output("important_index_list:", important_index_list)
    if len(important_index_list) != 0:
        output("important_index_range:", important_index_range)
    importantAnswer = ImportantIndex(index_set=important_index_list, range=important_index_range)

    output("USE_EXT:", USE_EXT)

    # 延長判定用予想取得
    if USE_EXT:
        if conf.CONF_TYPE == "LSTM":
            output("FILE_PREFIX_EXT:", conf_lstm_dict["FILE_PREFIX_EXT"])
            output("LEARNING_TYPE:", conf_lstm_dict["LEARNING_TYPE"])
            if conf_lstm_dict["LEARNING_TYPE"] == "REGRESSION":
                output("REG_CONF:", conf_lstm_dict["REG_CONF"])

            output("USE_DATASEQ_EXT:", conf_lstm_dict["USE_DATASEQ_EXT"])

            score_pred_dict_ext = get_score_pred_dict_ext_lstm(conf, conf_lstm_dict, dataSequence2)

        elif conf.CONF_TYPE == "LGBM":
            output("FILE_PREFIX_EXT:", conf_lgbm_dict["FILE_PREFIX_EXT"])
            output("FILE_PREFIX_EXT_SUFFIX:", conf_lgbm_dict["FILE_PREFIX_EXT_SUFFIX"])

            output("LEARNING_TYPE:", conf_lgbm_dict["LEARNING_TYPE"])
            if conf_lgbm_dict["LEARNING_TYPE"] == "REGRESSION":
                output("REG_CONF:", conf_lgbm_dict["REG_CONF"])

            output("USE_DATASEQ_EXT:", conf_lgbm_dict["USE_DATASEQ_EXT"])
            output("test_data_load_path_ext:", conf_lgbm_dict["test_data_load_path_ext"])
            output("conf_load_path_ext:", conf_lgbm_dict["conf_load_path_ext"])

            score_pred_dict_ext = get_score_pred_dict_ext_lgbm(conf, conf_lgbm_dict, test_lmd, test_conf, )

    # 同じ予想結果を使いまわすための変数
    prev_suffix = None
    prev_list = None

    for suffix in model_suffix:
        suffix, border_list, ext_border, border_ceil = suffix
        if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
            if prev_suffix[0] == suffix[0] and prev_suffix[1] == suffix[1]:
                # 予想を使いまわす
                predict_list = prev_list
            else:
                prev_suffix = suffix
                if conf.CONF_TYPE == "LSTM":
                    predict_list = get_predict_list_lstm(conf, FILE_PREFIX, suffix, dataSequence2, pred_close_list,
                                                         target_output_dict)
                elif conf.CONF_TYPE == "LGBM":
                    predict_list = get_predict_list_lgbm(conf, FILE_PREFIX, suffix, test_lmd, pred_close_list,
                                                         target_output_dict)
                prev_list = predict_list
        else:
            if prev_suffix == suffix:
                # 予想を使いまわす
                predict_list = prev_list
            else:
                prev_suffix = suffix
                if conf.CONF_TYPE == "LSTM":
                    predict_list = get_predict_list_lstm(conf, FILE_PREFIX, suffix, dataSequence2, pred_close_list,
                                                         target_output_dict)
                elif conf.CONF_TYPE == "LGBM":
                    predict_list = get_predict_list_lgbm(conf, FILE_PREFIX, suffix, test_lmd, pred_close_list,
                                                         target_output_dict)
                prev_list = predict_list

        if USE_EXT == False:
            # 延長判定用予想を作成
            predict_list_ext = predict_list
            if len(predict_list_ext) != len(target_score_list):
                print("length of predict_list_ext and length of target_score_list are not same:", len(predict_list_ext),
                      len(target_score_list))
                exit(1)
            score_pred_dict_ext = dict(zip(target_score_list, predict_list_ext))

        output("")
        output("suffix:", suffix)

        for border in border_list:
            # 予想結果表示用テキストを保持
            result_txt = []

            if ext_border == None:
                ext_border = border
            output("")
            output("border:", border)
            output("ext_border:", ext_border)
            output("border_ceil:", border_ceil)
            if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                filename = save_dir + "/" + "SUFFIX_" + suffix[0] + "-" + suffix[1] + "_BORDER_" + str(
                    border[0]) + "-" + str(border[1]) + "_EXTBORDER_" + str(ext_border[0]) + "-" + str(
                    ext_border[1]) + ".png"
            else:
                filename = save_dir + "/" + "SUFFIX_" + suffix + "_BORDER_" + str(border) + "_EXTBORDER_" + str(
                    ext_border) + "_BORDERCEIL_" + str(border_ceil) + ".png"
            max_drawdown = 0
            drawdown = 0
            max_drawdowns = []

            pips = []
            pips_tp = []
            pips_sl = []
            pips_sps = []

            atrs = []
            inds = []
            times = []

            tp_list = []
            sl_list = []
            limit_list = []

            money_y = []
            money_tmp = {}
            money = c.START_MONEY  # 始めの所持金

            position_num = 0
            position_num_tmp = {}  # 保持しているポジション数の推移

            fund_out_cnt = 0

            # bet情報を取引開始スコアをキーに保持
            # 指値注文はTERMの間だけ有効とする
            # deal(true,false):約定したかどうか。成り行きなら常にtrue、指値なら約定するまでfalse,
            # limit_price:指値,成り行きの場合はNoneを入れる
            # type:buy or sell,
            # price:bet時のレート,指値の場合は指値
            # stime:bet時の時間(score),指値の場合は注文を出した時間であり約定した時間ではない
            # len:betしている期間,
            # tp:takeprofit,
            # sl:stoploss
            # atr
            # spr:bet時のスプレッド

            bet_dicts = {}

            bet_len_dict = {}  # bet期間ごとの件数を保持

            j = 0
            bet_cnt = 0

            deal_hist = []  # 決済履歴を保持
            deal_hist_dict = {}
            prev_bet_start_score = None
            prev_bet_end_score = None

            spr_pred_pips_list = []  # スプレッド,予想, pips, 報酬を保持
            prev_sc = None

            ok_spread_cnt = 0
            ng_spread_cnt = 0
            no_pred_score = []
            pred_score = []

            for cnt, (sc, close, idx, spr, tick, atr, jpy, ind) in enumerate(
                    zip(score_list, close_list, train_list_index, spread_list, tick_list, atr_list, jpy_list,
                        ind_list)):

                # 取引時間外になったら抜ける
                tmp_now_dt = datetime.fromtimestamp(sc)
                if end < tmp_now_dt:
                    break

                j += 1

                # 今のレート
                now_price = close

                tick_spr_list = tick.split(",")  # tickとsprが:区切り
                ask_list = []
                bid_list = []
                for tick_spr in tick_spr_list:
                    tc, tcsp = tick_spr.split(":")
                    tc = float(tc)
                    tcsp = int(tcsp)
                    tmp_ask, tmp_bid = get_ask_bid(tc, tcsp, c.PIPS)
                    ask_list.append(tmp_ask)
                    bid_list.append(tmp_bid)

                now_ask, now_bid = get_ask_bid(close, spr, c.PIPS)
                order_ask = now_ask
                order_bid = now_bid
                deal_ask = now_ask
                deal_bid = now_bid

                if c.START_TERM != 0:
                    try:
                        tmp_close = close_list[cnt + c.START_TERM]
                        tmp_spr = spread_list[cnt + c.START_TERM]
                        order_ask, order_bid = get_ask_bid(tmp_close, tmp_spr, c.PIPS)
                    except Exception as e:
                        # 該当するclose,spreadがない場合は仕方ないのでそのままとする
                        pass
                if c.END_TERM != 0:
                    try:
                        tmp_close = close_list[cnt + c.END_TERM]
                        tmp_spr = spread_list[cnt + c.END_TERM]
                        deal_ask, deal_bid = get_ask_bid(tmp_close, tmp_spr, c.PIPS)
                    except Exception as e:
                        # 該当するclose,spreadがない場合は仕方ないのでそのままとする
                        pass

                all_deal_flg = False

                # ポジションがあり、予想がない場合はその日の最後の予想、もしくはデータが途切れた場合なので決済する
                # if idx == -1 or (c.FX_BORDER_ATR != None and c.FX_BORDER_ATR <= atr):  # 予想がない場合 or ATRが突然上がった場合
                if (c.FX_BORDER_ATR != None and atr != None and c.FX_BORDER_ATR <= atr):  # ATRが突然上がった場合
                    all_deal_flg = True

                if prev_sc != None and get_decimal_sub(sc, prev_sc) > c.BET_TERM:
                    # データが続いていなければ全て決済する
                    all_deal_flg = True
                prev_sc = sc

                total_profit_pips = 0  # 全建玉の利益

                # 今保持している建玉すべてでストップロス確認
                if c.FX_TAKE_PROFIT_FLG or c.FX_STOP_LOSS_FLG or c.FX_STOPLOSS_PER_SEC_FLG or TOTAL_STOPLOSS != None or ORDER_TOTAL_STOPLOSS != None:
                    del_key = []
                    for dict_key in bet_dicts.keys():
                        take_profit_flg = False
                        stop_loss_flg = False
                        bet_dict = copy.deepcopy(bet_dicts[dict_key])
                        stop_price = 0
                        tmp_profit = 0

                        if bet_dict["deal"] != True:
                            # 約定してない場合　このtermで約定するか確認　またそのままtk,slするかも確認
                            for tick_bid, tick_ask in zip(bid_list, ask_list):
                                if bet_dict["type"] == "BUY":
                                    if bet_dicts[dict_key]["deal"] == False and tick_ask <= bet_dict["limit_price"]:
                                        bet_dicts[dict_key]["deal"] = True
                                        if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                            bet_dicts[dict_key]["sps_stime"] = sc
                                        continue

                                    if bet_dicts[dict_key]["deal"] == True:
                                        if c.FX_STOP_LOSS_FLG and tick_bid <= bet_dict["sl"]:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True

                                            stop_price = bet_dict["sl"]

                                            base_price = tick_bid
                                            if conf.SYMBOL == "BTCUSD":
                                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                        "100"))) + 0.1
                                                tmp_profit = base_price - bet_dict["price"] - btcusd_spr
                                            else:
                                                tmp_profit = base_price - bet_dict["price"] + c.ADJUST_PIPS
                                            if c.JPY_FLG == False:
                                                profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                            bet_dict["jpy"]) * jpy
                                            else:
                                                profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                            pips_sl.append(tmp_profit)
                                            break
                                        elif c.FX_TAKE_PROFIT_FLG and tick_bid >= bet_dict["tp"]:
                                            take_profit_flg = True
                                            stop_price = bet_dict["tp"]
                                            if conf.SYMBOL == "BTCUSD":
                                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                        "100"))) + 0.1
                                                tmp_profit = tick_bid - bet_dict["price"] - btcusd_spr
                                            else:
                                                tmp_profit = tick_bid - bet_dict["price"] + c.ADJUST_PIPS
                                            if c.JPY_FLG == False:
                                                profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                            bet_dict["jpy"]) * jpy
                                            else:
                                                profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                            pips_tp.append(tmp_profit)
                                            break
                                elif bet_dict["type"] == "SELL":
                                    if bet_dicts[dict_key]["deal"] == False and tick_bid >= bet_dict["limit_price"]:
                                        bet_dicts[dict_key]["deal"] = True
                                        if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                            bet_dicts[dict_key]["sps_stime"] = sc
                                        continue

                                    if bet_dicts[dict_key]["deal"] == True:
                                        if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= tick_ask:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True

                                            stop_price = bet_dict["sl"]
                                            base_price = tick_ask
                                            if conf.SYMBOL == "BTCUSD":
                                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                        "100"))) + 0.1
                                                tmp_profit = bet_dict["price"] - base_price - btcusd_spr
                                            else:
                                                tmp_profit = bet_dict["price"] - base_price + c.ADJUST_PIPS
                                            if c.JPY_FLG == False:
                                                profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                            bet_dict["jpy"]) * jpy
                                            else:
                                                profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                            pips_sl.append(tmp_profit)
                                            break
                                        elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= tick_ask:
                                            take_profit_flg = True
                                            stop_price = bet_dict["tp"]
                                            if conf.SYMBOL == "BTCUSD":
                                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                        "100"))) + 0.1
                                                tmp_profit = bet_dict["price"] - tick_ask - btcusd_spr
                                            else:
                                                tmp_profit = bet_dict["price"] - tick_ask + c.ADJUST_PIPS
                                            if c.JPY_FLG == False:
                                                profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                            bet_dict["jpy"]) * jpy
                                            else:
                                                profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                            pips_tp.append(tmp_profit)
                                            break
                        else:
                            # 約定している場合

                            # 現在の利益を計算
                            # TRADE_SHIFTが実際に取引するときのループ間隔なのでその間隔の時で利益を計算する
                            if TOTAL_STOPLOSS != None or ORDER_TOTAL_STOPLOSS != None:
                                if c.TRADE_SHIFT != None and get_decimal_mod(sc, c.TRADE_SHIFT) == 0:
                                    if bet_dict["type"] == "BUY":
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                    "100"))) + 0.1
                                            tmp_profit = now_bid - bet_dict["price"] - btcusd_spr
                                        else:
                                            tmp_profit = now_bid - bet_dict["price"] + conf.ADJUST_PIPS

                                    elif bet_dict["type"] == "SELL":
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                            tmp_profit = bet_dict["price"] - now_ask - btcusd_spr
                                        else:
                                            tmp_profit = bet_dict["price"] - now_ask + conf.ADJUST_PIPS

                                    total_profit_pips = total_profit_pips + tmp_profit

                            if bet_dict["type"] == "BUY":
                                if c.TP_SL_MODE == "auto":
                                    for tick_bid in bid_list:
                                        if c.FX_STOP_LOSS_FLG and tick_bid <= bet_dict["sl"]:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True

                                            stop_price = tick_bid
                                            break

                                        elif c.FX_TAKE_PROFIT_FLG and tick_bid >= bet_dict["tp"]:
                                            take_profit_flg = True
                                            stop_price = tick_bid
                                            break

                                elif c.TP_SL_MODE == "manual":
                                    passed_sc = get_decimal_sub(sc, bet_dict["stime"])
                                    if get_decimal_mod(passed_sc, c.TP_SL_MANUAL_TERM) == 0:
                                        if c.FX_STOP_LOSS_FLG and now_bid <= bet_dict["sl"]:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True

                                            stop_price = now_bid
                                        elif c.FX_TAKE_PROFIT_FLG and now_bid >= bet_dict["tp"]:
                                            take_profit_flg = True
                                            stop_price = now_bid

                                if stop_loss_flg or take_profit_flg:
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                        tmp_profit = stop_price - bet_dict["price"] - btcusd_spr
                                    else:
                                        tmp_profit = stop_price - bet_dict["price"] + c.ADJUST_PIPS

                                    if c.JPY_FLG == False:
                                        profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                    bet_dict["jpy"]) * jpy
                                    else:
                                        profit = tmp_profit * c.get_fx_position(bet_dict["price"])

                                    if stop_loss_flg:
                                        pips_sl.append(tmp_profit)
                                    elif take_profit_flg:
                                        pips_tp.append(tmp_profit)

                                elif c.FX_STOPLOSS_PER_SEC_FLG:
                                    # 一定秒数経過ごとに損切り判定する
                                    # ここに来たということはtickのstoploss,takeprofitに引っかからなかったということ
                                    if bet_dict["sps_stime"] != None:
                                        passed_time = get_decimal_sub(sc, bet_dict["sps_stime"])
                                        if passed_time in c.FX_STOPLOSS_PER_SEC_CHK_TIMES and passed_time in stoploss_per_sec_dict.keys():

                                            rate_change = float(Decimal(str(now_bid)) - Decimal(str(bet_dict["price"])))
                                            if rate_change < 0 and rate_change in stoploss_per_sec_dict[
                                                passed_time].keys() and \
                                                    stoploss_per_sec_dict[passed_time][rate_change] <= rate_change:

                                                # ベットした結果のPIPSが現在の損失(rate_change)よりひどくなりそうな場合は損切りする
                                                stop_loss_flg = True
                                                if ALL_DEAL_FLG:
                                                    all_deal_flg = True

                                                stop_price = now_bid

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                            "100"))) + 0.1
                                                    tmp_profit = get_sub(bet_dict["price"], now_bid) - btcusd_spr
                                                else:
                                                    tmp_profit = get_sub(bet_dict["price"], now_bid) + c.ADJUST_PIPS
                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                                bet_dict["jpy"]) * jpy
                                                else:
                                                    profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                                pips_sps.append(tmp_profit)

                            elif bet_dict["type"] == "SELL":
                                if c.TP_SL_MODE == "auto":
                                    for tick_ask in ask_list:
                                        if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= tick_ask:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True

                                            stop_price = tick_ask
                                            break

                                        elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= tick_ask:
                                            take_profit_flg = True
                                            stop_price = tick_ask
                                            break

                                elif c.TP_SL_MODE == "manual":
                                    passed_sc = get_decimal_sub(sc, bet_dict["stime"])
                                    if get_decimal_mod(passed_sc, c.TP_SL_MANUAL_TERM) == 0:
                                        if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= now_ask:
                                            stop_loss_flg = True
                                            if ALL_DEAL_FLG:
                                                all_deal_flg = True
                                            stop_price = now_ask
                                        elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= now_ask:
                                            take_profit_flg = True
                                            stop_price = now_ask

                                if stop_loss_flg or take_profit_flg:
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                        tmp_profit = bet_dict["price"] - stop_price - btcusd_spr
                                    else:
                                        tmp_profit = bet_dict["price"] - stop_price + c.ADJUST_PIPS
                                    if c.JPY_FLG == False:
                                        profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                    bet_dict["jpy"]) * jpy
                                    else:
                                        profit = tmp_profit * c.get_fx_position(bet_dict["price"])

                                    if stop_loss_flg:
                                        pips_sl.append(tmp_profit)
                                    elif take_profit_flg:
                                        pips_tp.append(tmp_profit)

                                elif c.FX_STOPLOSS_PER_SEC_FLG:
                                    # 一定秒数経過ごとに損切り判定する
                                    # ここに来たということはtickのstoploss,takeprofitに引っかからなかったということ

                                    if bet_dict["sps_stime"] != None:
                                        passed_time = get_decimal_sub(sc, bet_dict["sps_stime"])
                                        if passed_time in c.FX_STOPLOSS_PER_SEC_CHK_TIMES and passed_time in stoploss_per_sec_dict.keys():

                                            rate_change = float(Decimal(str(now_ask)) - Decimal(str(bet_dict["price"])))
                                            if rate_change > 0 and rate_change in stoploss_per_sec_dict[
                                                passed_time].keys() and \
                                                    stoploss_per_sec_dict[passed_time][rate_change] <= (
                                                    rate_change * -1):
                                                # 損切りする
                                                stop_loss_flg = True
                                                if ALL_DEAL_FLG:
                                                    all_deal_flg = True

                                                stop_price = now_ask

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal(
                                                            "100"))) + 0.1
                                                    tmp_profit = get_sub(now_ask, bet_dict["price"]) - btcusd_spr
                                                else:
                                                    tmp_profit = get_sub(now_ask, bet_dict["price"]) + c.ADJUST_PIPS
                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * c.get_fx_position_jpy(bet_dict["price"],
                                                                                                bet_dict["jpy"]) * jpy
                                                else:
                                                    profit = tmp_profit * c.get_fx_position(bet_dict["price"])
                                                pips_sps.append(tmp_profit)
                            else:
                                # 想定外エラー
                                output("ERROR2")
                                sys.exit()

                        if stop_loss_flg or take_profit_flg:
                            prev_bet_end_score = sc

                            pips.append(tmp_profit)
                            atrs.append(bet_dict["atr"])
                            inds.append(bet_dict["ind"])
                            times.append(bet_dict["stime"])
                            spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], tmp_profit, profit])

                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money
                            position_num = position_num - 1
                            position_num_tmp[sc] = position_num

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(
                                bet_dict["len"], 0) != 0 else 1

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["price"],
                                          "eprice": stop_price,
                                          "profit_pips": tmp_profit,
                                          "profit": profit,
                                          "score":bet_dict["stime"],
                                          }
                            if target == "category":
                                hist_child["pred"] = bet_dict["pred"]

                            if show_profit_per_div:
                                for d in show_profit_per_div_list:
                                    key_str = "div" + str(d)
                                    hist_child[key_str] = bet_dict[key_str]

                            deal_hist_dict[hist_child["score"]] = hist_child
                            deal_hist.append(hist_child)
                            del_key.append(dict_key)

                    for dkey in del_key:
                        del bet_dicts[dkey]

                else:
                    if mode == "limit":
                        for dict_key in bet_dicts.keys():
                            bet_dict = copy.deepcopy(bet_dicts[dict_key])

                            if bet_dict["deal"] != True:
                                # 約定してない場合　このtermで約定するか確認
                                for tick_bid, tick_ask in zip(bid_list, ask_list):
                                    if bet_dict["type"] == "BUY":
                                        if bet_dicts[dict_key]["deal"] == False and tick_ask <= bet_dict["limit_price"]:
                                            bet_dicts[dict_key]["deal"] = True
                                            if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                                bet_dicts[dict_key]["sps_stime"] = sc
                                            break
                                    elif bet_dict["type"] == "SELL":
                                        if bet_dicts[dict_key]["deal"] == False and tick_bid >= bet_dict["limit_price"]:
                                            bet_dicts[dict_key]["deal"] = True
                                            if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                                bet_dicts[dict_key]["sps_stime"] = sc
                                            break

                # 全建玉の許容損失を超えたら全建玉を決済する
                if TOTAL_STOPLOSS != None and TOTAL_STOPLOSS >= total_profit_pips:
                    all_deal_flg = True

                if all_deal_flg:
                    prev_bet_end_score = sc

                    for dict_key in bet_dicts.keys():
                        bet_dict = copy.deepcopy(bet_dicts[dict_key])

                        if bet_dict["type"] == "BUY":
                            # 決済する
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                profit_pips = deal_bid - bet_dict["price"] - btcusd_spr
                            else:
                                profit_pips = deal_bid - bet_dict["price"] + c.ADJUST_PIPS
                            pips.append(profit_pips)
                            atrs.append(bet_dict["atr"])
                            inds.append(bet_dict["ind"])
                            times.append(bet_dict["stime"])
                            if c.JPY_FLG == False:
                                profit = profit_pips * c.get_fx_position_jpy(bet_dict["price"],
                                                                             bet_dict["jpy"]) * jpy
                            else:
                                profit = profit_pips * c.get_fx_position(bet_dict["price"])
                            spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                            stop_price = deal_bid
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(
                                bet_dict["len"], 0) != 0 else 1

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["price"],
                                          "eprice": stop_price,
                                          "profit_pips": profit_pips,
                                          "profit": profit,
                                          "score":bet_dict["stime"],}

                            if target == "category":
                                hist_child["pred"] = bet_dict["pred"]

                            if show_profit_per_div:
                                for d in show_profit_per_div_list:
                                    key_str = "div" + str(d)
                                    hist_child[key_str] = bet_dict[key_str]

                            deal_hist_dict[hist_child["score"]] = hist_child
                            deal_hist.append(hist_child)

                        elif bet_dict["type"] == "SELL":
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                profit_pips = ((deal_ask - bet_dict["price"]) * -1) - btcusd_spr
                            else:
                                profit_pips = ((deal_ask - bet_dict["price"]) * -1) + c.ADJUST_PIPS
                            pips.append(profit_pips)
                            atrs.append(bet_dict["atr"])
                            inds.append(bet_dict["ind"])
                            times.append(bet_dict["stime"])
                            if c.JPY_FLG == False:
                                profit = profit_pips * c.get_fx_position_jpy(bet_dict["price"],
                                                                             bet_dict["jpy"]) * jpy
                            else:
                                profit = profit_pips * c.get_fx_position(bet_dict["price"])
                            spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                            stop_price = deal_ask
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                            money_tmp[sc] = money

                            bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(
                                bet_dict["len"], 0) != 0 else 1

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["price"],
                                          "eprice": stop_price,
                                          "profit_pips": profit_pips,
                                          "profit": profit,
                                          "score":bet_dict["stime"],}

                            if target == "category":
                                hist_child["pred"] = bet_dict["pred"]

                            if show_profit_per_div:
                                for d in show_profit_per_div_list:
                                    key_str = "div" + str(d)
                                    hist_child[key_str] = bet_dict[key_str]

                            deal_hist_dict[hist_child["score"]] = hist_child
                            deal_hist.append(hist_child)
                        else:
                            # 想定外エラー
                            output("ERROR3")
                            sys.exit()

                    bet_dicts = {}

                    position_num = 0
                    position_num_tmp[sc] = position_num
                    continue

                if idx != -1:
                    if target in ["d", "sub"]:
                        if c.LEARNING_TYPE == "REGRESSION":
                            pred = predict_list[idx]
                        elif c.LEARNING_TYPE == "REGRESSION_OCOPS":
                            pred = predict_list[idx, :]
                    elif target in ["category", "category_bin", "category_bin_both"]:
                        pred = predict_list[idx, :]
                    else:
                        pred = None

                    if target in ["d", "sub", "category", "category_bin", "category_bin_both"]:
                        if TK_SL_TYPE == "pips":
                            real_spread = get_decimal_multi(c.PIPS, spr)

                            if target == 'category':
                                tp = get_category_tp_sl_by_pred(c.FX_TAKE_PROFIT, pred)
                                sl = get_category_tp_sl_by_pred(c.FX_STOP_LOSS, pred)
                                tp_width = get_decimal_sub(tp, real_spread)
                                sl_width = get_decimal_sub(sl, real_spread)
                            else:
                                tp_width = get_decimal_sub(c.FX_TAKE_PROFIT, real_spread)
                                sl_width = get_decimal_add(c.FX_STOP_LOSS, real_spread)

                        elif TK_SL_TYPE == "satr":
                            tp_width = get_decimal_multi(atr, c.FX_TP_ATR)
                            sl_width = get_decimal_multi(atr, c.FX_SL_ATR)

                        elif TK_SL_TYPE == "atr":
                            # atrの場合はsatrに変換する
                            t_satr = get_satr(atr, now_price)
                            tp_width = get_decimal_multi(t_satr, c.FX_TP_ATR)
                            sl_width = get_decimal_multi(t_satr, c.FX_SL_ATR)

                        elif TK_SL_TYPE == "pred":
                            if target == "d":
                                pred_close = get_rate_severe(pred, now_price)
                                abs_close = abs(pred_close - now_price)

                                tp_width = get_decimal_multi(abs_close, c.FX_TP_PRED)
                                sl_width = get_decimal_multi(abs_close, c.FX_SL_PRED)
                            elif target == "sub":
                                abs_close = abs(pred)
                                tp_width = get_decimal_multi(abs_close, c.FX_TP_PRED)
                                sl_width = get_decimal_multi(abs_close, c.FX_SL_PRED)

                        if c.FX_MIN_TP_SL != None:
                            if c.FX_MIN_TP_SL > tp_width:
                                tp_width = c.FX_MIN_TP_SL

                            if c.FX_MIN_TP_SL > sl_width:
                                sl_width = c.FX_MIN_TP_SL

                        x_std_buy_tp = get_decimal_add(now_ask, tp_width)
                        x_std_buy_sl = get_decimal_sub(now_ask, sl_width)

                        x_std_sell_tp = get_decimal_sub(now_bid, tp_width)
                        x_std_sell_sl = get_decimal_add(now_bid, sl_width)

                        if mode == "limit":
                            # todo:指値を設定
                            x_std_buy_limit = 0
                            x_std_sell_limit = 0
                        else:
                            x_std_buy_limit = 0
                            x_std_sell_limit = 0

                # 決済するかどうかを判断
                del_key = []
                for dict_key in bet_dicts.keys():
                    bet_dict = copy.deepcopy(bet_dicts[dict_key])

                    passed_sc = get_decimal_sub(sc, bet_dict["stime"])

                    finish_flg = False

                    if c.FX_MAX_TRADE_SEC != None and passed_sc >= c.FX_MAX_TRADE_SEC:
                        # 最大取引時間を迎えたら決済
                        finish_flg = True
                    elif bet_dict["deal"] != True and passed_sc >= c.TERM:
                        # 約定しないまま決済期間が到来した場合
                        finish_flg = True
                    elif c.FX_NOT_EXT_FLG and passed_sc >= c.TERM:
                        finish_flg = True

                    no_pred_flg = False
                    try:
                        pred_ext = score_pred_dict_ext[sc]  # scoreをもとに延長用予想を取得
                        pred_score.append(sc)
                    except Exception as e:
                        no_pred_flg = True
                        no_pred_score.append(sc)

                    if finish_flg == False:

                        if ext_start_sec_short_position_num != None and len(
                                bet_dicts) <= ext_start_sec_short_position_num:
                            ext_start_sec_tmp = ext_start_sec_short
                        else:
                            ext_start_sec_tmp = ext_start_sec

                        if ext_term_short_position_num != None and len(bet_dicts) <= ext_term_short_position_num:
                            ext_term_tmp = ext_term_short
                        else:
                            ext_term_tmp = ext_term

                        if passed_sc < ext_start_sec_tmp:
                            continue
                        elif get_decimal_mod(get_decimal_sub(passed_sc, ext_start_sec_tmp), ext_term_tmp) != 0:
                            # ext_termが経過するごとに延長判断する
                            continue
                        elif c.TRADE_SHIFT != None and get_decimal_mod(sc, c.TRADE_SHIFT) != 0:
                            # 指定した秒のシフトでないと取引しない
                            continue

                        else:
                            if c.FX_NOT_EXT_FLG == False:

                                # スプレッドが範囲内かどうか
                                if end_min_spread <= spr and spr <= end_max_spread:
                                    within_end_spread_flg = True
                                else:
                                    within_end_spread_flg = False

                                if bet_dict["type"] == "BUY":
                                    # 買いポジションがある場合
                                    bet_flg = False

                                    if no_pred_flg:
                                        # 予想がない場合は判断材料がないので延長しない
                                        bet_flg = False
                                    else:
                                        if target_ext in ["d", "sub"]:
                                            bet_flg = buy_d_cond_1(pred_ext, ext_border)
                                        elif target_ext == "category":
                                            bet_flg = buy_cat_cond_ext(pred_ext, ext_border)

                                            if OVER_BEF_PREDICT:
                                                # 前回延長時予想より下がったら決済
                                                bef_predict = bet_dict.get("bef_predict")
                                                if bef_predict != None and bef_predict > pred_ext[0]:
                                                    bet_flg = False

                                        elif target_ext == "category_bin":
                                            bet_flg = buy_cat_cond_ext_2(pred_ext, ext_border)
                                        elif target_ext == "category_bin_both":
                                            bet_flg = buy_cat_cond_ext(pred_ext, ext_border[0])

                                    tmp_bet_flg = True

                                    for j, col in enumerate(c.IND_COLS):
                                        if len(ind_range_exts[j]) != 0:  # INDを値で絞る
                                            ok_flg = False

                                            for r in ind_range_exts[j]:
                                                r_min, r_max = r.split("-")
                                                if (float(r_min) <= ind[j] and ind[j] < float(r_max)) == True:
                                                    ok_flg = True
                                                    break

                                            if ok_flg == False:
                                                tmp_bet_flg = False
                                                break

                                    if tmp_bet_flg == False:
                                        bet_flg = False

                                    if c.ATR_COL != "" and len(atr_range_ext) != 0:  # ATRを使用する場合は値で絞る
                                        ok_flg = False

                                        for t_atr in atr_range_ext:
                                            atr_min, atr_max = t_atr.split("-")
                                            if (float(atr_min) <= atr and atr < float(atr_max)) == True:
                                                ok_flg = True
                                                break

                                        if ok_flg == False:
                                            bet_flg = False

                                    if passed_sc == c.TERM and c.FX_NOT_EXT_MINUS != None:
                                        # 延長一回目でFX_NOT_EXT_MINUS未満の利益なら延長しない
                                        tmp_profit = now_bid - bet_dict["price"]

                                        if tmp_profit < c.FX_NOT_EXT_MINUS:
                                            bet_flg = False

                                    if bet_flg or (
                                            bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                        "cannot_deal_cnt"] < cannot_deal_cnt_max):
                                        # 更に上がると予想されている場合、決済しないままとする
                                        # またはスプレッドが範囲外で且つ、範囲外であった回数が規定内である場合、決済しないままとする

                                        # takeprofit,stoplossをいれなおす
                                        # bet_dict[dict_key]["tp"] = x_std_buy_tp
                                        # bet_dict[dict_key]["sl"] = x_std_buy_sl
                                        bet_dicts[dict_key]["len"] += 1
                                        # tp_list.append(x_std * FX_TP_SIG)
                                        # sl_list.append(x_std * FX_SL_SIG)
                                        if OVER_BEF_PREDICT and target_ext == "category":
                                            bet_dicts[dict_key]["bef_predict"] = pred_ext[0]

                                        if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                            bet_dicts[dict_key]["sps_stime"] = sc

                                        if bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                            "cannot_deal_cnt"] < cannot_deal_cnt_max:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] += 1
                                        else:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] = 0  # カウントを戻す

                                    else:
                                        finish_flg = True

                                elif bet_dict["type"] == "SELL":
                                    # 売りポジションがある場合
                                    bet_flg = False
                                    if no_pred_flg:
                                        # 予想がない場合は判断材料がないので延長しない
                                        bet_flg = False
                                    else:
                                        if target_ext in ["d", "sub"]:
                                            bet_flg = sell_d_cond_1(pred_ext, ext_border)
                                        elif target_ext == "category":
                                            bet_flg = sell_cat_cond_ext(pred_ext, ext_border)
                                            if OVER_BEF_PREDICT:
                                                # 前回延長時予想より下がったら決済
                                                bef_predict = bet_dict.get("bef_predict")
                                                if bef_predict != None and bef_predict > pred_ext[2]:
                                                    bet_flg = False

                                        elif target_ext == "category_bin":
                                            bet_flg = sell_cat_cond_ext_2(pred_ext, ext_border)
                                        elif target_ext == "category_bin_both":
                                            bet_flg = sell_cat_cond_ext(pred_ext, ext_border[1])

                                    tmp_bet_flg = True

                                    for j, col in enumerate(c.IND_COLS):
                                        if len(ind_range_exts[j]) != 0:  # INDを値で絞る
                                            ok_flg = False

                                            for r in ind_range_exts[j]:
                                                r_min, r_max = r.split("-")
                                                if (float(r_min) <= ind[j] and ind[j] < float(r_max)) == True:
                                                    ok_flg = True
                                                    break

                                            if ok_flg == False:
                                                tmp_bet_flg = False
                                                break

                                    if tmp_bet_flg == False:
                                        bet_flg = False

                                    if c.ATR_COL != "" and len(atr_range_ext) != 0:  # ATRを使用する場合は値で絞る
                                        ok_flg = False

                                        for t_atr in atr_range_ext:
                                            atr_min, atr_max = t_atr.split("-")
                                            if (float(atr_min) <= atr and atr < float(atr_max)) == True:
                                                ok_flg = True
                                                break

                                        if ok_flg == False:
                                            bet_flg = False

                                    if passed_sc == c.TERM and c.FX_NOT_EXT_MINUS != None:
                                        # 延長一回目でFX_NOT_EXT_MINUS未満の利益なら延長しない
                                        tmp_profit = ((now_ask - bet_dict["price"]) * -1)

                                        if tmp_profit < c.FX_NOT_EXT_MINUS:
                                            bet_flg = False

                                    if bet_flg or (
                                            bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                        "cannot_deal_cnt"] < cannot_deal_cnt_max):
                                        # bet_dict[dict_key]["tp"] = x_std_sell_tp
                                        # bet_dict[dict_key]["sl"] = x_std_sell_sl
                                        bet_dicts[dict_key]["len"] += 1
                                        # tp_list.append(x_std * FX_TP_SIG)
                                        # sl_list.append(x_std * FX_SL_SIG)
                                        if OVER_BEF_PREDICT and target_ext == "category":
                                            bet_dicts[dict_key]["bef_predict"] = pred_ext[2]

                                        if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                            bet_dicts[dict_key]["sps_stime"] = sc

                                        if bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                            "cannot_deal_cnt"] < cannot_deal_cnt_max:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] += 1
                                        else:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] = 0  # カウントを戻す

                                    else:
                                        finish_flg = True

                    if finish_flg:
                        if bet_dict["deal"] != True:
                            # 約定してない場合注文なしとする
                            del_key.append(dict_key)
                        else:
                            prev_bet_end_score = sc
                            if bet_dict["type"] == "BUY":
                                # 決済する
                                if conf.SYMBOL == "BTCUSD":
                                    btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                    profit_pips = get_sub(bet_dict["price"], deal_bid) - btcusd_spr
                                else:
                                    profit_pips = get_sub(bet_dict["price"], deal_bid) + c.ADJUST_PIPS
                                pips.append(profit_pips)
                                atrs.append(bet_dict["atr"])
                                inds.append(bet_dict["ind"])
                                times.append(bet_dict["stime"])
                                if c.JPY_FLG == False:
                                    profit = profit_pips * c.get_fx_position_jpy(bet_dict["price"],
                                                                                 bet_dict["jpy"]) * jpy
                                else:
                                    profit = profit_pips * c.get_fx_position(bet_dict["price"])
                                spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                                stop_price = deal_bid
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                position_num = position_num - 1
                                position_num_tmp[sc] = position_num

                                bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(
                                    bet_dict["len"], 0) != 0 else 1

                                hist_child = {"type": bet_dict["type"],
                                              "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                                  '%Y/%m/%d %H:%M:%S'),
                                              "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                              "sprice": bet_dict["price"],
                                              "eprice": stop_price,
                                              "profit_pips": profit_pips,
                                              "profit": profit,
                                              "score":bet_dict["stime"],}

                                if target == "category":
                                    hist_child["pred"] = bet_dict["pred"]

                                if show_profit_per_div:
                                    for d in show_profit_per_div_list:
                                        key_str = "div" + str(d)
                                        hist_child[key_str] = bet_dict[key_str]

                                deal_hist_dict[hist_child["score"]] = hist_child
                                deal_hist.append(hist_child)
                                del_key.append(dict_key)

                            elif bet_dict["type"] == "SELL":
                                if conf.SYMBOL == "BTCUSD":
                                    btcusd_spr = float(Decimal(str(bet_dict["price"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100"))) + 0.1
                                    profit_pips = get_sub(deal_ask, bet_dict["price"]) - btcusd_spr
                                else:
                                    profit_pips = get_sub(deal_ask, bet_dict["price"]) + c.ADJUST_PIPS
                                pips.append(profit_pips)
                                atrs.append(bet_dict["atr"])
                                inds.append(bet_dict["ind"])
                                times.append(bet_dict["stime"])
                                if c.JPY_FLG == False:
                                    profit = profit_pips * c.get_fx_position_jpy(bet_dict["price"],
                                                                                 bet_dict["jpy"]) * jpy
                                else:
                                    profit = profit_pips * c.get_fx_position(bet_dict["price"])
                                spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                                stop_price = deal_ask
                                money = money + profit
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)
                                money_tmp[sc] = money
                                position_num = position_num - 1
                                position_num_tmp[sc] = position_num

                                bet_len_dict[bet_dict["len"]] = bet_len_dict[bet_dict["len"]] + 1 if bet_len_dict.get(
                                    bet_dict["len"], 0) != 0 else 1

                                hist_child = {"type": bet_dict["type"],
                                              "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                                  '%Y/%m/%d %H:%M:%S'),
                                              "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                              "sprice": bet_dict["price"],
                                              "eprice": stop_price,
                                              "profit_pips": profit_pips,
                                              "profit": profit,
                                              "score":bet_dict["stime"],}

                                if target == "category":
                                    hist_child["pred"] = bet_dict["pred"]

                                if show_profit_per_div:
                                    for d in show_profit_per_div_list:
                                        key_str = "div" + str(d)
                                        hist_child[key_str] = bet_dict[key_str]

                                deal_hist_dict[hist_child["score"]] = hist_child
                                deal_hist.append(hist_child)
                                del_key.append(dict_key)
                            else:
                                # 想定外エラー
                                output("ERROR3")
                                sys.exit()

                for dkey in del_key:
                    del bet_dicts[dkey]

                # 一定利益が上がっていた場合、損切りラインを上げる
                if c.CHANGE_STOPLOSS_FLG:
                    for dict_key in bet_dicts.keys():
                        bet_dict = copy.deepcopy(bet_dicts[dict_key])
                        passed_sc = get_decimal_sub(sc, bet_dict["stime"])
                        if c.CHANGE_STOPLOGG_TERM == None or get_decimal_mod(passed_sc, c.CHANGE_STOPLOGG_TERM) == 0:
                            if bet_dict["type"] == "BUY":
                                tmp_profit = now_bid - bet_dict["price"]
                                prev_profit = bet_dict["prev_profit"]
                                profit_sub = get_decimal_sub(tmp_profit, prev_profit)
                                if profit_sub > 0:
                                    bet_dicts[dict_key]["sl"] = get_decimal_add(bet_dicts[dict_key]["sl"], profit_sub)
                                    bet_dicts[dict_key]["prev_profit"] = tmp_profit

                            elif bet_dict["type"] == "SELL":
                                tmp_profit = ((now_ask - bet_dict["price"]) * -1)
                                prev_profit = bet_dict["prev_profit"]
                                profit_sub = get_decimal_sub(tmp_profit, prev_profit)
                                if profit_sub > 0:
                                    bet_dicts[dict_key]["sl"] = get_decimal_sub(bet_dicts[dict_key]["sl"], profit_sub)
                                    bet_dicts[dict_key]["prev_profit"] = tmp_profit

                if idx == -1:
                    # 予想がない場合は判断材料がないので注文しない
                    continue

                # 新規注文
                buy_bet_flg = False
                if target in ["d", "sub"]:
                    buy_bet_flg = buy_d_cond_1(pred, border, border_ceil)
                elif target == "category":
                    buy_bet_flg = buy_cat_cond_1(pred, border, border_ceil)
                elif target == "category_bin":
                    buy_bet_flg = buy_cat_cond_2(pred, border)
                elif target == "category_bin_both":
                    buy_bet_flg = buy_cat_cond_1(pred, border[0])

                if c.BUY_FLG == False:
                    buy_bet_flg = False

                sell_bet_flg = False
                if target in ["d", "sub"]:
                    sell_bet_flg = sell_d_cond_1(pred, border, border_ceil)
                elif target == "category":
                    sell_bet_flg = sell_cat_cond_1(pred, border, border_ceil)
                elif target == "category_bin":
                    sell_bet_flg = sell_cat_cond_2(pred, border)
                elif target == "category_bin_both":
                    sell_bet_flg = sell_cat_cond_1(pred, border[1])

                if c.SELL_FLG == False:
                    sell_bet_flg = False

                if buy_bet_flg == False and sell_bet_flg == False:
                    continue
                else:
                    now_shift = int(Decimal(str(sc)) % Decimal(str(c.TERM)))
                    ng_shift = int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT)))
                    if c.SAME_SHIFT_NG_FLG == True:
                        # 同じshiftの建玉が既にあるかチェック
                        same_shift_flg = False

                        for k, v in bet_dicts.items():
                            if v["shift"] == ng_shift:
                                same_shift_flg = True
                                break
                        if same_shift_flg:
                            # 同じshiftの建玉が既にある場合新規発注しない
                            continue

                    if c.FX_MAX_POSITION_CNT != None and len(bet_dicts) == c.FX_MAX_POSITION_CNT:
                        # 既に最大保持可能ポジション数の場合、新規発注しない
                        continue

                    if c.FX_SINGLE_FLG == True and len(bet_dicts) != 0:
                        continue

                    """
                    # FX_FUND * FX_LEVERAGE - (既に保持しているポジション数 * FX_FIX_POSITION * rate)の値が0以上であれば(資金余裕があれば)新規ポジションを持てる
                    if c.FX_FIX_POSITION != 0:
                        if c.FX_FUND * c.FX_LEVERAGE - ((position_num + 1) * c.FX_FIX_POSITION * now_price) < 0:

                            fund_out_cnt += 1
                            continue
                    """

                    # 指定シフト以外トレードしない
                    if (len(c.FX_TARGET_SHIFT) == 0 or (
                            len(c.FX_TARGET_SHIFT) != 0 and now_shift in c.FX_TARGET_SHIFT)) == False:
                        continue

                    # 指定スプレッド以外のトレードは無視する
                    if (start_min_spread <= spr and spr <= start_max_spread) == False:
                        ng_spread_cnt += 1
                        # print("spr:",spr, " score:", sc)
                        continue
                    else:
                        ok_spread_cnt += 1

                    # 取引時間外になってしまうなら新規注文しない
                    tmp_now_dt = datetime.fromtimestamp(sc)
                    if start > tmp_now_dt or end < tmp_now_dt:
                        continue

                    # 予想結果時間が取引時間外になってしまうなら新規注文しない
                    tmp_limit_dt = tmp_now_dt + timedelta(seconds=c.TERM)
                    if tmp_limit_dt.hour in c.EXCEPT_LIST_HOUR_TEST:
                        continue

                    if tmp_now_dt.second in c.EXCEPT_LIST_SEC_TEST:
                        # 取引時間外設定(秒)
                        continue

                    if c.TRADE_SHIFT != None and get_decimal_mod(sc, c.TRADE_SHIFT) != 0:
                        # 指定した秒のシフトでないと取引しない
                        continue

                    # 一定時間経過しないとつづけて注文できない
                    if c.RESTRICT_FLG:
                        if prev_bet_start_score != None and get_decimal_add(prev_bet_start_score, c.RESTRICT_SEC) > sc:
                            continue
                        elif prev_bet_end_score != None and get_decimal_add(prev_bet_end_score, c.RESTRICT_SEC) > sc:
                            continue

                    bet_flg = True
                    for j, col in enumerate(c.IND_COLS):
                        if len(ind_range_bases[j]) != 0:  # INDを値で絞る
                            ok_flg = False

                            for r in ind_range_bases[j]:
                                r_min, r_max = r.split("-")
                                if (float(r_min) <= ind[j] and ind[j] < float(r_max)) == True:
                                    ok_flg = True
                                    break

                            if ok_flg == False:
                                bet_flg = False
                                break

                    if bet_flg == False:
                        continue

                    if c.ATR_COL != "" and len(atr_range_base) != 0:  # ATRを使用する場合は値で絞る
                        ok_flg = False

                        for t_atr in atr_range_base:
                            atr_min, atr_max = t_atr.split("-")
                            if (float(atr_min) <= atr and atr < float(atr_max)) == True:
                                ok_flg = True
                                break

                        if ok_flg == False:
                            continue

                    if importantAnswer.is_except(sc):
                        # 重要指標発表時は除外
                        continue

                    if max_div != None:
                        max_dw, max_up = max_div
                        tmp_bef = close_list[cnt - int(get_decimal_divide(max_div_sec, c.BET_TERM))]
                        tmp_div = get_divide(tmp_bef, close)
                        if (max_dw <= tmp_div and tmp_div <= max_up) == False:
                            continue

                    if ORDER_TOTAL_STOPLOSS != None:
                        if total_profit_pips <= ORDER_TOTAL_STOPLOSS:
                            # 現在の全建玉の損失が大きければ新たに発注しない
                            continue

                    if AGAINST_FLG == True:
                        # 逆張りのみ行う
                        tmp_bef = close_list[cnt - int(get_decimal_divide(AGAINST_SEC, c.BET_TERM))]
                        tmp_div = get_divide(tmp_bef, close)
                        if buy_bet_flg and tmp_div > AGAINST_DIV:
                            buy_bet_flg = False

                        if sell_bet_flg and tmp_div < AGAINST_DIV * -1:
                            sell_bet_flg = False

                    if buy_bet_flg:
                        prev_bet_start_score = sc
                        bet_dicts[sc] = {
                            "shift": int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT))),
                            "type": "BUY",
                            "stime": sc,
                            "len": 1,
                            "tp": x_std_buy_tp,
                            "sl": x_std_buy_sl,
                            "atr": atr,
                            "spr": spr,
                            "jpy": jpy,
                            "ind": ind,
                            "sps_stime": None,  # FX_STOPLOSS_PER_SECの基準となるベットした時間,または延長した時間
                            "prev_profit": 0,
                            "cannot_deal_cnt": 0,
                            # スプレッドがend_min_spreadからend_max_spreadの範囲外のために決済できなかった回数　cannot_deal_cnt_maxを超えた場合決済する
                        }

                        if mode == "market":
                            bet_dicts[sc]["price"] = order_ask
                            bet_dicts[sc]["deal"] = True
                            bet_dicts[sc]["limit_price"] = None
                            if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                bet_dicts[sc]["sps_stime"] = sc

                        elif mode == "limit":
                            bet_dicts[sc]["price"] = x_std_buy_limit
                            bet_dicts[sc]["deal"] = False
                            bet_dicts[sc]["limit_price"] = x_std_buy_limit

                        if target == "category":
                            bet_dicts[sc]["pred"] = pred[0]

                        if show_profit_per_div:
                            for d in show_profit_per_div_list:
                                bef_c = close_list[cnt - int(get_decimal_divide(d, c.BET_TERM))]
                                tmp_d = get_divide(bef_c, close)
                                bet_dicts[sc]["div" + str(d)] = tmp_d

                        bet_cnt += 1
                        position_num = position_num + 1
                        position_num_tmp[sc] = position_num

                        # tp_list.append(x_std * FX_TP_SIG)
                        # sl_list.append(x_std * FX_SL_SIG)
                        # limit_list.append(x_std * FX_LIMIT_SIG)

                    elif sell_bet_flg:
                        prev_bet_start_score = sc
                        bet_dicts[sc] = {
                            "shift": int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT))),
                            "type": "SELL",
                            "stime": sc,
                            "len": 1,
                            "tp": x_std_sell_tp,
                            "sl": x_std_sell_sl,
                            "atr": atr,
                            "spr": spr,
                            "jpy": jpy,
                            "ind": ind,
                            "sps_stime": None,  # FX_STOPLOSS_PER_SECの基準となるベットした時間,または延長した時間
                            "prev_profit": 0,
                            "cannot_deal_cnt": 0,
                            # スプレッドがend_min_spreadからend_max_spreadの範囲外のために決済できなかった回数　cannot_deal_cnt_maxを超えた場合決済する
                        }

                        if mode == "market":
                            bet_dicts[sc]["price"] = order_bid
                            bet_dicts[sc]["deal"] = True
                            bet_dicts[sc]["limit_price"] = None
                            if c.FX_STOPLOSS_PER_SEC_FLG == True:
                                bet_dicts[sc]["sps_stime"] = sc

                        elif mode == "limit":
                            bet_dicts[sc]["price"] = x_std_sell_limit
                            bet_dicts[sc]["deal"] = False
                            bet_dicts[sc]["limit_price"] = x_std_sell_limit

                        if target == "category":
                            bet_dicts[sc]["pred"] = pred[2]

                        if show_profit_per_div:
                            for d in show_profit_per_div_list:
                                bef_c = close_list[cnt - int(get_decimal_divide(d, c.BET_TERM))]
                                tmp_d = get_divide(bef_c, close)
                                bet_dicts[sc]["div" + str(d)] = tmp_d

                        bet_cnt += 1
                        position_num = position_num + 1
                        position_num_tmp[sc] = position_num

                        # tp_list.append(x_std * FX_TP_SIG)
                        # sl_list.append(x_std * FX_SL_SIG)
                        # limit_list.append(x_std * FX_LIMIT_SIG)

            prev_money = c.START_MONEY

            for i, score in enumerate(score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)

            detail_profit = prev_money - c.START_MONEY
            output("")
            if ok_spread_cnt > 0 or ng_spread_cnt > 0:
                output("ok_spread_cnt:", ok_spread_cnt, "ok_spread_percent:",
                       ok_spread_cnt / (ok_spread_cnt + ng_spread_cnt))
                output("ng_spread_cnt:", ng_spread_cnt, "ng_spread_percent:",
                       ng_spread_cnt / (ok_spread_cnt + ng_spread_cnt))

            output("pred_cnt:", len(pred_score))
            output("no_pred_cnt:", len(no_pred_score))
            # output("no_pred_score_min:", min(no_pred_score))
            # output("no_pred_score_max:", max(no_pred_score))

            output("")
            output("border ", border)
            output("bet cnt: ", bet_cnt)
            output("Detail Earned Money: ", detail_profit)
            output("fund_out_cnt: ", fund_out_cnt)
            # 儲けが出たらwinとする
            win_cnt = 0
            win_rate = 0
            if bet_cnt != 0:
                d_np = np.array(pips)
                d_np = np.sort(d_np)

                win_cnt = len(np.where(d_np >= 0)[0])
                win_rate = win_cnt / len(d_np)
                # 勝ち数から負け数を引いて、純粋な勝ち数とする
                # win_cnt = win_cnt - (len(d_np) - win_cnt)

                output("win_cnt:", win_cnt)
                output("win_rate:", win_rate)
                output("pips length:", len(d_np))
                output("pips avg:", np.average(d_np))
                output("pips max:", np.max(d_np))
                output("pips min:", np.min(d_np))

                if len(pips_sl) != 0:
                    d_np = np.array(pips_sl)
                    output("sl_pips length:", len(d_np))
                    output("sl_pips avg:", np.average(d_np))
                    output("sl_pips max:", np.max(d_np))
                    output("sl_pips min:", np.min(d_np))

                if len(pips_tp) != 0:
                    d_np = np.array(pips_tp)
                    output("tk_pips length:", len(d_np))
                    output("tk_pips avg:", np.average(d_np))
                    output("tk_pips max:", np.max(d_np))
                    output("tk_pips min:", np.min(d_np))

                if len(pips_sps) != 0:
                    d_np = np.array(pips_sps)
                    output("sps_pips length:", len(d_np))
                    output("sps_pips avg:", np.average(d_np))
                    output("sps_pips max:", np.max(d_np))
                    output("sps_pips min:", np.min(d_np))

                if len(tp_list) != 0:
                    tp_np = np.array(tp_list)
                    output("take_profit avg:", np.average(tp_np))
                    output("take_profit max:", np.max(tp_np))
                    output("take_profit min:", np.min(tp_np))
                if len(sl_list) != 0:
                    sl_np = np.array(sl_list)
                    output("stop_loss avg:", np.average(sl_np))
                    output("stop_loss max:", np.max(sl_np))
                    output("stop_loss min:", np.min(sl_np))

                if mode == "limit":
                    limit_np = np.array(limit_list)
                    output("limit avg:", np.average(limit_np))
                    output("limit max:", np.max(limit_np))
                    output("limit min:", np.min(limit_np))

            for i in result_txt:
                output_log(i)

            profit_per_drawdown = 0
            tmp_drawdown = 0
            if len(max_drawdowns) != 0:
                max_drawdowns.sort()
                profit_per_drawdown = int(detail_profit) / (int(max_drawdowns[0]) * -1 + c.FX_FUND)
                tmp_drawdown = int(max_drawdowns[0])

            output("profit_per_dd: ", profit_per_drawdown, tmp_drawdown)

            sl_bet_cnt = 0
            sl_cnt = len(pips_sl)
            if bet_cnt != 0:
                sl_bet_cnt = sl_cnt / bet_cnt

            result_per_suffix_border[
                str(suffix) + "-" + str(border) + "-" + str(ext_border) + "-" + str(border_ceil)] = {
                "profit_per_dd": profit_per_drawdown, "profit": int(detail_profit), "dd": tmp_drawdown,
                "bet_cnt": bet_cnt, "win_cnt": win_cnt, "win_rate": win_rate}

            # output(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")
            output("sl/bet cnt:", sl_bet_cnt)

            if show_profit_ind:
                showProfitIND(border, c, pips, inds, show_plot, save_dir)

            if show_profit_atr:
                showProfitAtr(pips, atrs)

            if show_profit_time:
                showProfitTime(c, pips, times)

            if show_profit_per_spread:
                showPipsPerSpread(np.array(spr_pred_pips_list))

            if show_profit_per_pred:
                showPipsPerPred(deal_hist)

            if show_profit_per_div:
                showPipsPerDiv(deal_hist, show_profit_per_div_list)

            if show_history:
                output("決済履歴")
                for h in deal_hist:
                    output(h)

            if show_high_profit_deal:
                showHighProfitDeal(deal_hist_dict)

            if show_detail:
                """
                output("bet期間 件数")
                for k, v in sorted(bet_len_dict.items()):
                    output(k, bet_len_dict.get(k, 0))
                """
                output("MAX DrawDowns(理論上のドローダウン)")
                output_log(max_drawdowns[0:10])

                drawdown_cnt = {}
                for i in max_drawdowns:
                    for k, v in c.DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k, 0) + 1
                            break
                for k, v in sorted(c.DRAWDOWN_LIST.items()):
                    output(k, drawdown_cnt.get(k, 0))

            if show_plot:
                fig = plt.figure(figsize=(6.4 * 0.7, 4.8 * 0.7))
                # 価格の遷移
                ax1 = fig.add_subplot(111)

                ax1.plot(close_list, 'g')

                ax2 = ax1.twinx()
                ax2.plot(money_y, 'b')

                if show_position:
                    """
                    output("ポジション数")
                    sorted_d = sorted(position_num_tmp.items(), key=lambda x: x[1], reverse=True)
                    cnt_t = 0
                    for k, v in sorted_d:
                        if cnt_t > 20:
                            break
                        output(v)
                        cnt_t += 1
                    output("")
                    """
                    sorted_d = sorted(position_num_tmp.items(), key=lambda x: x[1], reverse=True)
                    cnt_t = 0
                    for k, v in sorted_d:
                        if cnt_t > 0:
                            break
                        output("最大ポジション数:", v)
                        cnt_t += 1

                    """
                    prev_position_num = 0
                    position_num_y = []
                    for i, score in enumerate(score_list):
                        if score in position_num_tmp.keys():
                            prev_position_num = position_num_tmp[score]

                        position_num_y.append(prev_position_num)
                    ax3 = ax1.twinx()
                    ax3.plot(position_num_y, 'r')
                    """
                plt.title(
                    'border:' + str(border) + ' ext_border:' + str(ext_border) + " money:" + str(
                        money))
                # plt.show()
                fig.savefig(filename)

    output("利益が多い順")
    sorted_d = sorted(result_per_suffix_border.items(), key=lambda x: x[1]["profit"], reverse=True)
    cnt_t = 0
    for k, v in sorted_d:
        if cnt_t > 20:
            break

        output("suffix-border-extborder:", k, "profit:", v["profit"], "profit_per_dd:", v["profit_per_dd"],
               "dd:", v["dd"], "bet_cnt:", v["bet_cnt"], "win_cnt:", v["win_cnt"], "win_rate:", v["win_rate"],
               )
        cnt_t += 1
    output("")


if __name__ == "__main__":

    start_ends = [
        [datetime(2024, 8, 4), datetime(2024, 8, 10, )],
        #[datetime(2024, 3, 1), datetime(2024, 8, 10,)],
        #[datetime(2023, 4, 1), datetime(2024, 8, 10)],
        # [datetime(2024, 5, 1), datetime(2024, 6, 30)],
        # [datetime(2023, 5, 1), datetime(2024, 6, 30)],
        # [datetime(2024, 2, 1), datetime(2024, 3, 2)],
        # [datetime(2024, 2, 28,15, ), datetime(2024, 2, 29,15)],
        # [datetime(2023, 5, 1), datetime(2023, 12, 1)],
        # [datetime(2024, 1, 1), datetime(2024, 2, 24)],

    ]

    start_time = time.perf_counter()
    # output("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)

    # LSTMのテストの場合
    # conf = conf_class.ConfClass()

    # LGBMのテストの場合
    conf = conf_class_lgbm.ConfClassLgbm()

    if conf.FX == False:
        output("conf.FX == False !!!")
        exit(1)

    # conf.change_fx_real_spread_flg(True)

    # target_spread_list = []
    spread_confs = [

        [
            0,  # start_min_spread
            2,  # start_max_spread
            0,  # end_min_spread
            2,  # end_max_spread
            1,  # cannot_deal_cnt_max スプレッドが範囲外であることによる決済先延ばしを、この回数以上出来ない
        ],

    ]

    # pos_list = []
    pos_list = [{'0-1': 0.4, },]

    for start_end in start_ends:
        start, end = start_end

        if len(pos_list) != 0:
            for sl in pos_list:
                conf.FX_STOP_LOSS = sl
                for spread_conf in spread_confs:
                    do_predict(conf, start, end, spread_conf)
        else:
            for spread_conf in spread_confs:
                do_predict(conf, start, end, spread_conf)

    print("Processing Time(Sec)", time.perf_counter() - start_time)

    print("END!!!")
    # 終わったらメールで知らせる
    mail.send_message(host, ": testLstmFX2_rgr_limit finished!!!")
