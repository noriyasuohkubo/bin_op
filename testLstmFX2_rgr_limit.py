import json
import pickle

import psutil
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

from DataSequence2_from_pickle_test_raw import DataSequence2_from_pickle_test_raw
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
from DataSequence2_from_pickle_test import DataSequence2_from_pickle_test
from DataSequence2_from_pickle_test_on_memory import DataSequence2_from_pickle_test_on_memory
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow_addons.optimizers import AdamW, RectifiedAdam, LazyAdam
from make_chart import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + "-limit.txt"
output = output_log(output_log_name, print_flg=True)
output_file = output_log(output_log_name, print_flg=False)

"""
指値注文の場合、延長しない

"""
# 設定ファイル読み込み
c = None
png_dir = "/app/fx/png/"
chart_dir = "/app/fx/chart"


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


def get_regression_tp_sl_by_pred(tp_sl_dict, pred):
    tp_sl = None

    for k, v in tp_sl_dict.items():
        start_k, end_k = k.split('-')
        start_k = float(start_k)
        end_k = float(end_k)
        if start_k <= abs(pred) and abs(pred) < end_k:
            tp_sl = v

    if tp_sl == None:
        print("get_regression_tp_sl_by_pred cannot get tp_sl")
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


def countDrawdoan(max_drawdowns, max_drawdown, drawdown, money, sc, max_drawdown_sc):
    drawdown = drawdown + money
    if max_drawdown > drawdown:
        # 最大ドローダウンを更新してしまった場合
        max_drawdown = drawdown
        max_drawdown_sc = sc

    if drawdown > 0:
        if max_drawdown != 0:
            max_drawdowns[max_drawdown_sc] = max_drawdown
        drawdown = 0
        max_drawdown = 0

    return max_drawdown, drawdown, max_drawdown_sc


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


def buy_cat_cond_ext2(pred, border, ):
    if border <= pred[0] or (pred[1] >= pred[2] and pred[1] >= pred[0]):
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


def sell_cat_cond_ext2(pred, border, ):
    if border <= pred[2] or (pred[1] >= pred[2] and pred[1] >= pred[0]):
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


def get_predict(file_name, dataSequence2, conf):
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
                                                                 'AdamW': AdamW,
                                                                 'LazyAdam': LazyAdam,
                                                                 'Adamax': Adamax,
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
        pips_tmp_arr = np.array(per_min_dict[i])
        bet_cnt = len(per_min_dict[i])
        if bet_cnt != 0:
            win_cnt = len(np.where(pips_tmp_arr >= 0)[0])
            per_min_winrate_dict[i] = [win_cnt / bet_cnt, bet_cnt, np.average(pips_tmp_arr)]

    per_hour_winrate_dict = {}
    for i in per_hour_dict.keys():
        pips_tmp_arr = np.array(per_hour_dict[i])
        bet_cnt = len(per_hour_dict[i])
        if bet_cnt != 0:
            win_cnt = len(np.where(pips_tmp_arr >= 0)[0])
            per_hour_winrate_dict[i] = [win_cnt / bet_cnt, bet_cnt, np.average(pips_tmp_arr)]

    output("理論上の秒毎の勝率悪い順(勝率,賭数,平均PIPS):")
    worst_sorted = sorted(per_sec_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0], ":", i[1][0], i[1][1], i[1][2])

    output("理論上の分毎の勝率悪い順(勝率,賭数,平均PIPS):")
    worst_sorted = sorted(per_min_winrate_dict.items(), key=lambda x: x[1][0])
    for i in worst_sorted:
        output(i[0], ":", i[1][0], i[1][1], i[1][2])

    output("理論上の時毎の勝率悪い順(勝率,賭数,平均PIPS):")
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


def showPipsPerTradeSec(deal_hist):
    trade_sec = 1
    end = 10000
    while True:
        if trade_sec > end:
            break

        total = [d.get("profit_pips") for d in deal_hist if d.get("trade_sec") == trade_sec]
        take_profit_d = [d for d in deal_hist if (d.get("trade_sec") == trade_sec and d.get("take_profit") == True)]
        stop_loss_d = [d for d in deal_hist if (d.get("trade_sec") == trade_sec and d.get("stop_loss") == True)]

        if len(total) != 0:
            win_cnt = len(np.where(np.array(total) >= 0)[0])
            output("TRADE SEC:", trade_sec, "Total BET CNT:", len(total), " ACC:", win_cnt / len(total), " AVG_PIPS:",
                   np.average(total),
                   "TAKE_PROFIT CNT:", len(take_profit_d), "STOP_LOSS CNT:", len(stop_loss_d))

        trade_sec += 1


def showStoplossHistory(deal_hist):
    output("showStoplossHistory:")

    # ストップロスとなった履歴を取得
    buy_hist_dict = {}
    sell_hist_dict = {}

    for d in deal_hist:
        if (d.get("stop_loss") == True and d.get("type") == "BUY"):
            buy_hist_dict[d.get("stime")] = d

    for d in deal_hist:
        if (d.get("stop_loss") == True and d.get("type") == "SELL"):
            sell_hist_dict[d.get("stime")] = d

    sorted_buy_hist = sorted(buy_hist_dict.items(), key=lambda x: x[1]["pred"], reverse=True)
    sorted_sell_hist = sorted(sell_hist_dict.items(), key=lambda x: x[1]["pred"], reverse=True)

    output("BUY")
    for stime, d in sorted_buy_hist:
        output(" pred:", d.get("pred"), " trade_sec:", d.get("trade_sec"), " spr_end:", d.get("spr_end"), " div300:",
               d.get("div300"), " div30:", d.get("div30"), " div3:", d.get("div3"), " high_profit:",
               d.get("high_profit"), " high_profit_sec:", d.get("high_profit_sec"))

    output("")
    output("SELL")
    for stime, d in sorted_sell_hist:
        output(" pred:", d.get("pred"), " trade_sec:", d.get("trade_sec"), " spr_end:", d.get("spr_end"), " div300:",
               d.get("div300"), " div30:", d.get("div30"), " div3:", d.get("div3"), " high_profit:",
               d.get("high_profit"), " high_profit_sec:", d.get("high_profit_sec"))


def showPipsPerDivABS(deal_hist, show_profit_per_div_list):
    range = 5
    for div in show_profit_per_div_list:
        output("")
        output("Div sec" + str(div))
        # start = -100
        start = 0

        while True:
            end = get_decimal_add(start, range)
            if end > 1000:
                break
            # print("start:", start)
            total = [d.get("profit_pips") for d in deal_hist if
                     abs(d.get("div" + str(div))) >= start and abs(d.get("div" + str(div))) < end]
            if len(total) != 0:
                output("Div:", start)
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                output("Total BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), " TOTAL_PIPS:", np.sum(total), )

                total = [d.get("profit_pips") for d in deal_hist if
                         abs(d.get("div" + str(div))) >= start and abs(d.get("div" + str(div))) < end and d.get(
                             "type") == "BUY"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    output("BUY BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                           " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), " TOTAL_PIPS:",
                           np.sum(total), )

                total = [d.get("profit_pips") for d in deal_hist if
                         abs(d.get("div" + str(div))) >= start and abs(d.get("div" + str(div))) < end and d.get(
                             "type") == "SELL"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    output("SELL BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                           " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total), " TOTAL_PIPS:",
                           np.sum(total), )

            start = get_decimal_add(start, range)


def showPipsPerDiv(deal_hist, show_profit_per_div_list, suffix_txt_tmp):
    range = 5
    for div in show_profit_per_div_list:
        suffix_txt_tmp.append("")
        suffix_txt_tmp.append("Div sec" + str(div))
        start = -1000

        while True:
            end = get_decimal_add(start, range)
            if end > 1000:
                break
            # print("start:", start)
            total = [d.get("profit_pips") for d in deal_hist if
                     d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end]
            if len(total) != 0:
                suffix_txt_tmp.append("Div:" + str(start))
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                suffix_txt_tmp.append("Total BET CNT:" + str(len(total)) + " CORRECT CNT:" + str(win_cnt) +
                                      " ACC:" + str(win_cnt / len(total)) + " AVG_PIPS:" + str(
                    np.average(total)) + " TOTAL_PIPS:" + str(np.sum(total)))

                total = [d.get("profit_pips") for d in deal_hist if
                         d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end and d.get("type") == "BUY"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    suffix_txt_tmp.append(list_to_str(("BUY BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                                                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total),
                                                       " TOTAL_PIPS:",
                                                       np.sum(total),), ""))

                total = [d.get("profit_pips") for d in deal_hist if
                         d.get("div" + str(div)) >= start and d.get("div" + str(div)) < end and d.get("type") == "SELL"]
                if len(total) != 0:
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    suffix_txt_tmp.append(list_to_str(("SELL BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                                                       " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total),
                                                       " TOTAL_PIPS:",
                                                       np.sum(total),), ""))

            start = get_decimal_add(start, range)

    return suffix_txt_tmp


def showPipsPerBoliStd(deal_hist, col_name, suffix_txt_tmp):
    range = 0.05

    suffix_txt_tmp.append("")
    suffix_txt_tmp.append(col_name)
    start = 0.0

    # 値がNoneのデータははじく
    deal_hist_new = []
    for d in deal_hist:
        boli_data = d.get(col_name)
        if boli_data != None:
            deal_hist_new.append(d)

    while True:
        end = get_decimal_add(start, range)
        if end > 100:
            break
        # print("start:", start)
        total = [d.get("profit_pips") for d in deal_hist_new if d.get(col_name) >= start and d.get(col_name) < end]
        if len(total) != 0:
            suffix_txt_tmp.append("STD:" + str(start) + "~" + str(end))
            win_cnt = len(np.where(np.array(total) >= 0)[0])
            suffix_txt_tmp.append("Total BET CNT:" + str(len(total)) + " CORRECT CNT:" + str(win_cnt) +
                                  " ACC:" + str(win_cnt / len(total)) + " AVG_PIPS:" + str(
                np.average(total)) + " TOTAL_PIPS:" + str(np.sum(total)))

            total = [d.get("profit_pips") for d in deal_hist if
                     d.get(col_name) >= start and d.get(col_name) < end and d.get("type") == "BUY"]
            if len(total) != 0:
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                suffix_txt_tmp.append(list_to_str(("BUY BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                                                   " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total),
                                                   " TOTAL_PIPS:", np.sum(total),), ""))

            total = [d.get("profit_pips") for d in deal_hist if
                     d.get(col_name) >= start and d.get(col_name) < end and d.get("type") == "SELL"]
            if len(total) != 0:
                win_cnt = len(np.where(np.array(total) >= 0)[0])
                suffix_txt_tmp.append(list_to_str(("SELL BET CNT:", len(total), " CORRECT CNT:", win_cnt,
                                                   " ACC:", win_cnt / len(total), " AVG_PIPS:", np.average(total),
                                                   " TOTAL_PIPS:", np.sum(total),), ""))

        start = get_decimal_add(start, range)

    return suffix_txt_tmp


def showPipsPerBoliStdDiv(deal_hist, col_name, suffix_txt_tmp):
    range = 5

    suffix_txt_tmp.append("")
    suffix_txt_tmp.append(col_name)

    # 値がNoneのデータははじく
    buy_hist = []
    sell_hist = []

    col_name_list = [col_name + "-MEAN-DIV", col_name + "-UP-DIV", col_name + "-DW-DIV"]

    for d in deal_hist:
        boli_data = d.get(col_name_list[0])
        if boli_data != None:
            if d.get("type") == "BUY":
                buy_hist.append(d)
            elif d.get("type") == "SELL":
                sell_hist.append(d)

    for col_name in col_name_list:
        suffix_txt_tmp.append("COL_NAME:" + col_name)

        if "-H1-" in col_name:
            range = 10
        else:
            range = 5

        for i, hist in enumerate([buy_hist, sell_hist]):
            suffix_txt_tmp.append("")
            if i == 0:
                suffix_txt_tmp.append("BUY:")
            else:
                suffix_txt_tmp.append("SELL:")

            suffix_txt_tmp.append("")

            start = -1000
            while True:
                end = get_decimal_add(start, range)
                if end > 1000:
                    break
                # print("start:", start)
                total = [d.get("profit_pips") for d in hist if d.get(col_name) >= start and d.get(col_name) < end]
                if len(total) != 0:
                    suffix_txt_tmp.append("DIV:" + str(start) + "~" + str(end))
                    win_cnt = len(np.where(np.array(total) >= 0)[0])
                    suffix_txt_tmp.append("Total BET CNT:" + str(len(total)) + " CORRECT CNT:" + str(win_cnt) +
                                          " ACC:" + str(win_cnt / len(total)) + " AVG_PIPS:" + str(
                        np.average(total)) + " TOTAL_PIPS:" + str(np.sum(total)))

                start = get_decimal_add(start, range)

    return suffix_txt_tmp


def showHighProfitDeal(deal_hist_dict):
    output("利益が多い取引")

    host = 'win5'
    db_no = 8
    db_name = 'USDJPY_60_MONEYPARTNERS_HISTORY'
    r = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

    limit_cnt = 300
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
            output("stime:", datetime.fromtimestamp(tmps.get("order_score")), "etime:",
                   datetime.fromtimestamp(tmps.get("deal_score")), "sprice:", tmps.get("start_rate"), "eprice:",
                   tmps.get("end_rate"),
                   "type:", tmps.get("sign"), " profit:", tmps.get("profit"),
                   )
            output("")
        cnt_t += 1


def get_list_lstm(conf, start, end, ):
    if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST) != 0:
        dataSequence2 = DataSequence2_from_pickle_test(conf, )
        dataSequence2.load_ds2()
    elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY) != 0:
        dataSequence2 = DataSequence2_from_pickle_test_on_memory(conf, )
        dataSequence2.load_ds2()
    elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_RAW) != 0:
        ds2_c = DataSequence2_from_pickle_test_raw(conf, )
        dataSequence2 = ds2_c.get_ds2()
    else:
        dataSequence2 = DataSequence2(conf, start, end, test_flg=True, eval_flg=False, return_all=False,
                                      redis_stop=False)

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

    if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST) != 0:
        output(datetime.now(), "before del ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        dataSequence2.del_ds2()
        output(datetime.now(), "after del ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY) != 0:
        output(datetime.now(), "before del ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        dataSequence2.del_ds2()
        output(datetime.now(), "after del ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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

    return test_lmd, test_conf, pred_close_list, close_list, score_list, spread_list, tick_list, target_score_list, train_list_index, atr_list, jpy_list, ind_list, \
           target_output_dict


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
                if isinstance(t_p, list) or isinstance(t_p, np.ndarray):
                    tmp_list.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                else:
                    tmp_list.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))
        else:
            if len(OUTPUT_LIST) == 1:
                for j, tmp_k in enumerate(OUTPUT_LIST):
                    for t_c, t_p in zip(target_output_dict_ext[tmp_k], predict_list_ext):

                        if isinstance(t_p, list) or isinstance(t_p, np.ndarray):
                            tmp_list.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                        else:
                            tmp_list.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))

            else:
                for j, tmp_k in enumerate(OUTPUT_LIST):
                    t_l = []
                    for t_c, t_p in zip(target_output_dict_ext[tmp_k], predict_list_ext[:, j]):
                        if isinstance(t_p, list) or isinstance(t_p, np.ndarray):
                            t_l.append(get_decimal_divide((t_c * ((t_p[0] / 10000) + 1)) - t_c, OUTPUT_MULTI))
                        else:
                            t_l.append(get_decimal_divide((t_c * ((t_p / 10000) + 1)) - t_c, OUTPUT_MULTI))

                    tmp_list.append(t_l)

        predict_list = np.array(tmp_list)

    elif OUTPUT_TYPE == "sub":
        # 現実のレートに換算する
        tmp_list = []
        for t_p in predict_list_ext:
            if isinstance(t_p, list) or isinstance(t_p, np.ndarray):
                tmp_list.append(get_decimal_divide(t_p[0], OUTPUT_MULTI))
            else:
                tmp_list.append(get_decimal_divide(t_p, OUTPUT_MULTI))
        predict_list = np.array(tmp_list)

    return predict_list


def get_score_pred_dict_ext_lstm(conf_ext, conf_lstm_dict, dataSequence2):
    FILE_PREFIX_EXT = conf_lstm_dict["FILE_PREFIX_EXT"]

    output("FILE_PREFIX_EXT:", FILE_PREFIX_EXT)

    if len(conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF"]) != 0:
        dataSequence2_ext = DataSequence2_from_pickle_test(conf_ext)
        dataSequence2_ext.load_ds2()
    elif len(conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF"]) != 0:
        dataSequence2_ext = DataSequence2_from_pickle_test_on_memory(conf_ext)
        dataSequence2_ext.load_ds2()
    else:
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
        predict_list_ext_up = get_predict(FILE_PREFIX_EXT[0], dataSequence2_ext, conf_ext)

        predict_list_ext_dw = get_predict(FILE_PREFIX_EXT[1], dataSequence2_ext, conf_ext)
        # SAMEの予想結果は0とする
        predict_list_ext_zero = np.zeros((len(predict_list_ext_up), 2))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_ext_up, predict_list_ext_zero, predict_list_ext_dw], 1)
        predict_list_ext = all[:, [0, 2, 4]]

    else:
        predict_list_ext = get_predict(FILE_PREFIX_EXT, dataSequence2_ext, conf_ext)

    # 現実レートに変換する
    if conf_lstm_dict["LEARNING_TYPE"] == "REGRESSION":
        reg_conf = conf_lstm_dict["REG_CONF"]
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

        predict_list_up = get_predict(FILE_PREFIX[0] + "-" + suffix[0], dataSequence2, conf)
        predict_list_dw = get_predict(FILE_PREFIX[1] + "-" + suffix[1], dataSequence2, conf)

        # SAMEの予想結果は0とする
        predict_list_zero = np.zeros((len(predict_list_up), 2))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
        predict_list = all[:, [0, 2, 4]]

    elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP":

        predict_list_up = get_predict(FILE_PREFIX + "-" + suffix, dataSequence2, conf)

        # DWの予想結果は0とする
        predict_list_dw = np.zeros((len(predict_list_up), 1))

        # UP,SAME,DWの予想結果を合算する
        predict_list = np.concatenate([predict_list_up, predict_list_dw], 1)
        # predict_list = all[:, [0, 2, 4]]
    elif conf.LEARNING_TYPE == "CATEGORY_BIN_DW":

        predict_list_dw = get_predict(FILE_PREFIX + "-" + suffix, dataSequence2, conf)

        # UPの予想結果は0とする
        predict_list_up = np.zeros((len(predict_list_dw), 1))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_up, predict_list_dw], 1)
        predict_list = all[:, [0, 2, 1]]
    else:
        predict_list = get_predict(FILE_PREFIX + "-" + suffix, dataSequence2, conf)

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
    elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP":

        predict_list_up = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX)

        # DWの予想結果は0とする
        predict_list_dw = np.zeros((len(predict_list_up), 1))

        # UP,SAME,DWの予想結果を合算する
        predict_list = np.concatenate([predict_list_up, predict_list_dw], 1)
        # predict_list = all[:, [0, 2, 4]]
    elif conf.LEARNING_TYPE == "CATEGORY_BIN_DW":

        predict_list_dw = lgb.Booster(model_file=conf.MODEL_DIR + FILE_PREFIX)

        # UPの予想結果は0とする
        predict_list_up = np.zeros((len(predict_list_dw), 1))

        # UP,SAME,DWの予想結果を合算する
        all = np.concatenate([predict_list_up, predict_list_dw], 1)
        predict_list = all[:, [0, 2, 1]]

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

def boli_ng_range_judge(conf, redis_db_boli, sc, close, boli_ng_range_dict_list):
    boli_db_name = conf.SYMBOL + "_1_0"
    boli_data = redis_db_boli.zrangebyscore(boli_db_name, sc - 1, sc - 1, withscores=True)
    boli_data_tmp = json.loads(boli_data[0][0])

    bet_flg = True

    for boli_ng_range_dict in boli_ng_range_dict_list:
        foot, length, alpha, type = boli_ng_range_dict["foot"].split("-")
        boli_max = boli_ng_range_dict["up"]
        boli_min = boli_ng_range_dict["dw"]

        col_name = "BOLI-" + foot + "-" + length + "-STD"
        std = boli_data_tmp.get(col_name)

        if std != None:
            mean = boli_data_tmp.get("BOLI-" + foot + "-" + length + "-MEAN")

            if type == "MEAN":
                target_div = get_divide(mean, close)
                if boli_min <= target_div and target_div <= boli_max:
                    # 範囲外なら取引しない
                    bet_flg = False

            elif type == "UP":
                boli = get_decimal_add(mean, get_decimal_multi(std, alpha))
                target_div = get_divide(boli, close)
                if boli_min <= target_div and target_div <= boli_max:
                    # 範囲外なら取引しない
                    bet_flg = False

            elif type == "DW":
                boli = get_decimal_sub(mean, get_decimal_multi(std, alpha))
                target_div = get_divide(boli, close)
                if boli_min <= target_div and target_div <= boli_max:
                    # 範囲外なら取引しない
                    bet_flg = False

    return bet_flg

def get_tpsl(sc, now_ask, now_bid, spr, takeprofit_dict, stoploss_dict, stoploss_max, mode, pending_pips):
    tk_type = None
    sl_type = None

    if takeprofit_dict != None:
        tk_type = takeprofit_dict["type"]
    if stoploss_dict != None:
        sl_type = stoploss_dict["type"]

    if tk_type == "std" or sl_type == 'std':
        redis_db_std = redis.Redis(host=conf.DB_HOST, port=6379, db=conf.DB_EVAL_NO, decode_responses=True)
        std_db_name = conf.SYMBOL + "_1_0"

        std_data = redis_db_std.zrangebyscore(std_db_name, sc - 1, sc - 1, withscores=True)
        std_data_tmp = json.loads(std_data[0][0])

    #takeprofitの値を求める
    if takeprofit_dict == None:
        x_std_buy_tp = None
        x_std_sell_tp = None
    else:
        if tk_type == 'fix':
            pips = takeprofit_dict["pips"]
            x_std_buy_tp = get_decimal_add(now_ask, pips)
            x_std_sell_tp = get_decimal_sub(now_bid, pips)

        elif tk_type == 'std':
            multi = takeprofit_dict["multi"]
            std = std_data_tmp.get(takeprofit_dict["std_name"])
            if std == None:
                std = takeprofit_dict["no_data_pips"]
                #print("std none:",sc, std_data_tmp)
            std = float(std)

            x_std_buy_tp = get_decimal_add(now_ask, std * multi)
            x_std_sell_tp = get_decimal_sub(now_bid, std * multi)

        if mode == "sashine":
            x_std_buy_tp = get_decimal_sub(x_std_buy_tp, pending_pips)
            x_std_sell_tp = get_decimal_add(x_std_sell_tp, pending_pips)
        elif mode == "gyaku_sashine":
            x_std_buy_tp = get_decimal_add(x_std_buy_tp, pending_pips)
            x_std_buy_sl = get_decimal_sub(x_std_sell_tp, pending_pips)

    #stoplossの値を求める
    if stoploss_dict == None:
        x_std_buy_sl = None
        x_std_buy_sl_max = None

        x_std_sell_sl = None
        x_std_sell_sl_max = None

    else:
        if sl_type == 'fix':
            pips = stoploss_dict["pips"]
            x_std_buy_sl = get_decimal_sub(now_ask, pips)
            x_std_sell_sl = get_decimal_add(now_bid, pips)

        elif sl_type == 'std':
            multi = stoploss_dict["multi"]
            std = std_data_tmp.get(stoploss_dict["std_name"])
            if std == None:
                std = stoploss_dict["no_data_pips"]
                #print("std none:",sc, std_data_tmp)
            std = float(std)

            x_std_buy_sl = get_decimal_sub(now_ask, std * multi)
            x_std_sell_sl = get_decimal_add(now_bid, std * multi)

        if mode == "sashine":
            x_std_buy_sl = get_decimal_sub(x_std_buy_sl, pending_pips)
            x_std_sell_sl = get_decimal_add(x_std_sell_sl, pending_pips)
        elif mode == "gyaku_sashine":
            x_std_buy_sl = get_decimal_add(x_std_buy_sl, pending_pips)
            x_std_sell_sl = get_decimal_sub(x_std_sell_sl, pending_pips)


        if stoploss_max != None:
            x_std_buy_sl_max = get_decimal_sub(now_ask, stoploss_max)
            x_std_sell_sl_max = get_decimal_add(now_bid, stoploss_max)

            if x_std_buy_sl_max > x_std_buy_sl:
                x_std_buy_sl = x_std_buy_sl_max

            if x_std_sell_sl_max < x_std_sell_sl:
                x_std_sell_sl = x_std_sell_sl_max
        else:
            x_std_buy_sl_max = x_std_buy_sl
            x_std_sell_sl_max = x_std_sell_sl

    return x_std_buy_tp, x_std_buy_sl, x_std_buy_sl_max, x_std_sell_tp, x_std_sell_sl, x_std_sell_sl_max


def do_predict(conf, start, end, model_file, spread_conf, mode_conf, div_conf, refer_dict, change_stoploss_conf,
               refer_past_pred_conf, other_conf, ):
    start_min_spread, start_max_spread, ex_min_spread, ex_max_spread, end_min_spread, end_max_spread, cannot_deal_cnt_max = spread_conf
    global c
    c = conf

    FILE_PREFIX = model_file[0]
    # FILE_PREFIX = "USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.5_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub3_MN714"

    output("FILE_PREFIX:", FILE_PREFIX)
    output("MODEL_NO:", model_file[1])

    if conf.CONF_TYPE == "LSTM":
        if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST) != 0:
            output("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST:", conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["score"])
        elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY) != 0:
            output("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY:",
                   conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY["score"])
        elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_RAW) != 0:
            output("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_RAW:", conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_RAW["score"])

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
        # [model_file[1], [0.46], 0.1, "", ],
        # [model_file[1], [0.47], 0.1, "", ],
        # [model_file[1], [0.48], 0.1, "", ],
        # [model_file[1], [0.49], 0.1, "", ],
        # [model_file[1], [0.5], 0.1, "", ],
        # [model_file[1], [0.51], 0.1, "", ],
        # [model_file[1], [0.52], 0.1, "", ],
        # [model_file[1], [0.53], 0.1, "", ],
        # [model_file[1], [0.54], 0.1, "", ],
        # [model_file[1], [0.55], 0.1, "", ],
        #[model_file[1], [0.56], 0.1, "", ],
        #[model_file[1], [0.57], 0.1, "", ],
        [model_file[1], [0.58], 0.1, "", ],
        #[model_file[1], [0.59], 0.1, "", ],
        # [model_file[1], [0.6], 0.1, "", ],
        # [model_file[1], [0.61], 0.1, "", ],
        # [model_file[1], [0.62], 0.1, "", ],
        # [model_file[1], [0.65], 0.4, "", ],
        # [model_file[1], [0.66], 0.4, "", ],
        # [model_file[1], [0.62], 0.4, "", ],

    ]

    # 特定のsuffixとborder_list, ext_borderを組みでテストする
    # model_suffix = [ ["32", [ 0.52, 0.54, 0.56, 0.58 ], 0.5],] #category_bin_both 以外
    """
    #category_bin_both 用
    model_suffix = [
        [["10", "31"], [[0.53, 0.53], ], [0.53, 0.53], ["",""]], 
    ]
    """

    # True:延長判定用モデルを使用する
    USE_EXT = False
    USE_EXT_CONF_TYPE = "LSTM"
    # USE_EXT_CONF_TYPE = "LGBM"

    # lstm用の設定
    conf_lstm_dict = {
        # 延長判定用モデル suffixまで含める USE_EXT=Trueの場合設定
        "FILE_PREFIX_EXT": "MN2047-175",

        # 延長判定用モデルのLEARNING_TYPE
        "LEARNING_TYPE": "CATEGORY",
        # regressionモデルの場合に予想を現実のレートに換算するための設定
        "REG_CONF": {
            "OUTPUT_DATA_BEF_C": False,
            "OUTPUT_TYPE": "d",
            "OUTPUT_MULTI": 1,
            "OUTPUT_LIST": "c",
        },

        # ext独自用のDataseqを使う場合:True Dataseqの設定自体はget_score_pred_dict_ext_lstm()内で記述する
        "USE_DATASEQ_EXT": False,

        # pickle保存されたDataseqを使う場合
        "DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF": {
            # "score": "173",
            # "save_dir_path": "/nvme1/dataSequence2/USDJPY/DS2F173-0",
        },
        # pickle保存されたDataseqを使う場合
        "DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF": {
            "score": "556",
            "save_dir_path": "/nvme2/dataSequence2/USDJPY/DS2F556-0",
        }
    }

    # lgbm用の設定
    conf_lgbm_dict = {
        # 予想元テストデータファイル
        "test_data_load_path": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF413.pickle",
        "conf_load_path": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF413-conf.pickle",

        # 延長判定用モデル USE_EXT=Trueの場合設定 category_bin_bothの場合リストにする
        "FILE_PREFIX_EXT": "MN963",
        # 延長判定用モデルのsuffix category_bin_bothの場合リストにする
        "FILE_PREFIX_EXT_SUFFIX": "1539",

        # 延長判定用モデルのINPUT_DATA
        # "INPUT_DATA": [],
        "INPUT_DATA": "922-32-DW@922-32-DW-12@922-32-DW-4@922-32-DW-8@922-32-SAME@922-32-SAME-12@922-32-SAME-4@922-32-SAME-8@922-32-UP@922-32-UP-12@922-32-UP-4@922-32-UP-8@928-32-DW@928-32-DW-12@928-32-DW-4@928-32-DW-8@928-32-SAME@928-32-SAME-12@928-32-SAME-4@928-32-SAME-8@928-32-UP@928-32-UP-12@928-32-UP-4@928-32-UP-8@929-10-REG@929-10-REG-12@929-10-REG-4@929-10-REG-8@959-26-DW@959-26-DW-12@959-26-DW-4@959-26-DW-8@959-26-SAME@959-26-SAME-12@959-26-SAME-4@959-26-SAME-8@959-26-UP@959-26-UP-12@959-26-UP-4@959-26-UP-8".split(
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
        "USE_DATASEQ_EXT": False,
        # 延長用予想元テストデータファイル
        "test_data_load_path_ext": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF999.pickle",
        "conf_load_path_ext": "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF999-conf.pickle",
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

    # スコアとレートを辞書で保持
    score_close_dict = dict(zip(score_list, close_list))

    # 長さチェック
    if len(score_list) != len(close_list) or len(score_list) != len(train_list_index) or len(score_list) != len(
            spread_list) or \
            len(score_list) != len(tick_list) or len(score_list) != len(atr_list) or len(score_list) != len(
        jpy_list) or len(jpy_list) != len(ind_list):
        print("list length is wrong!!!", len(score_list), len(close_list), len(train_list_index), len(spread_list),
              len(tick_list), len(atr_list), len(jpy_list), len(ind_list))
        exit(1)

    # mode:発注方法
    # market:成行き sashine:指値 gyaku_sashine:逆指値
    mode = mode_conf["mode"]
    pending_pips = mode_conf["pending_pips"]
    pending_max_sec = mode_conf["pending_max_sec"]

    # deal_mode:決済方法
    # market:成行き sashine:指値 gyaku_sashine:逆指値
    deal_mode = mode_conf["deal_mode"]
    deal_pending_pips = mode_conf["deal_pending_pips"]
    deal_pending_max_cnt = mode_conf["deal_pending_max_cnt"]

    output("mode:", mode)

    if mode != "market":
        output("pending_pips:", pending_pips)
        output("pending_max_sec:", pending_max_sec)

    output("deal_mode:", deal_mode)

    if deal_mode != "market":
        output("deal_pending_pips:", deal_pending_pips)
        output("deal_pending_max_cnt:", deal_pending_max_cnt)

    # suffixを変えて検証する対象 d(regression) or sub or category or category_bin or category_bin_both 検証対象外は固定のモデルを使用する

    if conf.LEARNING_TYPE == "REGRESSION":
        target = "d"
        target_ext = "d"
    else:
        target = "category"
        target_ext = "category"

    output("target:", target)
    output("target_ext:", target_ext)

    output(start, end)
    if conf.CONF_TYPE == "LGBM":
        output("test_data_load_path:", conf_lgbm_dict["test_data_load_path"])
    output("BET_TERM:", c.BET_TERM)
    output("TERM:", c.TERM)
    output("START_SEC:", c.START_TERM * c.BET_TERM)
    output("END_SEC:", c.END_TERM * c.BET_TERM)

    # ベット延長するか判断するTERM
    ext_term = c.TERM
    output("ext_term", ext_term)

    # ポジション数がこの数以下の場合に延長判断するtermをext_term_shortにする
    ext_term_short_position_num = None  # None:設定なし
    # ext_term_short_position_num = 2

    # ポジション数がext_term_short_position_num以下の場合に延長判断するterm
    ext_term_short = c.TERM
    if ext_term_short_position_num != None:
        output("ext_term_short_position_num", ext_term_short_position_num)
        output("ext_term_short", ext_term_short)

    # ベット延長判断開始秒数(ベットしてから何秒経過すれば延長判断するか)
    # 決済期間が30秒でも30未満を設定する場合は損切り判定の役割となる
    # 基本的に予想期間を設定
    ext_start_sec = c.TERM
    output("ext_start_sec", ext_start_sec)

    ext_start_sec_short_position_num = None  # None:設定なし
    # ext_start_sec_short_position_num = 6

    # ポジション数がext_term_short_position_num以下の場合に延長判断するterm
    ext_start_sec_short = c.TERM
    if ext_start_sec_short_position_num != None:
        output("ext_start_sec_short_position_num", ext_start_sec_short_position_num)
        output("ext_start_sec_short", ext_start_sec_short)

    cat_cond_ext2_flg = False  # 延長判断メソッド
    if cat_cond_ext2_flg:
        output("cat_cond_ext2_flg:", cat_cond_ext2_flg)

    refer_past_pred_sec = refer_past_pred_conf["refer_past_pred_sec"]  # 指定された過去秒の予想を参照。0なら参照しない
    refer_past_pred = refer_past_pred_conf["refer_past_pred"]  # 指定された過去秒の予想閾値

    output("refer_past_pred_sec:", refer_past_pred_sec)
    if refer_past_pred_sec != 0:
        output("refer_past_pred:", refer_past_pred)

    # 過去のティックデータを参照して、
    # 買いの場合は一定回数レート上がっていたらベットする
    # 売りの場合は一定回数レート下がっていたらベットする
    # 直前データがずっと下がっているのに買いにベットするのは避けたいため

    if refer_dict == None:
        refer_tick_sec = 0  # 0なら参照しない
        refer_cnt = 0
        refer_tick_sec_ext = 0  # 0なら参照しない
        refer_ext_cnt = 0
        refer_tick_cnt_0_ng = False
        refer_vs = False

    else:
        refer_tick_sec = refer_dict[
            "refer_tick_sec"]  # 0なら参照しない 過去いくつのループ分のティックデータを参照するか。ループが1秒間隔でrefer_tick_numが1なら過去1秒分参照する
        refer_tick_num = int(get_decimal_divide(refer_tick_sec, c.BET_TERM))
        refer_cnt = refer_dict["refer_cnt"]  # 最初のティックデータよりレートがベットする方に動いた回数

        refer_tick_sec_ext = refer_dict["refer_tick_sec_ext"]  # 延長判断するときに参考にする回数. 0なら参考にしない
        refer_tick_num_ext = int(get_decimal_divide(refer_tick_sec_ext, c.BET_TERM))
        refer_ext_cnt = refer_dict["refer_ext_cnt"]
        refer_tick_cnt_0_ng = refer_dict["refer_tick_cnt_0_ng"]
        refer_vs = refer_dict["refer_vs"]

    output("refer_tick_sec:", refer_tick_sec)
    output("refer_cnt:", refer_cnt)
    output("refer_tick_sec_ext:", refer_tick_sec_ext)
    output("refer_ext_cnt:", refer_ext_cnt)
    output("refer_tick_cnt_0_ng:", refer_tick_cnt_0_ng)
    output("refer_vs:", refer_vs)

    # 利益がマイナスであるかチェックし、マイナスなら決済する秒数
    minus_check_sec = 0  # 0の場合はチェックなし
    # minus_check_sec = 2 #0の場合はチェックなし
    if minus_check_sec != 0:
        output("minus_check_sec:", minus_check_sec)

    # Trueなら決済モードが逆指値でも、決済時に利益がマイナスである場合は成行決済とする
    deal_gyaku_sashine = False
    # deal_gyaku_sashine = True
    if deal_mode == "gyaku_sashine":
        output("deal_gyaku_sashine:", deal_gyaku_sashine)

    # スプレッドが指定値を超えた場合に成行決済する
    # deal_over_max_spread = 10 #None:設定なし
    deal_over_max_spread = None  # None:設定なし
    output("deal_over_max_spread:", deal_over_max_spread)

    # レート変動が少ない場合取引しない

    min_div = div_conf["min_div"]  # None:設定なし
    min_div_sec = div_conf["min_div_sec"]
    min_div_ext = div_conf["min_div_ext"]  # None:設定なし
    min_div_ext_sec = div_conf["min_div_ext_sec"]

    min_div_minus = div_conf["min_div_minus"]  # None:設定なし
    min_div_ext_minus = div_conf["min_div_ext_minus"]  # None:設定なし

    max_div = div_conf["max_div"]  # None:設定なし
    max_div_sec = div_conf["max_div_sec"]
    max_div_ext = div_conf["max_div_ext"]  # None:設定なし
    max_div_ext_sec = div_conf["max_div_ext_sec"]
    max_div_minus = div_conf["max_div_minus"]  # None:設定なし
    max_div_ext_minus = div_conf["max_div_ext_minus"]  # None:設定なし

    output("min_div:", min_div)
    output("min_div_ext:", min_div_ext)
    output("min_div_minus:", min_div_minus)
    output("min_div_ext_minus:", min_div_ext_minus)

    if min_div != None:
        output("min_div_sec:", min_div_sec)

    if min_div_ext != None:
        output("min_div_ext_sec:", min_div_ext_sec)

    output("max_div:", max_div)
    output("max_div_ext:", max_div_ext)
    output("max_div_minus:", max_div_minus)
    output("max_div_ext_minus:", max_div_ext_minus)

    if max_div != None:
        output("max_div_sec:", max_div_sec)

    if max_div_ext != None:
        output("max_div_ext_sec:", max_div_ext_sec)

    takeprofit_dict = other_conf['takeprofit_dict']  # None:設定なし
    output("takeprofit_dict:", takeprofit_dict)
    if takeprofit_dict != None:
        conf.FX_TAKE_PROFIT_FLG = True
    else:
        conf.FX_TAKE_PROFIT_FLG = False

    stoploss_dict = other_conf['stoploss_dict']  # None:設定なし
    output("stoploss_dict:", stoploss_dict)
    if stoploss_dict != None:
        conf.FX_STOP_LOSS_FLG = True
    else:
        conf.FX_STOP_LOSS_FLG = False

    stoploss_max = other_conf['stoploss_max']
    output("stoploss_max:", stoploss_max)

    stoploss_max_day = other_conf['stoploss_max_day']  # None:設定なし
    output("stoploss_max_day:", stoploss_max_day)

    if stoploss_max_day != None:
        conf.FX_FUND = stoploss_max_day * -1

    stoploss_break_recovery = other_conf['stoploss_break_recovery']  # None:設定なし
    output("stoploss_break_recovery:", stoploss_break_recovery)

    stoploss_trail = other_conf['stoploss_trail']
    output("stoploss_trail:", stoploss_trail)

    loss_cut_percent = other_conf['loss_cut_percent']
    output("loss_cut_percent:", loss_cut_percent)

    loss_cut_percent_day = other_conf['loss_cut_percent_day']
    output("loss_cut_percent_day:", loss_cut_percent_day)

    trial_trade_flg = other_conf['trial_trade_flg']
    output("trial_trade_flg:", trial_trade_flg)
    if trial_trade_flg:
        trial_trade_sec = other_conf['trial_trade_sec']
        output("trial_trade_sec:", trial_trade_sec)

        trial_trade_pips_min = other_conf['trial_trade_pips_min']
        output("trial_trade_pips_min:", trial_trade_pips_min)

        trial_trade_lookup_sec = other_conf['trial_trade_lookup_sec']
        output("trial_trade_lookup_sec:", trial_trade_lookup_sec)

        trial_trade_stoploss = other_conf['trial_trade_stoploss']
        output("trial_trade_stoploss:", trial_trade_stoploss)

        trial_trade_position = other_conf['trial_trade_position']
        output("trial_trade_position:", trial_trade_position)

        trial_trade_pips_update_sec = other_conf['trial_trade_pips_update_sec']
        output("trial_trade_pips_update_sec:", trial_trade_pips_update_sec)

        traial_trade_deal_reset = other_conf['traial_trade_deal_reset']
        output("traial_trade_deal_reset:", traial_trade_deal_reset)

    low_spread = other_conf['low_spread']
    output("low_spread:", low_spread)
    if low_spread != None:
        low_spread_border = other_conf['low_spread_border']
        output("low_spread_border:", low_spread_border)

    emerg_div = other_conf['emerg_div']
    output("emerg_div:", emerg_div)
    if emerg_div != None:
        emerg_div_sec = other_conf['emerg_div_sec']
        output("emerg_div_sec:", emerg_div_sec)

        emerg_stop_sec = other_conf['emerg_stop_sec']
        output("emerg_stop_sec:", emerg_stop_sec)

    most_high_low_div_flg = other_conf['most_high_low_div_flg']
    output("most_high_low_div_flg:", most_high_low_div_flg)

    stoploss_max_partial = other_conf['stoploss_max_partial']
    output("stoploss_max_partial:", stoploss_max_partial)
    if stoploss_max_partial != None:
        stoploss_break_recovery_partial = other_conf['stoploss_break_recovery_partial']
        output("stoploss_break_recovery_partial:", stoploss_break_recovery_partial)

    stoploss_short_sec = other_conf['stoploss_short_sec']
    output("stoploss_short_sec:", stoploss_short_sec)
    if stoploss_short_sec != None:
        stoploss_short_pips = other_conf['stoploss_short_pips']
        output("stoploss_short_pips:", stoploss_short_pips)

    # spread_listのスプレッドをすべて0にする
    ignore_spread = other_conf['ignore_spread']
    output("ignore_spread:", ignore_spread)

    boli_ng_range_buy = other_conf['boli_ng_range_buy']
    output("boli_ng_range_buy:", boli_ng_range_buy)

    boli_ng_range_sell = other_conf['boli_ng_range_sell']
    output("boli_ng_range_sell:", boli_ng_range_sell)

    stoploss_lookup_sec = other_conf['stoploss_lookup_sec']
    output("stoploss_lookup_sec:", stoploss_lookup_sec)

    output("POSITION_BY_PRED:", c.POSITION_BY_PRED)
    if c.POSITION_BY_PRED:
        output(c.POSITION_BY_PRED_LIST)

    # 全建玉の許容最大損失を下回ったら全建玉を決済する
    TOTAL_STOPLOSS = None  # None:設定なし
    # TOTAL_STOPLOSS = -0.001
    output("TOTAL_STOPLOSS", TOTAL_STOPLOSS)

    # 全建玉の許容最大損失を下回ったら新規発注しない
    ORDER_TOTAL_STOPLOSS = None  # None:設定なし
    # ORDER_TOTAL_STOPLOSS = -0.7
    output("ORDER_TOTAL_STOPLOSS", ORDER_TOTAL_STOPLOSS)

    # 途中で損切りラインを上げる
    CHANGE_STOPLOSS_TERM = change_stoploss_conf[
        "change_stoploss_term"]  # None:毎ループで損切りラインを上げるか判断 or 秒数:この秒数ごとに損切りラインを上げるか判断
    CHANGE_STOPLOSS_PRICE = change_stoploss_conf["change_stoploss_price"]  # 前回よりこの値だけ利益が出ていたら損切りラインをこの分だけ上げる
    output("CHANGE_STOPLOSS_TERM:", CHANGE_STOPLOSS_TERM)

    if CHANGE_STOPLOSS_TERM != None:
        output("CHANGE_STOPLOSS_PRICE:", CHANGE_STOPLOSS_PRICE)

    atr_range_base = []  # 絞り込むATR幅
    atr_range_ext = []  # 絞り込むATR幅(延長判断用)

    ind_range_bases = []
    ind_range_exts = []

    show_detail = True  # 詳細表示
    show_history = False  # 取引履歴を表示
    show_history_chart = False  # 取引履歴のチャートを/app/fx/chartに保存する
    show_plot = True
    show_profit_atr = False
    show_profit_ind = False
    show_profit_time = False
    show_position = False
    show_profit_per_spread = False
    show_profit_per_pred = False
    show_profit_per_trade_sec = False
    show_profit_per_div = False
    show_profit_per_div_abs = False
    show_profit_per_div_list = [300, ]
    show_high_profit_deal = False
    show_stoploss_history = False
    show_h1_sma_list = []
    # show_h1_sma_list = [25,75,200,300]

    show_profit_per_boli_std = []
    #show_profit_per_boli_std = ["M1-5-3", "M1-20-3", "H1-20-3",]
    show_profit_per_boli_std_div = []
    #show_profit_per_boli_std_div = ["M1-5-3", "M1-20-3", "H1-20-3", ]  # buyなら下のアルファ、sellなら上のアルファとのdivを取る
    show_profit_per_boli_std_div_history = False

    if show_plot:
        plot_close_list, plot_score_list = [], []
        tmp_start_score = start.timestamp()
        tmp_end_score = end.timestamp()
        for tmp_c, tmp_s in zip(close_list, score_list):
            if tmp_start_score <= tmp_s and tmp_s <= tmp_end_score:
                plot_close_list.append(tmp_c)
                plot_score_list.append(tmp_s)

    # 重要指標の時間帯を除外してテストする
    important_index_range = None  # 除外する前後の時間秒 Noneなら除外なし
    # important_index_range = 60  # 除外する前後の時間秒 Noneなら除外なし
    importance = "importances_high"
    nichiginOnly = False

    result_per_suffix_border = {}

    if show_history_chart:
        chart_save_dir = chart_dir + "/" + datetime.now().strftime('%Y%m%d-%H%M%S')
        makedirs(chart_save_dir)
        output("CHART SAVE DIR:", chart_save_dir)

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

    output("IND_COLS:", c.IND_COLS)
    output("ind_range_bases:", ind_range_bases)
    output("ind_range_exts:", ind_range_exts)

    output("ATR_COL:", c.ATR_COL)
    output("atr_range_base:", atr_range_base)
    output("atr_range_ext:", atr_range_ext)

    output("start_min_spread:", start_min_spread)
    output("start_max_spread:", start_max_spread)
    output("ex_min_spread:", ex_min_spread)
    output("ex_max_spread:", ex_max_spread)
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
        output("TPSL_ADJUST_PIPS:", conf.TPSL_ADJUST_PIPS)
    output("FX_BORDER_ATR:", c.FX_BORDER_ATR)
    output("FX_NOT_EXT_MINUS:", c.FX_NOT_EXT_MINUS)
    output("FX_MAX_TRADE_SEC:", c.FX_MAX_TRADE_SEC)

    output("EXCEPT_LIST_SEC_TEST:", conf.EXCEPT_LIST_SEC_TEST)
    output("EXCEPT_LIST_HOUR_TEST:", conf.EXCEPT_LIST_HOUR_TEST)

    output("RESTRICT_FLG:", c.RESTRICT_FLG)
    if c.RESTRICT_FLG:
        output("RESTRICT_SEC:", c.RESTRICT_SEC)
        output("RESTRICT_END_SEC:", c.RESTRICT_END_SEC)
    output("FX_MAX_POSITION_CNT:", c.FX_MAX_POSITION_CNT)

    output("SAME_SHIFT_NG_FLG:", c.SAME_SHIFT_NG_FLG)
    if c.SAME_SHIFT_NG_FLG:
        output("NG_SHIFT:", c.NG_SHIFT)
        output("NG_SHIFT_MAX:", c.NG_SHIFT_MAX)

    output("important_index_range:", important_index_range)
    if important_index_range != None:
        output("importance:", importance)
        output("nichiginOnly:", nichiginOnly)
    importantAnswer = ImportantIndex(importance=importance, range=important_index_range, startDt=start, endDt=end,
                                     nichiginOnly=nichiginOnly)
    # importantAnswer.print_index()

    output("USE_EXT:", USE_EXT)
    if show_profit_per_div or show_profit_per_div_abs:
        if show_profit_per_div:
            output("show_profit_per_div:", show_profit_per_div)
        if show_profit_per_div_abs:
            output("show_profit_per_div_abs:", show_profit_per_div_abs)
        output("show_profit_per_div_list:", show_profit_per_div_list)

    if len(show_h1_sma_list) != 0:
        output("show_h1_sma_list:", show_h1_sma_list)

        redis_db_m1 = redis.Redis(host='127.0.0.1', port=6379, db=1, decode_responses=True)
        db_name = 'USDJPY_H1'

        start_stp = int(time.mktime(start.timetuple()))
        end_stp = int(time.mktime(end.timetuple())) - 1

        result_data = redis_db_m1.zrangebyscore(db_name, start_stp, end_stp, withscores=True)

        m1_data = {}

        for i, line in enumerate(result_data):
            body = line[0]
            score = line[1]
            tmps = json.loads(body)
            m1_data[score] = tmps

    if len(show_profit_per_boli_std) != 0:
        output("show_profit_per_boli_std:", show_profit_per_boli_std)

    if len(show_profit_per_boli_std_div) != 0:
        output("show_profit_per_boli_std_div:", show_profit_per_boli_std_div)

    redis_db_boli = redis.Redis(host=conf.DB_HOST, port=6379, db=conf.DB_EVAL_NO, decode_responses=True)

    output("BINARY:", conf.BINARY)
    if conf.BINARY:
        output("PAYOUT:", conf.PAYOUT)
        output("PAYOFF:", conf.PAYOFF)
        output("AtMoney:", conf.AtMoney)
        if mode == 'gyaku_sashine':
            output("BINARY_GYAKUSASHINE_SURVIVE:", conf.BINARY_GYAKUSASHINE_SURVIVE)

    output("")

    # 延長判定用予想取得
    if USE_EXT:
        if USE_EXT_CONF_TYPE == "LSTM":
            if conf.CONF_TYPE == "LGBM":
                conf_ext = conf_class.ConfClass()
                dataSequence2 = None
            else:
                conf_ext = copy.deepcopy(conf)

            output("FILE_PREFIX_EXT:", conf_lstm_dict["FILE_PREFIX_EXT"])
            output("LEARNING_TYPE:", conf_lstm_dict["LEARNING_TYPE"])
            if conf_lstm_dict["LEARNING_TYPE"] == "REGRESSION":
                output("REG_CONF:", conf_lstm_dict["REG_CONF"])

            output("USE_DATASEQ_EXT:", conf_lstm_dict["USE_DATASEQ_EXT"])
            if len(conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF"]) != 0:
                output("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF:",
                       conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF"])
                conf_ext.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST = conf_lstm_dict[
                    "DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_EXT_CONF"]

            elif len(conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF"]) != 0:
                output("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF:",
                       conf_lstm_dict["DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF"])
                conf_ext.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY = conf_lstm_dict[
                    "DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY_EXT_CONF"]

            score_pred_dict_ext = get_score_pred_dict_ext_lstm(conf_ext, conf_lstm_dict, dataSequence2)

        elif USE_EXT_CONF_TYPE == "LGBM":
            if conf.CONF_TYPE == "LSTM":
                conf_ext = conf_class_lgbm.ConfClassLgbm()
            else:
                conf_ext = copy.deepcopy(conf)

            output("FILE_PREFIX_EXT:", conf_lgbm_dict["FILE_PREFIX_EXT"])
            output("FILE_PREFIX_EXT_SUFFIX:", conf_lgbm_dict["FILE_PREFIX_EXT_SUFFIX"])

            output("LEARNING_TYPE:", conf_lgbm_dict["LEARNING_TYPE"])
            if conf_lgbm_dict["LEARNING_TYPE"] == "REGRESSION":
                output("REG_CONF:", conf_lgbm_dict["REG_CONF"])

            output("USE_DATASEQ_EXT:", conf_lgbm_dict["USE_DATASEQ_EXT"])
            output("test_data_load_path_ext:", conf_lgbm_dict["test_data_load_path_ext"])
            output("conf_load_path_ext:", conf_lgbm_dict["conf_load_path_ext"])

            score_pred_dict_ext = get_score_pred_dict_ext_lgbm(conf_ext, conf_lgbm_dict, test_lmd, test_conf, )

    # 同じ予想結果を使いまわすための変数
    prev_suffix = None
    prev_list = None

    sub_txt = []

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

        score_pred_dict = dict(zip(target_score_list, predict_list))
        score_close_dict = dict(zip(score_list, close_list))

        for border in border_list:
            suffix_txt_tmp = []

            suffix_txt_tmp.append("")
            suffix_txt_tmp.append("suffix:" + str(suffix))

            if ext_border == None:
                ext_border = border
            suffix_txt_tmp.append("")
            suffix_txt_tmp.append("border:" + str(border))
            suffix_txt_tmp.append("ext_border:" + str(ext_border))
            suffix_txt_tmp.append("border_ceil:" + str(border_ceil))
            if conf.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                filename = save_dir + "/" + "SUFFIX_" + suffix[0] + "-" + suffix[1] + "_BORDER_" + str(
                    border[0]) + "-" + str(border[1]) + "_EXTBORDER_" + str(ext_border[0]) + "-" + str(
                    ext_border[1]) + "_BORDERCEIL_" + str(border_ceil[0]) + "-" + str(border_ceil[1]) + ".png"
            else:
                filename = save_dir + "/" + "SUFFIX_" + suffix + "_BORDER_" + str(border) + "_EXTBORDER_" + str(
                    ext_border) + "_BORDERCEIL_" + str(border_ceil) + ".png"
            max_drawdown = 0
            max_drawdown_sc = 0
            drawdown = 0
            max_drawdowns = {}

            profit_day = 0  # 一日ごとの損益
            profit_day_break = False  # 一日ごとの最大損失を超えた場合:True
            profit_days = {}
            profit_day_break_sc = None  # 一日ごとの最大損失を超えたscore

            profit_days_losscut = {}  # 保有ポジションがロスカットされた
            profit_days_losscut_day = {}  # 損失が一定以上になった

            profit_day_partial = 0  # stoploss_max_partial用の損益保持
            profit_day_break_partial = False  # stoploss_max_partialを超えた場合:True
            profit_day_break_partial_sc = None  # stoploss_max_partialを超えたscore
            profit_days_partial = {}  # 損失が一定以上になった

            pips = []
            pips_tp = []
            pips_sl = []
            pips_sps = []
            pips_up = []
            pips_dw = []

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
            # type:buy or sell,
            # sprice:bet時のレート,
            # pprice:約定時のレート,
            # pending_price:指値,成り行きの場合はNoneを入れる
            # stime:bet時の時間(score),指値の場合は注文を出した時間であり約定した時間ではない
            # ptime:約定した時間(score),成行の場合はstimeと同じ
            # len:betしている期間,
            # tp:takeprofit,
            # sl:stoploss
            # atr
            # spr:bet時のスプレッド

            bet_dicts = {}

            j = 0
            bet_cnt = 0
            tmp_bet_cnt = 0

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

            total_tick_mid_list = []
            total_tick_mid_list_ext = []

            trial_trade_start_sc = None
            trial_trade_pips = None
            trial_trade_type = None
            trial_trade_dict = {}

            stoploss_lookup_start_sc = None

            emerg_stop_start_sc = None

            wc = 0
            lc = 0

            suffix_txt_tmp.append(datetime.now().__str__() + " loop start")
            print(datetime.now().__str__(), "loop start")

            for cnt, (sc, close, idx, spr, tick, atr, jpy, ind) in enumerate(
                    zip(score_list, close_list, train_list_index, spread_list, tick_list, atr_list, jpy_list,
                        ind_list)):

                # 取引時間外になったら抜ける
                tmp_now_dt = datetime.fromtimestamp(sc)

                if tmp_now_dt < start:
                    continue

                if end < tmp_now_dt:
                    break

                """
                #for test_lstm_local
                if idx != -1:
                    print(sc,predict_list[idx])
                """

                if profit_day_break == True and stoploss_break_recovery != None:
                    if (profit_day_break_sc + stoploss_break_recovery) <= sc:
                        profit_day = 0  # リセット
                        profit_day_break = False  # リセット
                        profit_day_break_sc = None  # リセット

                if profit_day_break_partial == True and stoploss_break_recovery_partial != None:
                    if (profit_day_break_partial_sc + stoploss_break_recovery_partial) <= sc:
                        profit_day_partial = 0  # リセット
                        profit_day_break_partial = False  # リセット
                        profit_day_break_partial_sc = None  # リセット

                if tmp_now_dt.hour == 0 and tmp_now_dt.minute == 0 and tmp_now_dt.second == 0:
                    profit_day = 0  # リセット
                    profit_day_break = False  # リセット
                    profit_day_break_sc = None  # リセット

                    profit_day_partial = 0  # リセット
                    profit_day_break_partial = False  # リセット
                    profit_day_break_partial_sc = None  # リセット

                j += 1

                if ignore_spread:
                    spr = 0

                if c.BINARY:
                    spr = 0

                # 今のレート
                now_price = close

                tick_spr_list = tick.split(",")  # tickとsprが:区切り
                ask_list = []
                bid_list = []
                spr_tick_list = []
                tick_mid_list = []
                for tick_spr in tick_spr_list:
                    tc, tcsp = tick_spr.split(":")
                    tc = float(tc)
                    tcsp = float(tcsp)
                    if ignore_spread:
                        tcsp = 0
                    tmp_ask, tmp_bid = get_ask_bid(tc, tcsp, c.PIPS)
                    ask_list.append(tmp_ask)
                    bid_list.append(tmp_bid)
                    spr_tick_list.append(tcsp)
                    tick_mid_list.append(tc)

                if refer_tick_num != 0:
                    total_tick_mid_list.append(tick_mid_list)
                    while True:
                        if len(total_tick_mid_list) > refer_tick_num:
                            # 余分な過去分ティックデータを削除
                            total_tick_mid_list.pop(0)
                        else:
                            break

                    tick_up_cnt = 0
                    tick_dw_cnt = 0
                    tick_cnt = 0  # tickが動いた回数
                    fisrt_tick_child = None
                    prev_tick_child = None

                    for t_list in total_tick_mid_list:
                        for tick_child in t_list:

                            if fisrt_tick_child != None:
                                if fisrt_tick_child < tick_child:
                                    tick_up_cnt += 1
                                elif fisrt_tick_child > tick_child:
                                    tick_dw_cnt += 1
                            else:
                                fisrt_tick_child = tick_child

                            if prev_tick_child != None:
                                if prev_tick_child != tick_child:
                                    tick_cnt += 1

                            prev_tick_child = tick_child

                if refer_tick_num_ext != 0:
                    total_tick_mid_list_ext.append(tick_mid_list)
                    while True:
                        if len(total_tick_mid_list_ext) > refer_tick_num_ext:
                            # 余分な過去分ティックデータを削除
                            total_tick_mid_list_ext.pop(0)
                        else:
                            break

                    tick_up_cnt_ext = 0
                    tick_dw_cnt_ext = 0
                    tick_cnt_ext = 0  # tickが動いた回数
                    fisrt_tick_child = None
                    prev_tick_child = None

                    for t_list in total_tick_mid_list_ext:
                        for tick_child in t_list:
                            if fisrt_tick_child != None:
                                if fisrt_tick_child < tick_child:
                                    tick_up_cnt_ext += 1
                                elif fisrt_tick_child > tick_child:
                                    tick_dw_cnt_ext += 1
                            else:
                                fisrt_tick_child = tick_child

                            if prev_tick_child != None:
                                if prev_tick_child != tick_child:
                                    tick_cnt_ext += 1

                            prev_tick_child = tick_child

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

                if spr < 0:
                    # スプレッドがマイナスの場合はデータが続いていない場合なので無かったこととする
                    bet_dicts = {}

                    position_num = 0
                    position_num_tmp[sc] = position_num

                if prev_sc != None and get_decimal_sub(sc, prev_sc) > c.BET_TERM:
                    # データが続いていなければ無かったこととする
                    bet_dicts = {}

                    position_num = 0
                    position_num_tmp[sc] = position_num

                prev_sc = sc

                total_profit = 0  # 全建玉の利益

                # 現在の利益を計算
                for dict_key in bet_dicts.keys():
                    bet_dict = copy.deepcopy(bet_dicts[dict_key])
                    if bet_dict["ptime"] != None:  # 約定している場合
                        if bet_dict["type"] == "BUY":
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                tmp_profit_pips = now_bid - bet_dict["pprice"] - btcusd_spr
                            else:
                                tmp_profit_pips = now_bid - bet_dict["pprice"] + conf.ADJUST_PIPS

                        elif bet_dict["type"] == "SELL":
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                tmp_profit_pips = bet_dict["pprice"] - now_ask - btcusd_spr
                            else:
                                tmp_profit_pips = bet_dict["pprice"] - now_ask + conf.ADJUST_PIPS

                        if c.JPY_FLG == False:
                            tmp_profit = tmp_profit_pips * bet_dict["position"] * jpy
                        else:
                            tmp_profit = tmp_profit_pips * bet_dict["position"]

                        total_profit = total_profit + tmp_profit

                        if bet_dicts[dict_key]["high_profit"] < tmp_profit_pips:
                            bet_dicts[dict_key]["high_profit"] = tmp_profit_pips
                            bet_dicts[dict_key]["high_profit_sec"] = sc - bet_dicts[dict_key]["ptime"]

                        # trial_tradeが有効な場合、trial_tradeの現在の損益を保存
                        if trial_trade_flg and trial_trade_start_sc != None:
                            if dict_key == trial_trade_start_sc:
                                # if get_decimal_sub(sc, trial_trade_start_sc) <= trial_trade_sec:
                                #    trial_trade_pips = tmp_profit_pips
                                #    trial_trade_dict[trial_trade_start_sc] = trial_trade_pips trial_trade_lookup_sec

                                # if get_decimal_sub(sc, trial_trade_start_sc) <= trial_trade_sec:
                                if get_decimal_sub(sc, trial_trade_start_sc) <= trial_trade_pips_update_sec:

                                    if trial_trade_type == "BUY":
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                            trial_trade_pips = now_bid - bet_dict["pprice"] - btcusd_spr
                                        else:
                                            trial_trade_pips = now_bid - bet_dict["pprice"] + conf.ADJUST_PIPS

                                    elif trial_trade_type == "SELL":
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                            trial_trade_pips = bet_dict["pprice"] - now_ask - btcusd_spr
                                        else:
                                            trial_trade_pips = bet_dict["pprice"] - now_ask + conf.ADJUST_PIPS

                                    trial_trade_dict[trial_trade_start_sc] = trial_trade_pips

                if stoploss_max_day != None:
                    yuukou_shoukokin = (stoploss_max_day * -1) + profit_day + total_profit
                else:
                    yuukou_shoukokin = c.FX_FUND + profit_day + total_profit

                margin_total = 0
                for dict_key in bet_dicts.keys():
                    margin_total = margin_total + bet_dicts[dict_key]["margin"]

                margin_level = 0
                if margin_total != 0:
                    margin_level = yuukou_shoukokin / margin_total

                if profit_day_break_partial == False:
                    if stoploss_max_partial != None and (profit_day_partial + total_profit) < stoploss_max_partial:
                        # ロスカットに達したら、一定時間取引停止
                        all_deal_flg = True
                        profit_day_break_partial = True
                        profit_day_partial = profit_day_partial + total_profit
                        profit_days_partial[sc] = profit_day_partial
                        profit_day_break_partial_sc = sc

                if profit_day_break == False:

                    if c.BINARY:
                        if stoploss_max_day != None and loss_cut_percent_day != None and (
                                (stoploss_max_day * -1) + profit_day) / (stoploss_max_day * -1) <= loss_cut_percent_day:
                            # その日の損失がロスカットに達したら、その日は取引停止
                            all_deal_flg = True
                            profit_day_break = True
                            profit_day = profit_day
                            profit_days_losscut_day[sc] = profit_day
                            profit_day_break_sc = sc
                    else:
                        if stoploss_max_day != None and (profit_day + total_profit) < stoploss_max_day:
                            all_deal_flg = True
                            profit_day_break = True
                            # このループで決済する損失が一日の最大損失(ロスカット)を超えないように調整
                            total_profit = total_profit - ((profit_day + total_profit) - stoploss_max_day)
                            # print("total_profit:", total_profit, profit_day, datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'))
                            profit_day = stoploss_max_day
                            profit_days[sc] = profit_day
                            profit_day_break_sc = sc

                        elif loss_cut_percent != None and margin_total != 0 and margin_level <= loss_cut_percent:
                            # ロスカットに達したら、その日は取引停止
                            all_deal_flg = True
                            profit_day_break = True
                            profit_day = profit_day + total_profit
                            profit_days_losscut[sc] = profit_day
                            profit_day_break_sc = sc

                        elif loss_cut_percent_day != None and yuukou_shoukokin / (
                                stoploss_max_day * -1) <= loss_cut_percent_day:
                            # その日の損失がロスカットに達したら、その日は取引停止
                            all_deal_flg = True
                            profit_day_break = True
                            profit_day = profit_day + total_profit
                            profit_days_losscut_day[sc] = profit_day
                            profit_day_break_sc = sc

                if all_deal_flg == False:
                    # 今保持している建玉すべてでストップロス確認

                    BINARY_TRADE_FLG = False  # Binaryの逆指値が成立したらTrueとする 同時に1件のみ成立させるため

                    if c.FX_TAKE_PROFIT_FLG or c.FX_STOP_LOSS_FLG or TOTAL_STOPLOSS != None or ORDER_TOTAL_STOPLOSS != None:
                        del_key = []

                        for dict_key in bet_dicts.keys():

                            take_profit_flg = False
                            stop_loss_flg = False
                            bet_dict = copy.deepcopy(bet_dicts[dict_key])
                            stop_price = 0
                            tmp_profit = 0
                            stop_spr = 0
                            gyaku_sashine_sl_flg = False

                            if bet_dict["ptime"] == None and (mode == "sashine" or mode == "gyaku_sashine"):
                                # 約定してない場合　このtermで約定するか確認　またそのままtk,slするかも確認
                                for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list, spr_tick_list):
                                    if bet_dict["type"] == "BUY":
                                        if mode == "sashine":
                                            if bet_dicts[dict_key]["ptime"] == None and tick_ask <= bet_dict[
                                                "pending_price"]:
                                                bet_dicts[dict_key]["ptime"] = prev_sc
                                                bet_dicts[dict_key]["pprice"] = get_decimal_sub(
                                                    bet_dict["pending_price"], c.TPSL_ADJUST_PIPS)
                                                continue

                                        elif mode == "gyaku_sashine":
                                            if bet_dicts[dict_key]["ptime"] == None and tick_ask >= bet_dict[
                                                "pending_price"]:
                                                bet_dicts[dict_key]["ptime"] = prev_sc
                                                bet_dicts[dict_key]["pprice"] = tick_ask
                                                continue

                                        if bet_dicts[dict_key]["ptime"] != None and c.TP_SL_MODE == "auto":
                                            if c.FX_STOP_LOSS_FLG and tick_bid <= bet_dict["sl"]:
                                                stop_loss_flg = True
                                                stop_price = get_decimal_add(tick_bid, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dicts[dict_key]["pprice"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                                    tmp_profit = stop_price - bet_dicts[dict_key]["pprice"] - btcusd_spr
                                                else:
                                                    tmp_profit = stop_price - bet_dicts[dict_key][
                                                        "pprice"] + c.ADJUST_PIPS

                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"] * jpy
                                                else:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"]

                                                pips_sl.append(tmp_profit)
                                                break

                                            elif c.FX_TAKE_PROFIT_FLG and tick_bid >= bet_dict["tp"]:
                                                take_profit_flg = True
                                                stop_price = get_decimal_sub(tick_bid, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dicts[dict_key]["pprice"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                                    tmp_profit = stop_price - bet_dicts[dict_key]["pprice"] - btcusd_spr
                                                else:
                                                    tmp_profit = stop_price - bet_dicts[dict_key][
                                                        "pprice"] + c.ADJUST_PIPS

                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"] * jpy
                                                else:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"]
                                                pips_tp.append(tmp_profit)
                                                break

                                    elif bet_dict["type"] == "SELL":
                                        if mode == "sashine":
                                            if bet_dicts[dict_key]["ptime"] == None and tick_bid >= bet_dict[
                                                "pending_price"]:
                                                bet_dicts[dict_key]["ptime"] = prev_sc
                                                bet_dicts[dict_key]["pprice"] = get_decimal_add(
                                                    bet_dict["pending_price"],
                                                    c.TPSL_ADJUST_PIPS)
                                                continue
                                        elif mode == "gyaku_sashine":
                                            if bet_dicts[dict_key]["ptime"] == None and tick_bid <= bet_dict[
                                                "pending_price"]:
                                                bet_dicts[dict_key]["ptime"] = prev_sc
                                                bet_dicts[dict_key]["pprice"] = tick_bid
                                                continue

                                        if bet_dicts[dict_key]["ptime"] != None and c.TP_SL_MODE == "auto":
                                            if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= tick_ask:
                                                stop_loss_flg = True
                                                stop_price = get_decimal_sub(tick_ask, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dicts[dict_key]["pprice"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                                    tmp_profit = bet_dicts[dict_key]["pprice"] - stop_price - btcusd_spr
                                                else:
                                                    tmp_profit = bet_dicts[dict_key][
                                                                     "pprice"] - stop_price + c.ADJUST_PIPS

                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"] * jpy
                                                else:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"]
                                                pips_sl.append(tmp_profit)
                                                break
                                            elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= tick_ask:
                                                take_profit_flg = True
                                                stop_price = get_decimal_add(tick_ask, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick

                                                if conf.SYMBOL == "BTCUSD":
                                                    btcusd_spr = float(Decimal(str(bet_dicts[dict_key]["pprice"])) * (
                                                            Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                                    tmp_profit = bet_dicts[dict_key]["pprice"] - stop_price - btcusd_spr
                                                else:
                                                    tmp_profit = bet_dicts[dict_key][
                                                                     "pprice"] - stop_price + c.ADJUST_PIPS

                                                if c.JPY_FLG == False:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"] * jpy
                                                else:
                                                    profit = tmp_profit * bet_dicts[dict_key]["position"]
                                                pips_tp.append(tmp_profit)
                                                break

                            elif bet_dict["ptime"] != None:
                                # 約定している場合

                                if bet_dict["type"] == "BUY":
                                    if c.TP_SL_MODE == "auto":
                                        for tick_bid, spr_tick in zip(bid_list, spr_tick_list):
                                            if c.FX_STOP_LOSS_FLG and tick_bid <= bet_dict["sl"]:
                                                stop_loss_flg = True
                                                stop_price = get_decimal_add(tick_bid, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick
                                                break

                                            elif c.FX_TAKE_PROFIT_FLG and tick_bid >= bet_dict["tp"]:
                                                take_profit_flg = True
                                                stop_price = get_decimal_sub(tick_bid, c.TPSL_ADJUST_PIPS)
                                                break

                                            if stoploss_short_sec != None and get_decimal_sub(sc, bet_dict[
                                                "ptime"]) == stoploss_short_sec:
                                                # 指定秒数経過時に損失が一定以上の場合は損切りする
                                                if get_decimal_sub(tick_bid, bet_dict["pprice"]) <= stoploss_short_pips:
                                                    stop_loss_flg = True
                                                    stop_price = get_decimal_add(tick_bid, c.TPSL_ADJUST_PIPS)
                                                    stop_spr = spr_tick
                                                    break

                                    elif c.TP_SL_MODE == "manual":
                                        passed_sc = get_decimal_sub(sc, bet_dict["ptime"])
                                        if get_decimal_mod(passed_sc, c.TP_SL_MANUAL_TERM) == 0:
                                            if c.FX_STOP_LOSS_FLG and now_bid <= bet_dict["sl"]:
                                                stop_loss_flg = True
                                                stop_spr = spr

                                                stop_price = now_bid
                                            elif c.FX_TAKE_PROFIT_FLG and now_bid >= bet_dict["tp"]:
                                                take_profit_flg = True
                                                stop_price = now_bid
                                                stop_spr = spr

                                    if stop_loss_flg == False:
                                        # 決済方法がgyaku_sashine_touch or gyaku_sashine_lastの場合は直前のtermのtickで成行決済するか決める
                                        if bet_dict.get("gyaku_sashine_sl") != None:
                                            if deal_mode == "gyaku_sashine_touch":
                                                touch_flg = False
                                                for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list,
                                                                                        spr_tick_list):
                                                    if tick_bid <= bet_dict["gyaku_sashine_sl"]:
                                                        touch_flg = True
                                                        break
                                                if touch_flg:
                                                    stop_loss_flg = True
                                                    stop_spr = spr
                                                    stop_price = now_bid
                                                    gyaku_sashine_sl_flg = True

                                            elif deal_mode == "gyaku_sashine_last":
                                                if now_bid <= bet_dict["gyaku_sashine_sl"]:
                                                    stop_loss_flg = True
                                                    stop_spr = spr
                                                    stop_price = now_bid
                                                    gyaku_sashine_sl_flg = True

                                    if stop_loss_flg or take_profit_flg:
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                            tmp_profit = stop_price - bet_dict["pprice"] - btcusd_spr
                                        else:
                                            tmp_profit = stop_price - bet_dict["pprice"] + c.ADJUST_PIPS

                                        if c.JPY_FLG == False:
                                            profit = tmp_profit * bet_dict["position"] * jpy
                                        else:
                                            profit = tmp_profit * bet_dict["position"]

                                        if stop_loss_flg and gyaku_sashine_sl_flg == False:
                                            pips_sl.append(tmp_profit)
                                        elif take_profit_flg:
                                            pips_tp.append(tmp_profit)

                                elif bet_dict["type"] == "SELL":
                                    if c.TP_SL_MODE == "auto":
                                        for tick_ask, spr_tick in zip(ask_list, spr_tick_list):
                                            if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= tick_ask:
                                                stop_loss_flg = True
                                                stop_price = get_decimal_sub(tick_ask, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick

                                                break

                                            elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= tick_ask:
                                                take_profit_flg = True
                                                stop_price = get_decimal_add(tick_ask, c.TPSL_ADJUST_PIPS)
                                                stop_spr = spr_tick
                                                break

                                            if stoploss_short_sec != None and get_decimal_sub(sc, bet_dict[
                                                "ptime"]) == stoploss_short_sec:
                                                # 指定秒数経過時に損失が一定以上の場合は損切りする
                                                if get_decimal_sub(bet_dict["pprice"], tick_ask) <= stoploss_short_pips:
                                                    stop_loss_flg = True
                                                    stop_price = get_decimal_add(tick_ask, c.TPSL_ADJUST_PIPS)
                                                    stop_spr = spr_tick
                                                    break

                                    elif c.TP_SL_MODE == "manual":
                                        passed_sc = get_decimal_sub(sc, bet_dict["ptime"])
                                        if get_decimal_mod(passed_sc, c.TP_SL_MANUAL_TERM) == 0:
                                            if c.FX_STOP_LOSS_FLG and bet_dict["sl"] <= now_ask:
                                                stop_loss_flg = True
                                                stop_price = now_ask
                                                stop_spr = spr
                                            elif c.FX_TAKE_PROFIT_FLG and bet_dict["tp"] >= now_ask:
                                                take_profit_flg = True
                                                stop_price = now_ask
                                                stop_spr = spr

                                    if stop_loss_flg == False:
                                        # 決済方法がgyaku_sashine_touch or gyaku_sashine_lastの場合は直前のtermのtickで成行決済するか決める
                                        if bet_dict.get("gyaku_sashine_sl") != None:
                                            if deal_mode == "gyaku_sashine_touch":
                                                touch_flg = False
                                                for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list,
                                                                                        spr_tick_list):
                                                    if tick_ask >= bet_dict["gyaku_sashine_sl"]:
                                                        touch_flg = True
                                                        break
                                                if touch_flg:
                                                    stop_loss_flg = True
                                                    stop_spr = spr
                                                    stop_price = now_ask
                                                    gyaku_sashine_sl_flg = True

                                            elif deal_mode == "gyaku_sashine_last":
                                                if now_ask >= bet_dict["gyaku_sashine_sl"]:
                                                    stop_loss_flg = True
                                                    stop_spr = spr
                                                    stop_price = now_ask
                                                    gyaku_sashine_sl_flg = True

                                    if stop_loss_flg or take_profit_flg:
                                        if conf.SYMBOL == "BTCUSD":
                                            btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                    Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                            tmp_profit = bet_dict["pprice"] - stop_price - btcusd_spr
                                        else:
                                            tmp_profit = bet_dict["pprice"] - stop_price + c.ADJUST_PIPS

                                        if c.JPY_FLG == False:
                                            profit = tmp_profit * bet_dict["position"] * jpy
                                        else:
                                            profit = tmp_profit * bet_dict["position"]

                                        if stop_loss_flg and gyaku_sashine_sl_flg == False:
                                            pips_sl.append(tmp_profit)
                                        elif take_profit_flg:
                                            pips_tp.append(tmp_profit)

                                else:
                                    # 想定外エラー
                                    print("ERROR2")
                                    sys.exit()

                            if stop_loss_flg or take_profit_flg:
                                prev_bet_end_score = sc

                                pips.append(tmp_profit)
                                if bet_dict["type"] == "BUY":
                                    pips_up.append(tmp_profit)
                                elif bet_dict["type"] == "SELL":
                                    pips_dw.append(tmp_profit)

                                atrs.append(bet_dict["atr"])
                                inds.append(bet_dict["ind"])
                                times.append(bet_dicts[dict_key]["ptime"])
                                spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], tmp_profit, profit])

                                profit_day = profit_day + profit
                                profit_day_partial = profit_day_partial + profit

                                money = money + profit
                                max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown,
                                                                                        drawdown, profit, sc,
                                                                                        max_drawdown_sc)

                                money_tmp[sc] = money
                                position_num = position_num - 1
                                position_num_tmp[sc] = position_num

                                hist_child = {"type": bet_dict["type"],
                                              "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                                  '%Y/%m/%d %H:%M:%S'),
                                              "ptime": datetime.fromtimestamp(bet_dicts[dict_key]["ptime"]).strftime(
                                                  '%Y/%m/%d %H:%M:%S'),
                                              "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                              "sprice": bet_dict["sprice"],
                                              "pprice": bet_dicts[dict_key]["pprice"],
                                              "eprice": stop_price,
                                              "profit_pips": tmp_profit,
                                              "profit": profit,
                                              "score": bet_dict["stime"],
                                              "spr": bet_dict["spr"],
                                              "spr_end": stop_spr,
                                              "trade_sec": sc - bet_dicts[dict_key]["ptime"],
                                              "stop_loss": stop_loss_flg and (gyaku_sashine_sl_flg == False),
                                              "take_profit": take_profit_flg,
                                              "past_pred_1": bet_dicts[dict_key]["past_pred_1"],
                                              "past_close_move_1": bet_dicts[dict_key]["past_close_move_1"],
                                              "high_profit": bet_dicts[dict_key]["high_profit"],
                                              "high_profit_sec": bet_dicts[dict_key]["high_profit_sec"],
                                              }
                                # print("error:", hist_child)
                                if target == "category":
                                    hist_child["pred"] = bet_dict["pred"]

                                if show_profit_per_div:
                                    for d in show_profit_per_div_list:
                                        key_str = "div" + str(d)
                                        hist_child[key_str] = bet_dict[key_str]

                                for m in show_h1_sma_list:
                                    key_str = "sma-" + str(m)
                                    hist_child[key_str] = bet_dict[key_str]

                                for b in show_profit_per_boli_std:
                                    foot, length, alpha = b.split("-")
                                    key_str = "BOLI-" + foot + "-" + length + "-STD" + alpha
                                    hist_child[key_str] = bet_dict[key_str]

                                for b in show_profit_per_boli_std_div:
                                    foot, length, alpha = b.split("-")
                                    key_str = "BOLI-" + foot + "-" + length + "-STD"

                                    hist_child[key_str] = bet_dict[key_str]
                                    hist_child[key_str + alpha + "-UP-DIV"] = bet_dict[key_str + alpha + "-UP-DIV"]
                                    hist_child[key_str + alpha + "-DW-DIV"] = bet_dict[key_str + alpha + "-DW-DIV"]
                                    hist_child[key_str + alpha + "-MEAN-DIV"] = bet_dict[key_str + alpha + "-MEAN-DIV"]

                                if bet_dicts[dict_key]["high_profit"] < tmp_profit:
                                    hist_child["high_profit"] = tmp_profit
                                    hist_child["high_profit_sec"] = sc - bet_dicts[dict_key]["ptime"]

                                if stop_loss_flg and stoploss_lookup_sec != None:
                                    stoploss_lookup_start_sc = sc

                                deal_hist_dict[hist_child["score"]] = hist_child
                                deal_hist.append(hist_child)
                                del_key.append(dict_key)

                        for dkey in del_key:
                            del bet_dicts[dkey]

                    else:
                        # stoploss,takeprofitの設定なしの場合
                        if mode == "sashine" or mode == "gyaku_sashine":

                            # 取引開始順に並びなおす(Binaryの逆指値のため ベットしたのが早い順に1件のみ成立させる)
                            bet_dicts_sorted = sorted(bet_dicts.items(), key=lambda x: x[0])
                            for dict_key, dict_body in bet_dicts_sorted:
                                # for dict_key in bet_dicts.keys():

                                bet_dict = copy.deepcopy(bet_dicts[dict_key])

                                if bet_dict["ptime"] == None:
                                    # 約定してない場合　このtermで約定するか確認
                                    for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list, spr_tick_list):
                                        if bet_dict["type"] == "BUY":
                                            if mode == "sashine":
                                                if bet_dicts[dict_key]["ptime"] == None and tick_ask <= bet_dict[
                                                    "pending_price"]:
                                                    bet_dicts[dict_key]["ptime"] = prev_sc
                                                    bet_dicts[dict_key]["pprice"] = get_decimal_sub(
                                                        bet_dict["pending_price"], c.TPSL_ADJUST_PIPS)
                                                    bet_dicts[dict_key]["spr"] = spr_tick
                                                    break
                                            elif mode == "gyaku_sashine":
                                                if bet_dicts[dict_key]["ptime"] == None and tick_ask >= bet_dict[
                                                    "pending_price"]:

                                                    if conf.BINARY:
                                                        if BINARY_TRADE_FLG == False:
                                                            BINARY_TRADE_FLG = True
                                                            bet_dicts[dict_key]["stime"] = sc
                                                            bet_dicts[dict_key]["ptime"] = sc
                                                            bet_dicts[dict_key]["pprice"] = now_ask
                                                        else:
                                                            if conf.BINARY_GYAKUSASHINE_SURVIVE == False:
                                                                # 既に同時に逆指値が成立している建玉がある場合はこの建玉をキャンセルする
                                                                del bet_dicts[dict_key]
                                                    else:
                                                        bet_dicts[dict_key]["ptime"] = prev_sc
                                                        bet_dicts[dict_key]["pprice"] = tick_ask
                                                        bet_dicts[dict_key]["spr"] = spr_tick

                                                    break

                                        elif bet_dict["type"] == "SELL":
                                            if mode == "sashine":
                                                if bet_dicts[dict_key]["ptime"] == None and tick_bid >= bet_dict[
                                                    "pending_price"]:
                                                    bet_dicts[dict_key]["ptime"] = prev_sc
                                                    bet_dicts[dict_key]["pprice"] = get_decimal_add(
                                                        bet_dict["pending_price"], c.TPSL_ADJUST_PIPS)
                                                    bet_dicts[dict_key]["spr"] = spr_tick
                                                    break
                                            elif mode == "gyaku_sashine":
                                                if bet_dicts[dict_key]["ptime"] == None and tick_bid <= bet_dict[
                                                    "pending_price"]:
                                                    if conf.BINARY:
                                                        if BINARY_TRADE_FLG == False:
                                                            BINARY_TRADE_FLG = True
                                                            bet_dicts[dict_key]["stime"] = sc
                                                            bet_dicts[dict_key]["ptime"] = sc
                                                            bet_dicts[dict_key]["pprice"] = now_bid
                                                        else:
                                                            if conf.BINARY_GYAKUSASHINE_SURVIVE == False:
                                                                # 既に同時に逆指値が成立している建玉がある場合はこの建玉をキャンセルする
                                                                del bet_dicts[dict_key]
                                                    else:
                                                        bet_dicts[dict_key]["ptime"] = prev_sc
                                                        bet_dicts[dict_key]["pprice"] = tick_bid
                                                        bet_dicts[dict_key]["spr"] = spr_tick
                                                    break

                # 全建玉の許容損失を超えたら全建玉を決済する
                if TOTAL_STOPLOSS != None and TOTAL_STOPLOSS >= total_profit:
                    all_deal_flg = True

                # ATRが突然上がった場合全て決済
                if (c.FX_BORDER_ATR != None and atr != None and c.FX_BORDER_ATR <= atr):
                    all_deal_flg = True

                if deal_over_max_spread != None and deal_over_max_spread <= spr:
                    all_deal_flg = True

                if importantAnswer.is_except(sc):
                    # print("重要指標発表時は延長しない", sc)
                    all_deal_flg = True

                if emerg_div != None:
                    if emerg_stop_start_sc == None or get_decimal_sub(sc, emerg_stop_start_sc) > emerg_stop_sec:
                        tmp_bef = close_list[cnt - int(get_decimal_divide(emerg_div_sec, c.BET_TERM))]
                        tmp_div = abs(get_divide(tmp_bef, close))
                        if emerg_div <= tmp_div:
                            all_deal_flg = True
                            emerg_stop_start_sc = sc

                if all_deal_flg:
                    # 全決済の場合はdeal_modeに関わらず成行で決済する
                    prev_bet_end_score = sc
                    del_key = []

                    if stoploss_max_day != None and profit_day_break == True:
                        # 一日の最大損失を超えている場合
                        money = money + total_profit
                        # print("max_drawdown:", max_drawdown, "drawdown:",drawdown, "total_profit:", total_profit, datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'))
                        max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown, drawdown,
                                                                                total_profit, sc, max_drawdown_sc)
                        # print("max_drawdown:", max_drawdown, "drawdown:",drawdown, "total_profit:", total_profit, datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'))
                        money_tmp[sc] = money

                    for dict_key in bet_dicts.keys():
                        del_key.append(dict_key)

                        bet_dict = copy.deepcopy(bet_dicts[dict_key])

                        if bet_dict["ptime"] == None:
                            continue

                        if bet_dict["type"] == "BUY":
                            # 決済する
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                profit_pips = get_decimal_sub(deal_bid, bet_dict["pprice"]) - btcusd_spr
                            else:
                                try:
                                    profit_pips = get_decimal_sub(deal_bid, bet_dict["pprice"]) + c.ADJUST_PIPS
                                except Exception as e:

                                    print("deal_bid:", deal_bid)
                                    print(bet_dict)
                                    print(tracebackPrint(e))

                            pips.append(profit_pips)
                            pips_up.append(profit_pips)

                            atrs.append(bet_dict["atr"])
                            inds.append(bet_dict["ind"])
                            times.append(bet_dict["ptime"])

                            if c.BINARY == True:
                                if 0.0 < profit_pips:
                                    profit = c.PAYOUT
                                    wc += 1
                                elif 0.0 == profit_pips and c.AtMoney:
                                    profit = 0
                                else:
                                    profit = c.PAYOFF
                                    lc += 1
                            else:
                                if c.JPY_FLG == False:
                                    profit = profit_pips * bet_dict["position"] * jpy
                                else:
                                    profit = profit_pips * bet_dict["position"]
                            spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                            stop_price = deal_bid

                            if profit_day_break == False:
                                money = money + profit
                                max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown,
                                                                                        drawdown, profit, sc,
                                                                                        max_drawdown_sc)
                                money_tmp[sc] = money

                                profit_day = profit_day + profit

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "ptime": datetime.fromtimestamp(bet_dict["ptime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["sprice"],
                                          "pprice": bet_dict["pprice"],
                                          "eprice": stop_price,
                                          "profit_pips": profit_pips,
                                          "profit": profit,
                                          "spr": bet_dict["spr"],
                                          "spr_end": spr,
                                          "score": bet_dict["stime"],
                                          "trade_sec": sc - bet_dict["ptime"],
                                          "stop_loss": False,
                                          "take_profit": False,
                                          "past_pred_1": bet_dict["past_pred_1"],
                                          "past_close_move_1": bet_dict["past_close_move_1"],
                                          "high_profit": bet_dict["high_profit"],
                                          "high_profit_sec": bet_dict["high_profit_sec"],
                                          }

                            if target == "category":
                                hist_child["pred"] = bet_dict["pred"]

                            if show_profit_per_div:
                                for d in show_profit_per_div_list:
                                    key_str = "div" + str(d)
                                    hist_child[key_str] = bet_dict[key_str]

                            for m in show_h1_sma_list:
                                key_str = "sma-" + str(m)
                                hist_child[key_str] = bet_dict[key_str]

                            for b in show_profit_per_boli_std:
                                foot, length, alpha = b.split("-")
                                key_str = "BOLI-" + foot + "-" + length + "-STD" + alpha
                                hist_child[key_str] = bet_dict[key_str]

                            for b in show_profit_per_boli_std_div:
                                foot, length, alpha = b.split("-")
                                key_str = "BOLI-" + foot + "-" + length + "-STD"

                                hist_child[key_str] = bet_dict[key_str]
                                hist_child[key_str + alpha + "-UP-DIV"] = bet_dict[key_str + alpha + "-UP-DIV"]
                                hist_child[key_str + alpha + "-DW-DIV"] = bet_dict[key_str + alpha + "-DW-DIV"]
                                hist_child[key_str + alpha + "-MEAN-DIV"] = bet_dict[key_str + alpha + "-MEAN-DIV"]

                            if bet_dict["high_profit"] < profit_pips:
                                hist_child["high_profit"] = profit_pips
                                hist_child["high_profit_sec"] = sc - bet_dict["ptime"]

                            if trial_trade_flg and trial_trade_start_sc != None:
                                if bet_dict["ptime"] == trial_trade_start_sc and traial_trade_deal_reset:
                                    #トライアルポジションの場合は決済時にリセットする
                                    trial_trade_start_sc = None

                            deal_hist_dict[hist_child["score"]] = hist_child
                            deal_hist.append(hist_child)

                        elif bet_dict["type"] == "SELL":
                            if conf.SYMBOL == "BTCUSD":
                                btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                        Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                profit_pips = (get_decimal_sub(deal_ask, bet_dict["pprice"]) * -1) - btcusd_spr
                            else:
                                profit_pips = (get_decimal_sub(deal_ask, bet_dict["pprice"]) * -1) + c.ADJUST_PIPS

                            pips.append(profit_pips)
                            pips_dw.append(profit_pips)

                            atrs.append(bet_dict["atr"])
                            inds.append(bet_dict["ind"])
                            times.append(bet_dict["ptime"])
                            if c.BINARY == True:
                                if 0.0 < profit_pips:
                                    profit = c.PAYOUT
                                    wc += 1
                                elif 0.0 == profit_pips and c.AtMoney:
                                    profit = 0
                                else:
                                    profit = c.PAYOFF
                                    lc += 1
                            else:
                                if c.JPY_FLG == False:
                                    profit = profit_pips * bet_dict["position"] * jpy
                                else:
                                    profit = profit_pips * bet_dict["position"]
                                spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                            stop_price = deal_ask
                            if profit_day_break == False:
                                money = money + profit
                                max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown,
                                                                                        drawdown, profit, sc,
                                                                                        max_drawdown_sc)
                                money_tmp[sc] = money

                                profit_day = profit_day + profit

                            hist_child = {"type": bet_dict["type"],
                                          "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "ptime": datetime.fromtimestamp(bet_dict["ptime"]).strftime(
                                              '%Y/%m/%d %H:%M:%S'),
                                          "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                          "sprice": bet_dict["sprice"],
                                          "pprice": bet_dict["pprice"],
                                          "eprice": stop_price,
                                          "profit_pips": profit_pips,
                                          "profit": profit,
                                          "spr": bet_dict["spr"],
                                          "spr_end": spr,
                                          "score": bet_dict["stime"],
                                          "trade_sec": sc - bet_dict["ptime"],
                                          "stop_loss": False,
                                          "take_profit": False,
                                          "past_pred_1": bet_dict["past_pred_1"],
                                          "past_close_move_1": bet_dict["past_close_move_1"],
                                          "high_profit": bet_dict["high_profit"],
                                          "high_profit_sec": bet_dict["high_profit_sec"],
                                          }

                            if target == "category":
                                hist_child["pred"] = bet_dict["pred"]

                            if show_profit_per_div:
                                for d in show_profit_per_div_list:
                                    key_str = "div" + str(d)
                                    hist_child[key_str] = bet_dict[key_str]

                            for m in show_h1_sma_list:
                                key_str = "sma-" + str(m)
                                hist_child[key_str] = bet_dict[key_str]

                            for b in show_profit_per_boli_std:
                                foot, length, alpha = b.split("-")
                                key_str = "BOLI-" + foot + "-" + length + "-STD" + alpha
                                hist_child[key_str] = bet_dict[key_str]

                            for b in show_profit_per_boli_std_div:
                                foot, length, alpha = b.split("-")
                                key_str = "BOLI-" + foot + "-" + length + "-STD"

                                hist_child[key_str] = bet_dict[key_str]
                                hist_child[key_str + alpha + "-UP-DIV"] = bet_dict[key_str + alpha + "-UP-DIV"]
                                hist_child[key_str + alpha + "-DW-DIV"] = bet_dict[key_str + alpha + "-DW-DIV"]
                                hist_child[key_str + alpha + "-MEAN-DIV"] = bet_dict[key_str + alpha + "-MEAN-DIV"]

                            if bet_dict["high_profit"] < profit_pips:
                                hist_child["high_profit"] = profit_pips
                                hist_child["high_profit_sec"] = sc - bet_dict["ptime"]

                            if trial_trade_flg and trial_trade_start_sc != None:
                                if bet_dict["ptime"] == trial_trade_start_sc and traial_trade_deal_reset:
                                    #トライアルポジションの場合は決済時にリセットする
                                    trial_trade_start_sc = None

                            deal_hist_dict[hist_child["score"]] = hist_child
                            deal_hist.append(hist_child)
                        else:
                            # 想定外エラー
                            print("ERROR3")
                            sys.exit()

                    for dkey in del_key:
                        del bet_dicts[dkey]

                    position_num = 0
                    position_num_tmp[sc] = position_num
                    continue

                # 注文方法がgyaku_sashine_touch or gyaku_sashine_lastの場合は直前のtermのtickで成行注文するか決める
                if mode == "gyaku_sashine_touch" or mode == "gyaku_sashine_last":
                    for dict_key in bet_dicts.keys():
                        bet_dict = copy.deepcopy(bet_dicts[dict_key])

                        if bet_dict["ptime"] == None:
                            if bet_dict["type"] == "BUY":
                                if mode == "gyaku_sashine_touch":
                                    touch_flg = False
                                    for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list, spr_tick_list):
                                        if tick_ask >= bet_dict["pending_price"]:
                                            touch_flg = True
                                            break
                                    if touch_flg:
                                        # 成行注文する
                                        bet_dicts[dict_key]["ptime"] = sc
                                        bet_dicts[dict_key]["pprice"] = now_ask
                                        bet_dicts[dict_key]["spr"] = spr

                                elif mode == "gyaku_sashine_last":
                                    if now_ask >= bet_dict["pending_price"]:
                                        bet_dicts[dict_key]["ptime"] = sc
                                        bet_dicts[dict_key]["pprice"] = now_ask
                                        bet_dicts[dict_key]["spr"] = spr

                            elif bet_dict["type"] == "SELL":
                                if mode == "gyaku_sashine_touch":
                                    touch_flg = False
                                    for tick_bid, tick_ask, spr_tick in zip(bid_list, ask_list, spr_tick_list):
                                        if tick_bid <= bet_dict["pending_price"]:
                                            touch_flg = True
                                            break
                                    if touch_flg:
                                        # 成行注文する
                                        bet_dicts[dict_key]["ptime"] = sc
                                        bet_dicts[dict_key]["pprice"] = now_bid
                                        bet_dicts[dict_key]["spr"] = spr

                                elif mode == "gyaku_sashine_last":
                                    if now_bid <= bet_dict["pending_price"]:
                                        bet_dicts[dict_key]["ptime"] = sc
                                        bet_dicts[dict_key]["pprice"] = now_bid
                                        bet_dicts[dict_key]["spr"] = spr

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

                # 決済するかどうかを判断
                del_key = []
                for dict_key in bet_dicts.keys():
                    bet_dict = copy.deepcopy(bet_dicts[dict_key])

                    passed_sc = get_decimal_sub(sc, bet_dict["stime"])

                    finish_flg = False

                    if ext_start_sec_short_position_num != None and len(
                            bet_dicts) <= ext_start_sec_short_position_num:
                        ext_start_sec_tmp = ext_start_sec_short
                    else:
                        ext_start_sec_tmp = ext_start_sec

                    if ext_term_short_position_num != None and len(bet_dicts) <= ext_term_short_position_num:
                        ext_term_tmp = ext_term_short
                    else:
                        ext_term_tmp = ext_term

                    if c.FX_MAX_TRADE_SEC != None and passed_sc >= c.FX_MAX_TRADE_SEC:
                        # 最大取引時間を迎えたら決済
                        finish_flg = True
                    elif bet_dict["ptime"] == None and passed_sc >= pending_max_sec:
                        # 約定しないまま決済期間が到来した場合
                        finish_flg = True
                    elif c.FX_NOT_EXT_FLG and passed_sc >= ext_start_sec_tmp:
                        finish_flg = True

                    no_pred_flg = False
                    try:
                        pred_ext = score_pred_dict_ext[sc]  # scoreをもとに延長用予想を取得
                        pred_score.append(sc)
                    except Exception as e:
                        no_pred_flg = True
                        no_pred_score.append(sc)

                    if finish_flg == False:

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
                                            if cat_cond_ext2_flg:
                                                bet_flg = buy_cat_cond_ext2(pred_ext, ext_border)
                                            else:
                                                bet_flg = buy_cat_cond_ext(pred_ext, ext_border)

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

                                    if c.FX_NOT_EXT_MINUS != None:
                                        # X_NOT_EXT_MINUS未満の利益なら延長しない
                                        tmp_profit = now_bid - bet_dict["pprice"]

                                        if tmp_profit < c.FX_NOT_EXT_MINUS:
                                            bet_flg = False
                                    if minus_check_sec != 0 and passed_sc == minus_check_sec:
                                        # 指定秒経過時に利益がマイナスなら延長しない
                                        tmp_profit = now_bid - bet_dict["pprice"]

                                        if tmp_profit < 0:
                                            bet_flg = False

                                    if refer_tick_num_ext != 0 and refer_ext_cnt != 0:
                                        if tick_cnt_ext == 0:
                                            if refer_tick_cnt_0_ng == True:
                                                bet_flg = False
                                        else:
                                            if tick_up_cnt_ext < refer_ext_cnt:
                                                bet_flg = False
                                            if refer_vs and tick_up_cnt_ext < tick_dw_cnt_ext:
                                                bet_flg = False

                                    if most_high_low_div_flg:
                                        tmp_list = np.array(
                                            close_list[cnt - int(get_decimal_divide(min_div_ext_sec, c.BET_TERM)):cnt])
                                        tmp_h = tmp_list.max()
                                        tmp_l = tmp_list.min()
                                        tmp_div = abs(get_divide(tmp_l, tmp_h))
                                    else:
                                        tmp_bef = close_list[cnt - int(get_decimal_divide(min_div_ext_sec, c.BET_TERM))]
                                        tmp_div = get_divide(tmp_bef, close)

                                    if min_div_ext != None:
                                        if 0 <= tmp_div and tmp_div < min_div_ext:
                                            bet_flg = False

                                    if min_div_ext_minus != None:
                                        if min_div_ext_minus < tmp_div and tmp_div <= 0:
                                            bet_flg = False

                                    if max_div_ext != None:
                                        if max_div_ext < tmp_div:
                                            bet_flg = False

                                    if max_div_ext_minus != None:
                                        if tmp_div < max_div_ext_minus:
                                            bet_flg = False

                                    if (ex_min_spread <= spr and spr <= ex_max_spread) == False:
                                        bet_flg = False

                                    if bet_flg or (
                                            bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                        "cannot_deal_cnt"] < cannot_deal_cnt_max):
                                        # 更に上がると予想されている場合、決済しないままとする
                                        # またはスプレッドが範囲外で且つ、範囲外であった回数が規定内である場合、決済しないままとする

                                        bet_dicts[dict_key]["len"] += 1

                                        if bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                            "cannot_deal_cnt"] < cannot_deal_cnt_max:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] += 1
                                        else:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] = 0  # カウントを戻す

                                        x_std_buy_tp, x_std_buy_sl, x_std_buy_sl_max, \
                                        x_std_sell_tp, x_std_sell_sl, x_std_sell_sl_max = get_tpsl(sc, now_ask, now_bid, spr, takeprofit_dict, stoploss_dict, stoploss_max, mode, pending_pips)

                                        if bet_dicts[dict_key]["deal_try_cnt"] != 0:
                                            # 前のループまでは決済しようとしていた場合
                                            if deal_mode == "trail":
                                                # trailの場合は決済しないと判断されても,一度決済すると判断されたらトレイルする
                                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                                new_sl = get_decimal_sub(deal_bid, deal_pending_pips)
                                                if new_sl > bet_dicts[dict_key]["sl"]:
                                                    # trailなので今の逆指値より新しい逆指値より高いなら更新
                                                    bet_dicts[dict_key]["sl"] = new_sl

                                            else:

                                                if deal_mode == 'gyaku_sashine':
                                                    # 決済レートをstoplossに設定しなおす
                                                    bet_dicts[dict_key]["sl"] = x_std_buy_sl
                                                elif deal_mode == "gyaku_sashine_touch" or deal_mode == "gyaku_sashine_last":
                                                    # 決済レートをstoplossに設定しなおす
                                                    bet_dicts[dict_key]["gyaku_sashine_sl"] = x_std_buy_sl
                                                elif deal_mode == 'sashine':
                                                    # 決済レートをtakeprofitに設定しなおす
                                                    bet_dicts[dict_key]["tp"] = x_std_buy_tp

                                                bet_dicts[dict_key]["deal_try_cnt"] = 0

                                        if c.FX_TAKE_PROFIT_FLG:
                                            bet_dicts[dict_key]["tp"] = x_std_buy_tp

                                        if c.FX_STOP_LOSS_FLG:
                                            if stoploss_trail == True:
                                                # 少しでも利益が出ているなら最大ストップロスを切り上げる
                                                if bet_dicts[dict_key]["sl_max"] < x_std_buy_sl_max:
                                                    bet_dicts[dict_key]["sl_max"] = x_std_buy_sl_max

                                            if bet_dicts[dict_key]["sl_max"] < x_std_buy_sl:
                                                bet_dicts[dict_key]["sl"] = x_std_buy_sl

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
                                            if cat_cond_ext2_flg:
                                                bet_flg = sell_cat_cond_ext2(pred_ext, ext_border)
                                            else:
                                                bet_flg = sell_cat_cond_ext(pred_ext, ext_border)
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

                                    if c.FX_NOT_EXT_MINUS != None:
                                        # FX_NOT_EXT_MINUS未満の利益なら延長しない
                                        tmp_profit = ((now_ask - bet_dict["pprice"]) * -1)

                                        if tmp_profit < c.FX_NOT_EXT_MINUS:
                                            bet_flg = False

                                    if minus_check_sec != 0 and passed_sc == minus_check_sec:
                                        # 指定秒経過時に利益がマイナスなら延長しない
                                        tmp_profit = ((now_ask - bet_dict["pprice"]) * -1)

                                        if tmp_profit < 0:
                                            bet_flg = False

                                    if refer_tick_num_ext != 0 and refer_ext_cnt != 0:
                                        if tick_cnt_ext == 0:
                                            if refer_tick_cnt_0_ng == True:
                                                bet_flg = False
                                        else:
                                            if tick_dw_cnt_ext < refer_ext_cnt:
                                                bet_flg = False
                                            if refer_vs and tick_dw_cnt_ext < tick_up_cnt_ext:
                                                bet_flg = False

                                    if most_high_low_div_flg:
                                        tmp_list = np.array(
                                            close_list[cnt - int(get_decimal_divide(min_div_ext_sec, c.BET_TERM)):cnt])
                                        tmp_h = tmp_list.max()
                                        tmp_l = tmp_list.min()
                                        tmp_div = abs(get_divide(tmp_l, tmp_h))
                                    else:
                                        tmp_bef = close_list[cnt - int(get_decimal_divide(min_div_ext_sec, c.BET_TERM))]
                                        tmp_div = get_divide(tmp_bef, close)

                                    if min_div_ext != None:
                                        if 0 <= tmp_div and tmp_div < min_div_ext:
                                            bet_flg = False

                                    if min_div_ext_minus != None:
                                        if min_div_ext_minus < tmp_div and tmp_div <= 0:
                                            bet_flg = False

                                    if max_div_ext != None:
                                        if max_div_ext < tmp_div:
                                            bet_flg = False

                                    if max_div_ext_minus != None:
                                        if tmp_div < max_div_ext_minus:
                                            bet_flg = False

                                    if (ex_min_spread <= spr and spr <= ex_max_spread) == False:
                                        bet_flg = False

                                    if bet_flg or (
                                            bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                        "cannot_deal_cnt"] < cannot_deal_cnt_max):
                                        bet_dicts[dict_key]["len"] += 1

                                        if bet_flg == False and within_end_spread_flg == False and bet_dicts[dict_key][
                                            "cannot_deal_cnt"] < cannot_deal_cnt_max:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] += 1
                                        else:
                                            bet_dicts[dict_key]["cannot_deal_cnt"] = 0  # カウントを戻す

                                        x_std_buy_tp, x_std_buy_sl, x_std_buy_sl_max, \
                                        x_std_sell_tp, x_std_sell_sl, x_std_sell_sl_max = get_tpsl(sc, now_ask, now_bid, spr, takeprofit_dict, stoploss_dict, stoploss_max, mode, pending_pips)

                                        if bet_dicts[dict_key]["deal_try_cnt"] != 0:
                                            # 前のループまでは決済しようとしていた場合
                                            if deal_mode == "trail":
                                                # trailの場合は決済しないと判断されても,一度決済すると判断されたらトレイルする
                                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                                new_sl = get_decimal_add(deal_ask, deal_pending_pips)
                                                if new_sl < bet_dicts[dict_key]["sl"]:
                                                    bet_dicts[dict_key]["sl"] = new_sl

                                            else:
                                                if deal_mode == 'gyaku_sashine':
                                                    # 決済レートをstoplossに設定しなおす
                                                    bet_dicts[dict_key]["sl"] = x_std_sell_sl
                                                elif deal_mode == "gyaku_sashine_touch" or deal_mode == "gyaku_sashine_last":
                                                    # 決済レートをstoplossに設定しなおす
                                                    bet_dicts[dict_key]["gyaku_sashine_sl"] = x_std_sell_sl
                                                elif deal_mode == 'sashine':
                                                    # 決済レートをtakeprofitに設定しなおす
                                                    bet_dicts[dict_key]["tp"] = x_std_sell_tp

                                                bet_dicts[dict_key]["deal_try_cnt"] = 0

                                        if c.FX_TAKE_PROFIT_FLG:
                                            bet_dicts[dict_key]["tp"] = x_std_sell_tp

                                        if c.FX_STOP_LOSS_FLG:
                                            if stoploss_trail == True:
                                                # 少しでも利益が出ているなら最大ストップロスを切り上げる
                                                if bet_dicts[dict_key]["sl_max"] > x_std_sell_sl_max:
                                                    bet_dicts[dict_key]["sl_max"] = x_std_sell_sl_max

                                            if bet_dicts[dict_key]["sl_max"] > x_std_sell_sl:
                                                bet_dicts[dict_key]["sl"] = x_std_sell_sl

                                    else:
                                        finish_flg = True

                    if finish_flg:
                        if bet_dict["ptime"] == None:
                            # 約定してない場合注文なしとする
                            del_key.append(dict_key)
                        elif spr < 0:
                            # スプレッドがマイナスの場合、スプレッド情報がなかったという判断をし取引無効とする
                            del_key.append(dict_key)
                        else:
                            # 現時点での利益(手数料などを考慮しない)
                            if bet_dict["type"] == "BUY":
                                if conf.SYMBOL == "BTCUSD":
                                    tmp_profit_pips = get_sub(bet_dict["pprice"], deal_bid)
                                else:
                                    tmp_profit_pips = get_sub(bet_dict["pprice"], deal_bid)
                            elif bet_dict["type"] == "SELL":
                                if conf.SYMBOL == "BTCUSD":
                                    tmp_profit_pips = get_sub(deal_ask, bet_dict["pprice"])
                                else:
                                    tmp_profit_pips = get_sub(deal_ask, bet_dict["pprice"])

                            if deal_mode == 'market' or bet_dicts[dict_key]["deal_try_cnt"] >= deal_pending_max_cnt or \
                                    (
                                            deal_mode == "gyaku_sashine" and deal_gyaku_sashine == True and tmp_profit_pips < 0):

                                # 成行決済の場合　もしくは指値、逆指値決済で決済連続で指定回数以上の場合は諦めて成行決済する
                                # deal_gyaku_sashine==Trueなら逆指値決済モードでも現時点での利益がマイナスなら成行決済する

                                prev_bet_end_score = sc
                                if bet_dict["type"] == "BUY":
                                    # 決済する
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                        profit_pips = get_decimal_sub(deal_bid, bet_dict["pprice"]) - btcusd_spr
                                    else:
                                        profit_pips = get_decimal_sub(deal_bid, bet_dict["pprice"]) + c.ADJUST_PIPS

                                    pips.append(profit_pips)
                                    pips_up.append(profit_pips)

                                    atrs.append(bet_dict["atr"])
                                    inds.append(bet_dict["ind"])
                                    times.append(bet_dict["ptime"])
                                    if c.BINARY == True:
                                        if 0.0 < profit_pips:
                                            profit = c.PAYOUT
                                            wc += 1
                                        elif 0.0 == profit_pips and c.AtMoney:
                                            profit = 0
                                        else:
                                            profit = c.PAYOFF
                                            lc += 1
                                    else:
                                        if c.JPY_FLG == False:
                                            profit = profit_pips * bet_dict["position"] * jpy
                                        else:
                                            profit = profit_pips * bet_dict["position"]

                                    spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                                    stop_price = deal_bid

                                    profit_day = profit_day + profit
                                    profit_day_partial = profit_day_partial + profit

                                    money = money + profit
                                    max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown,
                                                                                            drawdown, profit, sc,
                                                                                            max_drawdown_sc)
                                    money_tmp[sc] = money
                                    position_num = position_num - 1
                                    position_num_tmp[sc] = position_num

                                    hist_child = {"type": bet_dict["type"],
                                                  "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                                      '%Y/%m/%d %H:%M:%S'),
                                                  "ptime": datetime.fromtimestamp(bet_dict["ptime"]).strftime(
                                                      '%Y/%m/%d %H:%M:%S'),
                                                  "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                                  "sprice": bet_dict["sprice"],
                                                  "pprice": bet_dict["pprice"],
                                                  "eprice": stop_price,
                                                  "profit_pips": profit_pips,
                                                  "profit": profit,
                                                  "spr": bet_dict["spr"],
                                                  "spr_end": spr,
                                                  "score": bet_dict["stime"],
                                                  "trade_sec": sc - bet_dict["ptime"],
                                                  "stop_loss": False,
                                                  "take_profit": False,
                                                  "past_pred_1": bet_dict["past_pred_1"],
                                                  "past_close_move_1": bet_dict["past_close_move_1"],
                                                  "high_profit": bet_dict["high_profit"],
                                                  "high_profit_sec": bet_dict["high_profit_sec"],
                                                  }

                                    if target == "category":
                                        hist_child["pred"] = bet_dict["pred"]

                                    if show_profit_per_div:
                                        for d in show_profit_per_div_list:
                                            key_str = "div" + str(d)
                                            hist_child[key_str] = bet_dict[key_str]

                                    for m in show_h1_sma_list:
                                        key_str = "sma-" + str(m)
                                        hist_child[key_str] = bet_dict[key_str]

                                    for b in show_profit_per_boli_std:
                                        foot, length, alpha = b.split("-")
                                        key_str = "BOLI-" + foot + "-" + length + "-STD" + alpha
                                        hist_child[key_str] = bet_dict[key_str]

                                    for b in show_profit_per_boli_std_div:
                                        foot, length, alpha = b.split("-")
                                        key_str = "BOLI-" + foot + "-" + length + "-STD"

                                        hist_child[key_str] = bet_dict[key_str]
                                        hist_child[key_str + alpha + "-UP-DIV"] = bet_dict[key_str + alpha + "-UP-DIV"]
                                        hist_child[key_str + alpha + "-DW-DIV"] = bet_dict[key_str + alpha + "-DW-DIV"]
                                        hist_child[key_str + alpha + "-MEAN-DIV"] = bet_dict[
                                            key_str + alpha + "-MEAN-DIV"]

                                    if bet_dict["high_profit"] < profit_pips:
                                        hist_child["high_profit"] = profit_pips
                                        hist_child["high_profit_sec"] = sc - bet_dict["ptime"]

                                    if trial_trade_flg and trial_trade_start_sc != None:
                                        if bet_dict["ptime"] == trial_trade_start_sc and traial_trade_deal_reset:
                                            # トライアルポジションの場合は決済時にリセットする
                                            trial_trade_start_sc = None

                                    deal_hist_dict[hist_child["score"]] = hist_child
                                    deal_hist.append(hist_child)
                                    del_key.append(dict_key)

                                elif bet_dict["type"] == "SELL":
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                        profit_pips = (get_decimal_sub(deal_ask, bet_dict["pprice"]) * -1) - btcusd_spr
                                    else:
                                        profit_pips = (get_decimal_sub(deal_ask,
                                                                       bet_dict["pprice"]) * -1) + c.ADJUST_PIPS

                                    pips.append(profit_pips)
                                    pips_dw.append(profit_pips)

                                    atrs.append(bet_dict["atr"])
                                    inds.append(bet_dict["ind"])
                                    times.append(bet_dict["ptime"])
                                    if c.BINARY == True:
                                        if 0.0 < profit_pips:
                                            profit = c.PAYOUT
                                            wc += 1
                                        elif 0.0 == profit_pips and c.AtMoney:
                                            profit = 0
                                        else:
                                            profit = c.PAYOFF
                                            lc += 1
                                    else:
                                        if c.JPY_FLG == False:
                                            profit = profit_pips * bet_dict["position"] * jpy
                                        else:
                                            profit = profit_pips * bet_dict["position"]
                                    spr_pred_pips_list.append([bet_dict["spr"], bet_dict["type"], profit_pips, profit])

                                    stop_price = deal_ask

                                    profit_day = profit_day + profit
                                    profit_day_partial = profit_day_partial + profit

                                    money = money + profit
                                    max_drawdown, drawdown, max_drawdown_sc = countDrawdoan(max_drawdowns, max_drawdown,
                                                                                            drawdown, profit, sc,
                                                                                            max_drawdown_sc)
                                    money_tmp[sc] = money
                                    position_num = position_num - 1
                                    position_num_tmp[sc] = position_num

                                    hist_child = {"type": bet_dict["type"],
                                                  "stime": datetime.fromtimestamp(bet_dict["stime"]).strftime(
                                                      '%Y/%m/%d %H:%M:%S'),
                                                  "ptime": datetime.fromtimestamp(bet_dict["ptime"]).strftime(
                                                      '%Y/%m/%d %H:%M:%S'),
                                                  "etime": datetime.fromtimestamp(sc).strftime('%Y/%m/%d %H:%M:%S'),
                                                  "sprice": bet_dict["sprice"],
                                                  "pprice": bet_dict["pprice"],
                                                  "eprice": stop_price,
                                                  "profit_pips": profit_pips,
                                                  "profit": profit,
                                                  "spr": bet_dict["spr"],
                                                  "spr_end": spr,
                                                  "score": bet_dict["stime"],
                                                  "trade_sec": sc - bet_dict["ptime"],
                                                  "stop_loss": False,
                                                  "take_profit": False,
                                                  "past_pred_1": bet_dict["past_pred_1"],
                                                  "past_close_move_1": bet_dict["past_close_move_1"],
                                                  "high_profit": bet_dict["high_profit"],
                                                  "high_profit_sec": bet_dict["high_profit_sec"],
                                                  }

                                    if target == "category":
                                        hist_child["pred"] = bet_dict["pred"]

                                    if show_profit_per_div:
                                        for d in show_profit_per_div_list:
                                            key_str = "div" + str(d)
                                            hist_child[key_str] = bet_dict[key_str]

                                    for m in show_h1_sma_list:
                                        key_str = "sma-" + str(m)
                                        hist_child[key_str] = bet_dict[key_str]

                                    for b in show_profit_per_boli_std:
                                        foot, length, alpha = b.split("-")
                                        key_str = "BOLI-" + foot + "-" + length + "-STD" + alpha
                                        hist_child[key_str] = bet_dict[key_str]

                                    for b in show_profit_per_boli_std_div:
                                        foot, length, alpha = b.split("-")
                                        key_str = "BOLI-" + foot + "-" + length + "-STD"

                                        hist_child[key_str] = bet_dict[key_str]
                                        hist_child[key_str + alpha + "-UP-DIV"] = bet_dict[key_str + alpha + "-UP-DIV"]
                                        hist_child[key_str + alpha + "-DW-DIV"] = bet_dict[key_str + alpha + "-DW-DIV"]
                                        hist_child[key_str + alpha + "-MEAN-DIV"] = bet_dict[
                                            key_str + alpha + "-MEAN-DIV"]

                                    if bet_dict["high_profit"] < profit_pips:
                                        hist_child["high_profit"] = profit_pips
                                        hist_child["high_profit_sec"] = sc - bet_dict["ptime"]

                                    if trial_trade_flg and trial_trade_start_sc != None:
                                        if bet_dict["ptime"] == trial_trade_start_sc and traial_trade_deal_reset:
                                            # トライアルポジションの場合は決済時にリセットする
                                            trial_trade_start_sc = None

                                    deal_hist_dict[hist_child["score"]] = hist_child
                                    deal_hist.append(hist_child)
                                    del_key.append(dict_key)
                                else:
                                    # 想定外エラー
                                    print("ERROR3")
                                    sys.exit()

                            elif deal_mode == "sashine":
                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                if bet_dict["type"] == "BUY":
                                    bet_dicts[dict_key]["tp"] = get_decimal_add(deal_bid, deal_pending_pips)
                                elif bet_dict["type"] == "SELL":
                                    bet_dicts[dict_key]["tp"] = get_decimal_sub(deal_ask, deal_pending_pips)

                            elif deal_mode == "gyaku_sashine":
                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                if bet_dict["type"] == "BUY":
                                    bet_dicts[dict_key]["sl"] = get_decimal_sub(deal_bid, deal_pending_pips)
                                elif bet_dict["type"] == "SELL":
                                    bet_dicts[dict_key]["sl"] = get_decimal_add(deal_ask, deal_pending_pips)

                            elif deal_mode == "gyaku_sashine_touch" or deal_mode == "gyaku_sashine_last":
                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                if bet_dict["type"] == "BUY":
                                    bet_dicts[dict_key]["gyaku_sashine_sl"] = get_decimal_sub(deal_bid,
                                                                                              deal_pending_pips)
                                elif bet_dict["type"] == "SELL":
                                    bet_dicts[dict_key]["gyaku_sashine_sl"] = get_decimal_add(deal_ask,
                                                                                              deal_pending_pips)

                            elif deal_mode == "trail":
                                bet_dicts[dict_key]["deal_try_cnt"] += 1

                                if bet_dict["type"] == "BUY":
                                    bet_dicts[dict_key]["sl"] = get_decimal_sub(deal_bid, deal_pending_pips)
                                elif bet_dict["type"] == "SELL":
                                    bet_dicts[dict_key]["sl"] = get_decimal_add(deal_ask, deal_pending_pips)

                for dkey in del_key:
                    del bet_dicts[dkey]

                # 一定利益が上がっていた場合、損切りラインを上げる
                if CHANGE_STOPLOSS_TERM != None:
                    x_std_buy_tp, x_std_buy_sl, x_std_buy_sl_max, \
                    x_std_sell_tp, x_std_sell_sl, x_std_sell_sl_max = get_tpsl(sc, now_ask, now_bid, spr, takeprofit_dict, stoploss_dict, stoploss_max, mode, pending_pips)

                    for dict_key in bet_dicts.keys():
                        bet_dict = copy.deepcopy(bet_dicts[dict_key])
                        if bet_dict["ptime"] != None and bet_dict["deal_try_cnt"] == 0:
                            # 約定済みの場合　かつ　gyaku_sashineなどで決済しようとしていない場合
                            passed_sc = get_decimal_sub(sc, bet_dict["ptime"])
                            if CHANGE_STOPLOSS_TERM == 0 or get_decimal_mod(passed_sc, CHANGE_STOPLOSS_TERM) == 0:
                                if bet_dict["type"] == "BUY":
                                    tmp_profit = get_decimal_sub(now_bid, bet_dict["pprice"])
                                    prev_profit = bet_dict["prev_profit"]
                                    profit_sub = get_decimal_sub(tmp_profit, prev_profit)
                                    if profit_sub > CHANGE_STOPLOSS_PRICE:
                                        bet_dicts[dict_key]["sl"] = x_std_buy_sl
                                        bet_dicts[dict_key]["prev_profit"] = tmp_profit

                                elif bet_dict["type"] == "SELL":
                                    tmp_profit = get_decimal_sub(now_ask, bet_dict["pprice"]) * -1
                                    prev_profit = bet_dict["prev_profit"]
                                    profit_sub = get_decimal_sub(tmp_profit, prev_profit)
                                    if profit_sub > CHANGE_STOPLOSS_PRICE:
                                        bet_dicts[dict_key]["sl"] = x_std_sell_sl
                                        bet_dicts[dict_key]["prev_profit"] = tmp_profit

                if idx == -1:
                    # 予想がない場合は判断材料がないので注文しない
                    continue

                # 新規注文
                buy_bet_flg = False
                sell_bet_flg = False

                if low_spread != None and spr <= low_spread:
                    # スプレッドが低ければborderも低くして取引回数を増やす
                    border = low_spread_border

                if target in ["d", "sub"]:
                    buy_bet_flg = buy_d_cond_1(pred, border, border_ceil)
                elif target == "category":
                    buy_bet_flg = buy_cat_cond_1(pred, border, border_ceil)
                elif target == "category_bin":
                    buy_bet_flg = buy_cat_cond_2(pred, border)
                elif target == "category_bin_both":
                    buy_bet_flg = buy_cat_cond_1(pred, border[0], border_ceil)

                if target in ["d", "sub"]:
                    sell_bet_flg = sell_d_cond_1(pred, border, border_ceil)
                elif target == "category":
                    sell_bet_flg = sell_cat_cond_1(pred, border, border_ceil)
                elif target == "category_bin":
                    sell_bet_flg = sell_cat_cond_2(pred, border)
                elif target == "category_bin_both":
                    sell_bet_flg = sell_cat_cond_1(pred, border[1], border_ceil)

                if c.BUY_FLG == False:
                    buy_bet_flg = False

                if c.SELL_FLG == False:
                    sell_bet_flg = False

                if buy_bet_flg == False and sell_bet_flg == False:
                    continue
                else:
                    tmp_bet_cnt += 1

                    now_shift = int(Decimal(str(sc)) % Decimal(str(c.TERM)))
                    ng_shift = int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT)))

                    tmp_position_buy = c.get_fx_position_jpy(order_ask,
                                                             jpy) if c.JPY_FLG == False else c.get_fx_position(
                        order_ask)
                    tmp_position_sell = c.get_fx_position_jpy(order_bid,
                                                              jpy) if c.JPY_FLG == False else c.get_fx_position(
                        order_bid)

                    if c.POSITION_BY_PRED:
                        # 予想確率ごとにポジション数を変える
                        for p_l in c.POSITION_BY_PRED_LIST:
                            if p_l[0] <= pred[0] and pred[0] < p_l[1]:
                                tmp_position_buy = p_l[2]
                            if p_l[0] <= pred[2] and pred[2] < p_l[1]:
                                tmp_position_sell = p_l[2]

                    # 現在のmargin,margin_freeを計算し資金余力がない場合は新規発注しない
                    if stoploss_max_day != None and c.BINARY == False:
                        # 現在の含み損益を計算
                        total_profit_tmp = 0
                        for dict_key in bet_dicts.keys():
                            bet_dict = copy.deepcopy(bet_dicts[dict_key])
                            if bet_dict["ptime"] != None:  # 約定している場合
                                if bet_dict["type"] == "BUY":
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                        tmp_profit_pips = now_bid - bet_dict["pprice"] - btcusd_spr
                                    else:
                                        tmp_profit_pips = now_bid - bet_dict["pprice"] + conf.ADJUST_PIPS

                                elif bet_dict["type"] == "SELL":
                                    if conf.SYMBOL == "BTCUSD":
                                        btcusd_spr = float(Decimal(str(bet_dict["pprice"])) * (
                                                Decimal(str(conf.BTCUSD_SPREAD_PERCENT)) / Decimal("100")))
                                        tmp_profit_pips = bet_dict["pprice"] - now_ask - btcusd_spr
                                    else:
                                        tmp_profit_pips = bet_dict["pprice"] - now_ask + conf.ADJUST_PIPS

                                if c.JPY_FLG == False:
                                    tmp_profit = tmp_profit_pips * bet_dict["position"] * jpy
                                else:
                                    tmp_profit = tmp_profit_pips * bet_dict["position"]

                                total_profit_tmp = total_profit_tmp + tmp_profit

                        margin_total = 0
                        for dict_key in bet_dicts.keys():
                            margin_total = margin_total + bet_dicts[dict_key]["margin"]

                        margin_free = (stoploss_max_day * -1) + profit_day + total_profit_tmp - margin_total
                        if buy_bet_flg and margin_free < (tmp_position_buy * order_ask) / c.FX_LEVERAGE:
                            continue
                        elif sell_bet_flg and margin_free < (tmp_position_sell * order_bid) / c.FX_LEVERAGE:
                            continue

                    if importantAnswer.is_except(sc):
                        # print("重要指標発表時は除外", sc)
                        continue

                    if c.TRADE_SHIFT != None and get_decimal_mod(sc, c.TRADE_SHIFT) != 0:
                        # 指定した秒のシフトでないと取引しない
                        # print("NG TRADE_SHIFT")
                        continue

                    # 指定シフト以外トレードしない
                    if (len(c.FX_TARGET_SHIFT) == 0 or (
                            len(c.FX_TARGET_SHIFT) != 0 and now_shift in c.FX_TARGET_SHIFT)) == False:
                        # print("NG FX_TARGET_SHIFT")
                        continue

                    # マイナススプレッドを対象外とする
                    if c.IGNORE_MINUS_SPREAD and spr < 0:
                        # print("NG IGNORE_MINUS_SPREAD")
                        continue

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

                    # 一定時間経過しないとつづけて注文できない
                    if c.RESTRICT_FLG:
                        if prev_bet_start_score != None and get_decimal_add(prev_bet_start_score, c.RESTRICT_SEC) > sc:
                            continue
                        elif prev_bet_end_score != None and get_decimal_add(prev_bet_end_score, c.RESTRICT_END_SEC) > sc:
                            continue

                    if c.SAME_SHIFT_NG_FLG == True:
                        # 同じshiftの建玉が既にあるかチェック
                        same_shift_num = 0

                        for k, v in bet_dicts.items():
                            if v["shift"] == ng_shift:
                                same_shift_num += 1
                        if same_shift_num >= c.NG_SHIFT_MAX:
                            # 同じshiftの建玉が既に規定数よりある場合新規発注しない
                            continue

                    if c.FX_MAX_POSITION_CNT != None and len(bet_dicts) == c.FX_MAX_POSITION_CNT:
                        # 既に最大保持可能ポジション数の場合、新規発注しない
                        # print("NG FX_MAX_POSITION_CNT")
                        continue

                    if c.FX_SINGLE_FLG == True and len(bet_dicts) != 0:
                        # print("NG FX_SINGLE")
                        continue

                    if profit_day_break_partial == True:
                        # stoploss_max_partialの損失を超えた場合は新規発注しない
                        continue

                    if profit_day_break == True:
                        # 一日の最大損失を超えた場合(ロスカットされた場合)は新規発注しない
                        continue

                    if emerg_div != None and emerg_stop_start_sc != None:
                        if get_decimal_sub(sc, emerg_stop_start_sc) <= emerg_stop_sec:
                            # divの急騰から指定時間内は取引しない
                            continue

                    """
                    # FX_FUND * FX_LEVERAGE - (既に保持しているポジション数 * FX_FIX_POSITION * rate)の値が0以上であれば(資金余裕があれば)新規ポジションを持てる
                    if c.FX_FIX_POSITION != 0:
                        if c.FX_FUND * c.FX_LEVERAGE - ((position_num + 1) * c.FX_FIX_POSITION * now_price) < 0:

                            fund_out_cnt += 1
                            continue
                    """

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
                    if most_high_low_div_flg:
                        tmp_list = np.array(close_list[cnt - int(get_decimal_divide(min_div_sec, c.BET_TERM)):cnt])
                        tmp_h = tmp_list.max()
                        tmp_l = tmp_list.min()
                        tmp_div = abs(get_divide(tmp_l, tmp_h))
                    else:
                        tmp_bef = close_list[cnt - int(get_decimal_divide(max_div_sec, c.BET_TERM))]
                        tmp_div = get_divide(tmp_bef, close)

                    if min_div != None:
                        if 0 <= tmp_div and tmp_div < min_div:
                            continue

                    if min_div_minus != None:
                        if min_div_minus < tmp_div and tmp_div <= 0:
                            continue

                    if max_div != None:
                        if max_div < tmp_div:
                            continue

                    if max_div_minus != None:
                        if tmp_div < max_div_minus:
                            continue

                    if ORDER_TOTAL_STOPLOSS != None:
                        if total_profit <= ORDER_TOTAL_STOPLOSS:
                            # 現在の全建玉の損失が大きければ新たに発注しない
                            continue

                    if refer_tick_num != 0 and refer_cnt != 0:
                        if buy_bet_flg:
                            if tick_cnt == 0:
                                if refer_tick_cnt_0_ng == True:
                                    continue
                            else:
                                if tick_up_cnt < refer_cnt:
                                    continue
                                if refer_vs and tick_up_cnt < tick_dw_cnt:
                                    continue

                        elif sell_bet_flg:
                            if tick_cnt == 0:
                                if refer_tick_cnt_0_ng == True:
                                    continue
                            else:
                                if tick_dw_cnt < refer_cnt:
                                    continue
                                if refer_vs and tick_dw_cnt < tick_up_cnt:
                                    continue

                    # 指定スプレッド以外のトレードは無視する
                    if (start_min_spread <= spr and spr <= start_max_spread) == False:
                        ng_spread_cnt += 1
                        # print("spr:",spr, " score:", sc)
                        continue
                    else:
                        ok_spread_cnt += 1

                    # 1秒前の予想を取得
                    try:
                        past_pred_1 = score_pred_dict.get(sc - 1)
                        past_close = score_close_dict.get(sc - 1)
                        past_close_move_1 = get_decimal_sub(now_price, past_close)
                    except Exception as e:
                        pass

                    # 過去の予想結果が低い場合取引しない
                    if refer_past_pred_sec != 0:
                        try:
                            past_pred = score_pred_dict.get(sc - refer_past_pred_sec)

                            if buy_bet_flg:
                                if past_pred[0] < refer_past_pred:
                                    buy_bet_flg = False

                            elif sell_bet_flg:
                                if past_pred[2] < refer_past_pred:
                                    sell_bet_flg = False

                        except Exception as e:
                            # 過去予想がないならそのまま取引する
                            pass


                    if trial_trade_flg:
                        if trial_trade_start_sc != None:
                            if (trial_trade_type == "BUY" and buy_bet_flg) or (trial_trade_type == "SELL" and sell_bet_flg):

                                if get_decimal_sub(sc, trial_trade_start_sc) <= trial_trade_sec:
                                    # トライアル秒数がまだ経過していないなら取引しない
                                    continue
                                else:
                                    if get_decimal_sub(sc, trial_trade_start_sc) <= trial_trade_lookup_sec:
                                        if trial_trade_pips <= trial_trade_pips_min:
                                            # トライアル秒数が経過したが、まだ参照すべき時間内の場合、成績が悪かったら取引しない
                                            continue

                    if stoploss_lookup_start_sc != None:
                        if get_decimal_sub(sc, stoploss_lookup_start_sc) <= stoploss_lookup_sec:
                            # ストップロス発生から参照秒数がまだ経過していないなら取引しない
                            continue

                    if buy_bet_flg and len(boli_ng_range_buy) != 0:
                        buy_bet_flg = boli_ng_range_judge(conf, redis_db_boli, sc, close, boli_ng_range_buy)

                    elif sell_bet_flg and len(boli_ng_range_sell) != 0:
                        sell_bet_flg = boli_ng_range_judge(conf, redis_db_boli, sc, close, boli_ng_range_sell)

                    if buy_bet_flg or sell_bet_flg:
                        x_std_buy_tp, x_std_buy_sl, x_std_buy_sl_max, \
                        x_std_sell_tp, x_std_sell_sl, x_std_sell_sl_max = get_tpsl(sc, now_ask, now_bid, spr,
                                                                                   takeprofit_dict, stoploss_dict,
                                                                                   stoploss_max, mode, pending_pips)
                    if buy_bet_flg:
                        prev_bet_start_score = sc
                        bet_dicts[sc] = {
                            "shift": int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT))),
                            "type": "BUY",
                            "stime": sc,
                            "sprice": order_ask,
                            "len": 1,
                            "tp": x_std_buy_tp,
                            "sl": x_std_buy_sl,
                            "sl_max": x_std_buy_sl_max,
                            "atr": atr,
                            "spr": spr,
                            "jpy": jpy,
                            "ind": ind,
                            "prev_profit": 0,
                            "cannot_deal_cnt": 0,
                            # スプレッドがend_min_spreadからend_max_spreadの範囲外のために決済できなかった回数　cannot_deal_cnt_maxを超えた場合決済する
                            "deal_try_cnt": 0,  # 指値で決済しようとした回数
                            "past_pred_1": past_pred_1,
                            "past_close_move_1": past_close_move_1,
                            "high_profit": 0,
                            "high_profit_sec": 0,
                            "position": tmp_position_buy,
                            "margin": (tmp_position_buy * order_ask) / c.FX_LEVERAGE if c.JPY_FLG else (tmp_position_buy * order_ask) * jpy / c.FX_LEVERAGE,
                        }

                        if mode == "market":
                            bet_dicts[sc]["pprice"] = order_ask
                            bet_dicts[sc]["ptime"] = sc
                            bet_dicts[sc]["pending_price"] = None

                        else:
                            bet_dicts[sc]["pprice"] = None
                            bet_dicts[sc]["ptime"] = None
                            if mode == "sashine":
                                bet_dicts[sc]["pending_price"] = get_decimal_sub(order_ask, pending_pips)
                            elif mode in ("gyaku_sashine", "gyaku_sashine_touch", "gyaku_sashine_last"):
                                bet_dicts[sc]["pending_price"] = get_decimal_add(order_ask, pending_pips)

                        if target == "category":
                            bet_dicts[sc]["pred"] = pred[0]

                        bet_cnt += 1
                        position_num = position_num + 1
                        position_num_tmp[sc] = position_num

                        if trial_trade_flg:
                            if trial_trade_start_sc != None:
                                if get_decimal_sub(sc, trial_trade_start_sc) > trial_trade_lookup_sec or trial_trade_type == "SELL":
                                    # 参照すべき時間が経過したら新たなトライアル取引として保存
                                    trial_trade_start_sc = sc
                                    trial_trade_pips = trial_trade_pips_min
                                    trial_trade_type = "BUY"
                                    if trial_trade_stoploss != None:
                                        bet_dicts[sc]["sl_max"] = get_decimal_sub(now_ask, trial_trade_stoploss)
                                    if trial_trade_position != None:
                                        bet_dicts[sc]["position"] = trial_trade_position
                            else:
                                trial_trade_start_sc = sc
                                trial_trade_pips = trial_trade_pips_min
                                trial_trade_type = "BUY"
                                if trial_trade_stoploss != None:
                                    bet_dicts[sc]["sl_max"] = get_decimal_sub(now_ask, trial_trade_stoploss)
                                if trial_trade_position != None:
                                    bet_dicts[sc]["position"] = trial_trade_position

                    elif sell_bet_flg:
                        prev_bet_start_score = sc
                        bet_dicts[sc] = {
                            "shift": int(Decimal(str(sc)) % Decimal(str(c.NG_SHIFT))),
                            "type": "SELL",
                            "stime": sc,
                            "sprice": order_bid,
                            "len": 1,
                            "tp": x_std_sell_tp,
                            "sl": x_std_sell_sl,
                            "sl_max": x_std_sell_sl_max,
                            "atr": atr,
                            "spr": spr,
                            "jpy": jpy,
                            "ind": ind,
                            "prev_profit": 0,
                            "cannot_deal_cnt": 0,
                            # スプレッドがend_min_spreadからend_max_spreadの範囲外のために決済できなかった回数　cannot_deal_cnt_maxを超えた場合決済する
                            "deal_try_cnt": 0,  # 指値で決済しようとした回数
                            "past_pred_1": past_pred_1,
                            "past_close_move_1": past_close_move_1,
                            "high_profit": 0,
                            "high_profit_sec": 0,
                            "position": tmp_position_sell,
                            "margin": (tmp_position_sell * order_bid) / c.FX_LEVERAGE if c.JPY_FLG else (tmp_position_sell * order_bid) * jpy / c.FX_LEVERAGE,
                        }

                        if mode == "market":
                            bet_dicts[sc]["pprice"] = order_bid
                            bet_dicts[sc]["ptime"] = sc
                            bet_dicts[sc]["pending_price"] = None

                        else:
                            bet_dicts[sc]["pprice"] = None
                            bet_dicts[sc]["ptime"] = None
                            if mode == "sashine":
                                bet_dicts[sc]["pending_price"] = get_decimal_add(order_bid, pending_pips)
                            elif mode in ("gyaku_sashine", "gyaku_sashine_touch", "gyaku_sashine_last"):
                                bet_dicts[sc]["pending_price"] = get_decimal_sub(order_bid, pending_pips)

                        if target == "category":
                            bet_dicts[sc]["pred"] = pred[2]

                        bet_cnt += 1
                        position_num = position_num + 1
                        position_num_tmp[sc] = position_num

                        if trial_trade_flg:
                            if trial_trade_start_sc != None:
                                if get_decimal_sub(sc, trial_trade_start_sc) > trial_trade_lookup_sec or trial_trade_type == "BUY":
                                    # 参照すべき時間が経過したら新たなトライアル取引として保存
                                    trial_trade_start_sc = sc
                                    trial_trade_pips = trial_trade_pips_min
                                    trial_trade_type = "SELL"
                                    if trial_trade_stoploss != None:
                                        bet_dicts[sc]["sl_max"] = get_decimal_add(now_bid, trial_trade_stoploss)
                                    if trial_trade_position != None:
                                        bet_dicts[sc]["position"] = trial_trade_position

                            else:
                                trial_trade_start_sc = sc
                                trial_trade_pips = trial_trade_pips_min
                                trial_trade_type = "SELL"
                                if trial_trade_stoploss != None:
                                    bet_dicts[sc]["sl_max"] = get_decimal_add(now_bid, trial_trade_stoploss)
                                if trial_trade_position != None:
                                    bet_dicts[sc]["position"] = trial_trade_position

                    if buy_bet_flg or sell_bet_flg:
                        if show_profit_per_div:
                            for d in show_profit_per_div_list:
                                tmp_bef = close_list[cnt - int(get_decimal_divide(d, c.BET_TERM))]
                                bet_dicts[sc]["div" + str(d)] = get_divide(tmp_bef, close)

                        if len(show_h1_sma_list) != 0:
                            # 予想時のスコアが属する1時間足の一つ前の1分足のSMAデータを取得
                            predict_score_m1 = get_decimal_sub(sc, get_decimal_mod(sc, 3600))  # 予想時のスコアが属する1時間足のスコア
                            target_score = get_decimal_sub(predict_score_m1, 3600)  # 一つ前の1時間足のスコア
                            m1 = m1_data.get(target_score)

                            for m in show_h1_sma_list:
                                if m1 == None:
                                    bet_dicts[sc]["sma-" + str(m)] = None
                                else:
                                    sma = m1.get("sma-" + str(m))
                                    if sma == None:
                                        bet_dicts[sc]["sma-" + str(m)] = None
                                    else:
                                        # 新たに今のレートを加えて移動平均を求め直す
                                        sma = get_decimal_divide(get_decimal_multi(sma, m) + now_price, m + 1)
                                        bet_dicts[sc]["sma-" + str(m)] = get_divide(float(sma), now_price)
                                        # bet_dicts[sc]["sma-" + str(m)] = float(sma)

                        if len(show_profit_per_boli_std) != 0:
                            boli_db_name = conf.SYMBOL + "_1_0"
                            boli_data = redis_db_boli.zrangebyscore(boli_db_name, sc - 1, sc - 1, withscores=True)
                            boli_data_tmp = json.loads(boli_data[0][0])

                            for b in show_profit_per_boli_std:
                                foot, length, alpha = b.split("-")
                                col_name = "BOLI-" + foot + "-" + length + "-STD"
                                std = boli_data_tmp.get(col_name)
                                if std != None:
                                    bet_dicts[sc][col_name + alpha] = get_decimal_multi(std, alpha)
                                else:
                                    bet_dicts[sc][col_name + alpha] = None

                        if len(show_profit_per_boli_std_div) != 0:
                            boli_db_name = conf.SYMBOL + "_1_0"
                            boli_data = redis_db_boli.zrangebyscore(boli_db_name, sc - 1, sc - 1, withscores=True)
                            boli_data_tmp = json.loads(boli_data[0][0])

                            for b in show_profit_per_boli_std_div:
                                foot, length, alpha = b.split("-")
                                col_name = "BOLI-" + foot + "-" + length + "-STD"
                                std = boli_data_tmp.get(col_name)
                                if std != None:
                                    mean = boli_data_tmp.get("BOLI-" + foot + "-" + length + "-MEAN")

                                    bet_dicts[sc][col_name] = std

                                    # 下のボリバンのアルファを求める
                                    boli = get_decimal_sub(mean, get_decimal_multi(std, alpha))
                                    bet_dicts[sc][col_name + alpha + "-DW-DIV"] = get_divide(boli, close)

                                    # 上のボリバンのアルファを求める
                                    boli = get_decimal_add(mean, get_decimal_multi(std, alpha))
                                    bet_dicts[sc][col_name + alpha + "-UP-DIV"] = get_divide(boli, close)

                                    # ボリバンのMEANとのDIVを求める
                                    bet_dicts[sc][col_name + alpha + "-MEAN-DIV"] = get_divide(mean, close)

            suffix_txt_tmp.append(datetime.now().__str__() + " loop end")
            print(datetime.now().__str__() + " loop end")

            max_drawdowns[end.timestamp() - 1] = max_drawdown
            prev_money = c.START_MONEY

            for i, score in enumerate(plot_score_list):
                if score in money_tmp.keys():
                    prev_money = money_tmp[score]

                money_y.append(prev_money)

            detail_profit = prev_money - c.START_MONEY
            suffix_txt_tmp.append("")
            if ok_spread_cnt > 0 or ng_spread_cnt > 0:
                suffix_txt_tmp.append("ok_spread_cnt:" + str(ok_spread_cnt) + " ok_spread_percent:" + str(
                    ok_spread_cnt / (ok_spread_cnt + ng_spread_cnt)))
                suffix_txt_tmp.append("ng_spread_cnt:" + str(ng_spread_cnt) + " ng_spread_percent:" + str(
                    ng_spread_cnt / (ok_spread_cnt + ng_spread_cnt)))

            suffix_txt_tmp.append("pred_cnt:" + str(len(pred_score)))
            suffix_txt_tmp.append("no_pred_cnt:" + str(len(no_pred_score)))

            suffix_txt_tmp.append("")
            suffix_txt_tmp.append("tmp bet cnt: " + str(tmp_bet_cnt))
            suffix_txt_tmp.append("bet cnt: " + str(bet_cnt))
            suffix_txt_tmp.append("Detail Earned Money: " + str(detail_profit))
            suffix_txt_tmp.append("fund_out_cnt: " + str(fund_out_cnt))
            # 儲けが出たらwinとする
            win_cnt = 0
            win_rate = 0
            if bet_cnt != 0:

                open_spread_dict = {}
                close_spread_dict = {}
                open_sperad_list = []
                close_sperad_list = []
                for dh in deal_hist:
                    tmp_o_s = dh["spr"]
                    open_sperad_list.append(tmp_o_s)
                    if open_spread_dict.get(tmp_o_s) == None:
                        open_spread_dict[tmp_o_s] = 1
                    else:
                        open_spread_dict[tmp_o_s] += 1

                    tmp_c_s = dh["spr_end"]
                    close_sperad_list.append(tmp_c_s)
                    if close_spread_dict.get(tmp_c_s) == None:
                        close_spread_dict[tmp_c_s] = 1
                    else:
                        close_spread_dict[tmp_c_s] += 1
                """
                output("spread_open")
                for k, v in sorted(open_spread_dict.items()):
                    output(k, v, v / sum(open_sperad_list))

                output("")
                output("spread_close")
                for k, v in sorted(close_spread_dict.items()):
                    output(k, v, v / sum(close_sperad_list))
                """

                suffix_txt_tmp.append("")
                suffix_txt_tmp.append("average_spread_open:" + str(sum(open_sperad_list) / len(deal_hist)))
                suffix_txt_tmp.append("")
                suffix_txt_tmp.append("average_spread_close:" + str(sum(close_sperad_list) / len(deal_hist)))
                suffix_txt_tmp.append("")
                suffix_txt_tmp.append(
                    "average_spread:" + str((sum(open_sperad_list) + sum(close_sperad_list)) / (len(deal_hist) * 2)))

                d_np = np.array(pips)
                d_np = np.sort(d_np)

                d_np_up = np.array(pips_up)
                d_np_dw = np.array(pips_dw)

                win_cnt = len(np.where(d_np >= 0)[0])
                win_rate = win_cnt / len(d_np)
                # 勝ち数から負け数を引いて、純粋な勝ち数とする
                # win_cnt = win_cnt - (len(d_np) - win_cnt)

                win_cnt_up = len(np.where(d_np_up >= 0)[0])
                win_rate_up = win_cnt_up / len(d_np_up) if len(d_np_up) != 0 else 0

                win_cnt_dw = len(np.where(d_np_dw >= 0)[0])
                win_rate_dw = win_cnt_dw / len(d_np_dw) if len(d_np_dw) != 0 else 0

                if c.BINARY:
                    suffix_txt_tmp.append("win_cnt:" + str(wc))
                    suffix_txt_tmp.append("loose_cnt:" + str(lc))
                    suffix_txt_tmp.append("win_rate:" + str(wc / (wc + lc)))
                else:
                    suffix_txt_tmp.append("win_cnt:" + str(win_cnt))
                    suffix_txt_tmp.append("win_rate:" + str(win_rate))

                suffix_txt_tmp.append("pips length:" + str(len(d_np)))
                suffix_txt_tmp.append("pips avg:" + str(np.average(d_np)))
                suffix_txt_tmp.append("pips max:" + str(np.max(d_np)))
                suffix_txt_tmp.append("pips min:" + str(np.min(d_np)))

                suffix_txt_tmp.append("win_cnt_up:" + str(win_cnt_up))
                suffix_txt_tmp.append("win_rate_up:" + str(win_rate_up))
                suffix_txt_tmp.append("pips length up:" + str(len(d_np_up)))
                suffix_txt_tmp.append("pips avg up:" + str(np.average(d_np_up)))

                suffix_txt_tmp.append("win_cnt_dw:" + str(win_cnt_dw))
                suffix_txt_tmp.append("win_rate_dw:" + str(win_rate_dw))
                suffix_txt_tmp.append("pips length dw:" + str(len(d_np_dw)))
                suffix_txt_tmp.append("pips avg dw:" + str(np.average(d_np_dw)))

                if len(pips_sl) != 0:
                    d_np = np.array(pips_sl)
                    suffix_txt_tmp.append("sl_pips length:" + str(len(d_np)))
                    suffix_txt_tmp.append("sl_pips avg:" + str(np.average(d_np)))
                    suffix_txt_tmp.append("sl_pips max:" + str(np.max(d_np)))
                    suffix_txt_tmp.append("sl_pips min:" + str(np.min(d_np)))

                if len(pips_tp) != 0:
                    d_np = np.array(pips_tp)
                    suffix_txt_tmp.append("tk_pips length:" + str(len(d_np)))
                    suffix_txt_tmp.append("tk_pips avg:" + str(np.average(d_np)))
                    suffix_txt_tmp.append("tk_pips max:" + str(np.max(d_np)))
                    suffix_txt_tmp.append("tk_pips min:" + str(np.min(d_np)))

                if len(pips_sps) != 0:
                    d_np = np.array(pips_sps)
                    suffix_txt_tmp.append("sps_pips length:" + str(len(d_np)))
                    suffix_txt_tmp.append("sps_pips avg:" + str(np.average(d_np)))
                    suffix_txt_tmp.append("sps_pips max:" + str(np.max(d_np)))
                    suffix_txt_tmp.append("sps_pips min:" + str(np.min(d_np)))

                if len(tp_list) != 0:
                    tp_np = np.array(tp_list)
                    suffix_txt_tmp.append("take_profit avg:" + str(np.average(tp_np)))
                    suffix_txt_tmp.append("take_profit max:" + str(np.max(tp_np)))
                    suffix_txt_tmp.append("take_profit min:" + str(np.min(tp_np)))
                if len(sl_list) != 0:
                    sl_np = np.array(sl_list)
                    suffix_txt_tmp.append("stop_loss avg:" + str(np.average(sl_np)))
                    suffix_txt_tmp.append("stop_loss max:" + str(np.max(sl_np)))
                    suffix_txt_tmp.append("stop_loss min:" + str(np.min(sl_np)))

                if mode == "limit":
                    limit_np = np.array(limit_list)
                    suffix_txt_tmp.append("limit avg:" + str(np.average(limit_np)))
                    suffix_txt_tmp.append("limit max:" + str(np.max(limit_np)))
                    suffix_txt_tmp.append("limit min:" + str(np.min(limit_np)))

            profit_per_drawdown = 0
            tmp_drawdown = 0

            # print(max_drawdowns)

            if len(max_drawdowns) != 0:
                sorted_d = sorted(max_drawdowns.items(), key=lambda x: x[1], reverse=False)

                if sorted_d[0][1] != 0:
                    if c.BINARY:
                        profit_per_drawdown = int(detail_profit) / (int(sorted_d[0][1]) * -1)
                    else:
                        profit_per_drawdown = int(detail_profit) / (int(sorted_d[0][1]) * -1 + c.FX_FUND)

                tmp_drawdown = int(sorted_d[0][1])

            suffix_txt_tmp.append("profit_per_dd: " + str(profit_per_drawdown) + " " + str(tmp_drawdown))

            sl_bet_cnt = 0
            sl_cnt = len(pips_sl)
            if bet_cnt != 0:
                sl_bet_cnt = sl_cnt / bet_cnt

            result_per_suffix_border[
                str(suffix) + "-" + str(border) + "-" + str(ext_border) + "-" + str(border_ceil)] = {
                "profit_per_dd": profit_per_drawdown, "profit": int(detail_profit), "dd": tmp_drawdown,
                "bet_cnt": bet_cnt, "win_cnt": win_cnt, "win_rate": win_rate}

            # output(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), " Now Plotting")
            suffix_txt_tmp.append("sl/bet cnt:" + str(sl_bet_cnt))
            suffix_txt_tmp.append("")

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

            if show_profit_per_trade_sec:
                showPipsPerTradeSec(deal_hist)

            if show_profit_per_div:
                suffix_txt_tmp = showPipsPerDiv(deal_hist, show_profit_per_div_list, suffix_txt_tmp)

            if show_profit_per_div_abs:
                showPipsPerDivABS(deal_hist, show_profit_per_div_list)

            if len(show_h1_sma_list) != 0:
                suffix_txt_tmp.append("SMAデータ")
                sorted_d = sorted(deal_hist, key=lambda x: x['score'])

                for h in sorted_d:
                    output_list = [h["score"], h["stime"], h["etime"], h["sprice"], h["eprice"], h["type"],
                                   h["profit_pips"], ]

                    for m in show_h1_sma_list:
                        output_list.append(h["sma-" + str(m)])
                    suffix_txt_tmp.append(list_to_str(output_list, ","))

            for b in show_profit_per_boli_std:
                foot, length, alpha = b.split("-")
                col_name = "BOLI-" + foot + "-" + length + "-STD" + alpha

                suffix_txt_tmp = showPipsPerBoliStd(deal_hist, col_name, suffix_txt_tmp)

            for b in show_profit_per_boli_std_div:
                foot, length, alpha = b.split("-")
                col_name = "BOLI-" + foot + "-" + length + "-STD" + alpha

                suffix_txt_tmp = showPipsPerBoliStdDiv(deal_hist, col_name, suffix_txt_tmp)

            if show_profit_per_boli_std_div_history:
                suffix_txt_tmp.append("STDデータ")

                col_list = ["score","stime","etime","sprice","eprice","type","profit_pips"]
                for b in show_profit_per_boli_std_div:
                    foot, length, alpha = b.split("-")
                    col_name = "BOLI-" + foot + "-" + length + "-STD"
                    col_list.extend([col_name, col_name + alpha + "-UP-DIV", col_name + alpha + "-DW-DIV",col_name + alpha + "-MEAN-DIV"])

                suffix_txt_tmp.append(list_to_str(col_list, ","))

                sorted_d = sorted(deal_hist, key=lambda x: x['score'])

                for h in sorted_d:
                    output_list = [h["score"], h["stime"], h["etime"], h["sprice"], h["eprice"], h["type"],
                                   h["profit_pips"], ]

                    for b in show_profit_per_boli_std_div:
                        foot, length, alpha = b.split("-")
                        col_name = "BOLI-" + foot + "-" + length + "-STD"

                        output_list.append(h[col_name])
                        output_list.append(h[col_name + alpha + "-UP-DIV"])
                        output_list.append(h[col_name + alpha + "-DW-DIV"])
                        output_list.append(h[col_name + alpha + "-MEAN-DIV"])
                    suffix_txt_tmp.append(list_to_str(output_list, ","))

            if show_stoploss_history:
                showStoplossHistory(deal_hist)

            if show_history:
                output("決済履歴")
                sorted_d = sorted(deal_hist, key=lambda x: x['score'])

                for h in sorted_d:
                    # output(h["score"], h["sprice"], h["profit_pips"], h["spr"], h["spr_end"], h["trade_sec"])
                    output(h)

            if show_history_chart:
                sorted_d = sorted(deal_hist, key=lambda x: x['score'])

                chart_score = []

                for i, h in enumerate(sorted_d):
                    tmp_score = h["score"]
                    if tmp_score in chart_score:
                        # 既に登録済み
                        continue

                    chart_list = [h]
                    idx = i + 1
                    while True:
                        if idx >= len(sorted_d):
                            break
                        next = sorted_d[idx]
                        child_score = next["score"]
                        if child_score <= tmp_score + 3600:
                            # 今のデータから一時間以内なら一緒にチャート作成する
                            chart_list.append(next)
                            chart_score.append(child_score)

                        else:
                            break

                        idx = idx + 1

                    make_chart(chart_list, chart_save_dir)

            if show_high_profit_deal:
                showHighProfitDeal(deal_hist_dict)

            if show_detail:
                if len(max_drawdowns) != 0:
                    suffix_txt_tmp.append("MAX DrawDowns(理論上のドローダウン 40)")
                    sorted_d = sorted(max_drawdowns.items(), key=lambda x: x[1], reverse=False)

                    cnt_t = 1
                    for k, v in sorted_d:
                        if cnt_t > 40:
                            break
                        # output(v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))
                        suffix_txt_tmp.append(list_to_str((v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))))
                        cnt_t += 1

                if stoploss_max_day != None:
                    if len(profit_days) != 0:

                        for i, (k, v) in enumerate(profit_days.items()):
                            if i == 0:
                                # output("ゼロカットされた日付")
                                suffix_txt_tmp.append("ゼロカットされた日付")
                            if v <= stoploss_max_day:
                                # output(v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))
                                suffix_txt_tmp.append(
                                    list_to_str((v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))))

                    if loss_cut_percent != None:
                        suffix_txt_tmp.append("")
                        suffix_txt_tmp.append("ロスカットされた日付")
                        for k, v in profit_days_losscut.items():
                            # output(v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))
                            suffix_txt_tmp.append(
                                list_to_str((v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))))
                    if loss_cut_percent_day != None:
                        suffix_txt_tmp.append("")
                        suffix_txt_tmp.append("損失が多く取引停止した日付")

                        for k, v in profit_days_losscut_day.items():
                            # output(v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))
                            suffix_txt_tmp.append(
                                list_to_str((v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))))

                if stoploss_max_partial != None:
                    if len(profit_days_partial) != 0:
                        suffix_txt_tmp.append("")
                        suffix_txt_tmp.append("一時的に取引停止した日付")

                        for k, v in profit_days_partial.items():
                            suffix_txt_tmp.append(
                                list_to_str((v, datetime.fromtimestamp(k).strftime('%Y/%m/%d %H:%M:%S'))))

            if show_plot:
                fig = plt.figure(figsize=(6.4 * 0.7, 4.8 * 0.7))
                # 価格の遷移
                ax1 = fig.add_subplot(111)

                ax1.plot(plot_close_list, 'g')

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

            for i in suffix_txt_tmp:
                print(i)

            sub_txt.extend(suffix_txt_tmp)

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

    for i in sub_txt:
        output_file(i)


if __name__ == "__main__":

    start_ends = [
        #[datetime(2024, 12, 1), datetime(2024, 12, 14)],
        [datetime(2024, 12, 1), datetime(2026, 6, 27)],
        # [datetime(2025, 1, 1), datetime(2026, 5, 2)],
        # [datetime(2022, 1, 1), datetime(2026, 3, 24)],

    ]

    start_time = time.perf_counter()
    # output("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)

    # LSTMのテストの場合
    # conf = conf_class.ConfClass()

    # LGBMのテストの場合
    conf = conf_class_lgbm.ConfClassLgbm()

    conf.BATCH_SIZE = 1024 * 10

    if conf.FX == False:
        output("conf.FX == False !!!")
        exit(1)

    # conf.change_fx_real_spread_flg(True)

    # target_spread_list = []
    spread_confs = [
        [
            0,  # start_min_spread
            20,  # start_max_spread
            0,  # ex_min_spread
            20,  # ex_max_spread
            0,  # end_min_spread
            20,  # end_max_spread
            0,  # cannot_deal_cnt_max スプレッドが範囲外であることによる決済先延ばしを、この回数以上出来ない
        ],

    ]

    mode_conf_list = [
        {
            'mode': 'market',  # market:成行 gyaku_sashine:逆指値
            'pending_pips': None,
            'pending_max_sec': None,
            'deal_mode': 'market',
            'deal_pending_pips': None,
            'deal_pending_max_cnt': None,
        },
        # {
        #    'mode': 'gyaku_sashine',  # market:成行 gyaku_sashine:逆指値
        #    'pending_pips': 0.008,
        #    'pending_max_sec': 25,
        #    'deal_mode': 'market',
        #    'deal_pending_pips': None,
        #    'deal_pending_max_cnt': None,
        # },

    ]
    div_conf_list = [
        #{
        #    'min_div_sec': 300,
        #    'min_div_ext_sec': 300,
        #    'min_div': None,  # None:設定なし
        #    'min_div_ext': None,  # None:設定なし
        #    'min_div_minus': None,  # None:設定なし
        #    'min_div_ext_minus': None,  # None:設定なし
        #    'max_div_sec': 300,
        #    'max_div_ext_sec': 300,
        #    'max_div': None,  # None:設定なし
        #    'max_div_ext': None,  # None:設定なし
        #    'max_div_minus': None,  # None:設定なし
        #    'max_div_ext_minus': None,  # None:設定なし
        #},
        #{
        #    'min_div_sec': 300,
        #    'min_div_ext_sec': 300,
        #    'min_div': 5,  # None:設定なし
        #    'min_div_ext': 5,  # None:設定なし
        #    'min_div_minus': -5,  # None:設定なし
        #    'min_div_ext_minus': -5,  # None:設定なし
        #    'max_div_sec': 300,
        #    'max_div_ext_sec': 300,
        #    'max_div': 80,  # None:設定なし
        #    'max_div_ext': 80,  # None:設定なし
        #    'max_div_minus': -35,  # None:設定なし
        #    'max_div_ext_minus': -35,  # None:設定なし
        #},
         {
            'min_div_sec': 300,
            'min_div_ext_sec': 300,
            'min_div': 5,  # None:設定なし
            'min_div_ext': 5,  # None:設定なし
            'min_div_minus': -5,  # None:設定なし
            'min_div_ext_minus': -5,  # None:設定なし
            'max_div_sec': 300,
            'max_div_ext_sec': 300,
            'max_div': 80,  # None:設定なし
            'max_div_ext': 80,  # None:設定なし
            'max_div_minus': -40,  # None:設定なし
            'max_div_ext_minus': -40,  # None:設定なし
         },
    ]

    other_conf_list = [
        {
            'takeprofit_dict': None,  # None:設定なし
            'stoploss_dict': {
                'type':'std','multi': 4,'std_name':'BOLI-M1-5-STD', 'no_data_pips': 0.5,
                #'type':'fix', 'pips':0.5,
            }, #None:設定なし
            'stoploss_max': 0.5,  # 許容する最大ストップロス(円)
            'stoploss_max_day': -600000,  # 一日に許容する最大損失(円) None:設定なし
            'stoploss_break_recovery': None,  # 最大損失を超えてロスカットされてから取引再開するまでの秒数 None:設定なし(翌日に再開する)
            'stoploss_trail': False,  # True:trail設定する
            'loss_cut_percent': None,  # None:設定なし
            'loss_cut_percent_day': None,  # None:設定なし
            'trial_trade_flg': True,  # 試しに1ポジションだけ取引して様子をみる
            'trial_trade_sec': 4,  # 試しに1ポジションだけにする秒数
            'trial_trade_pips_min': -0.15,  # 試しに1ポジション取引したときに、このpips未満なら後続は取引しない
            'trial_trade_lookup_sec': 90,  # 試しに1ポジション取引したときに、後続がその結果を参照する秒数
            'trial_trade_pips_update_sec': 90,  # 試しに1ポジション取引したときに、その結果を更新しつづける秒数
            'traial_trade_deal_reset': False, # True:トライアルポジション決済時にリセットする(ストップロスの場合はリセットしない)
            'low_spread': None,  # このスプレッド以下ならlow_spread_border以上あれば取引する None:設定なし
            'low_spread_border': 0.55,  # このスプレッド以下なら None:設定なし
            'trial_trade_stoploss': None,  # 試しに1ポジションだけにする場合のstoploss.損失を少なくする None:設定なし
            'trial_trade_position': None,  # 試しに1ポジションだけにする場合のポジション None:設定なし
            'emerg_div': None,  # このdivになったら全決済してemerg_stop_sec秒の間は取引しない None:設定なし
            'emerg_stop_sec': 600,
            'emerg_div_sec': 300,
            'most_high_low_div_flg': False,  # 新規発注時に参照するdivの期間で最高値と最安値のdivを取る
            'stoploss_max_partial': None,  # 一時的に取引停止する損失(円) None:設定なし
            'stoploss_break_recovery_partial': 3600,  # stoploss_max_partialを超えてから取引再開するまでの秒数
            'stoploss_short_sec': None,  # この秒数経過時に指定損失が出ていた場合は損切りする None:設定なし
            'stoploss_short_pips': -0.1,  # stoploss_short_sec秒数経過時にこの損失が出ていた場合は損切りする
            'ignore_spread': False,  # True:DBのスプレッドを無視する
            #'boli_ng_range_buy': [],
            'boli_ng_range_sell': [],
            'boli_ng_range_buy': [{"foot":"H1-20-3-MEAN", "up":-50, "dw":-999}], #空配列:設定なし 辞書リストでNG対象の足の設定とレンジを指定する foot:H1-20-3-MEAN or H1-20-3-UP or H1-20-3-DW, up:-50, dw:-80
            #'boli_ng_range_sell': [{"foot":"H1-20-3-MEAN", "up":100, "dw":50}],# 空配列:設定なし 辞書リストで対象の足の設定とレンジを指定する foot:H1-20-3-MEAN or H1-20-3-UP or H1-20-3-DW, up:-50, dw:-80
            'stoploss_lookup_sec': None,  # ポジションがストップロスの場合、後続がその結果を参照する秒数 None:設定なし
        },

    ]

    """
    refer_dict_list = [
        {
            "refer_tick_sec": 0, #0:設定なし
            "refer_cnt": 0,
            "refer_tick_sec_ext": 0,
            "refer_ext_cnt": 0,
        },
    ]
    """

    refer_dict_list = [

        {
            "refer_tick_sec": 0,
            "refer_cnt": 2,
            "refer_tick_sec_ext": 0,
            "refer_ext_cnt": 0,
            "refer_tick_cnt_0_ng": False,  # True:tickの数が0なら取引しない
            "refer_vs": True,  # True:掛けたい方向のtick数が相手より多くなければ取引しない
        },

    ]

    refer_past_pred_conf_list = [

        {
            "refer_past_pred_sec": 0,  # 指定された過去秒の予想を参照。0なら参照しない
            "refer_past_pred": 0.4,  # 指定された過去秒の予想閾値
        },

    ]

    change_stoploss_conf_list = [
        {
            "change_stoploss_term": None,  # None:設定なし
            "change_stoploss_price": 0.01
        },
    ]

    # モデルの親と子をリストにする
    model_list = [
        # ["MN2074", "164"],
        # ["MN1504", "81"],
        ["MN2009", "14"],
        # ["MN2074", "164"],
    ]

    conf.ADJUST_PIPS = 0.0
    # conf.ADJUST_PIPS = -0.002

    conf.FX_SINGLE_FLG = False
    conf.FX_NOT_EXT_FLG = False

    conf.RESTRICT_FLG = True

    conf.RESTRICT_SEC = 1
    conf.RESTRICT_END_SEC = 1
    conf.BUY_FLG = True
    conf.SELL_FLG = True

    conf.FX_MAX_POSITION_CNT = 30
    conf.TERM = 30
    conf.NG_SHIFT = 30

    conf.FX_LEVERAGE = 500

    conf.FX_FIX_POSITION = int(get_decimal_divide(1800000, conf.FX_MAX_POSITION_CNT))

    if conf.FX_LEVERAGE == 25:
        conf.FX_FIX_POSITION = int(get_decimal_divide(90000, conf.FX_MAX_POSITION_CNT))

    conf.FX_NOT_EXT_MINUS = None
    conf.FX_MAX_TRADE_SEC = None  # 最大取引時間 設定しない場合はNone

    conf.NG_SHIFT_MAX = 1  # 同じシフトでの最大ポジション数
    conf.TRADE_SHIFT = 1

    conf.BINARY = False
    conf.BINARY_GYAKUSASHINE_SURVIVE = True  # True:逆指値成立時に同時に他の建玉も成立した場合に建玉を残す False:建玉をキャンセル

    conf.PAYOUT = 850
    conf.PAYOFF = -1000
    conf.AtMoney = False

    if conf.BINARY:
        conf.RESTRICT_FLG = False
        conf.FX_NOT_EXT_FLG = True

    conf.POSITION_BY_PRED = False  # True:予想確率が高いほどポジション数を多くする
    conf.POSITION_BY_PRED_LIST = [
        [0, 0.6, 40000],
        [0.6, 0.61, 50000],
        [0.61, 0.62, 60000],
        [0.62, 0.63, 70000],
        [0.63, 0.64, 80000],
        [0.64, 1.0, 90000],
    ]

    conf.EXCEPT_LIST_HOUR_TEST = [20, 21, 22, 23]

    conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST = {

    }

    conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY = {
        "score": "640",
        "save_dir_path": "/nvme2/dataSequence2/USDJPY/DS2F640-0",
    }

    for start_end in start_ends:
        start, end = start_end

        for spread_conf in spread_confs:
            for mode_conf in mode_conf_list:
                for change_stoploss_conf in change_stoploss_conf_list:
                    for refer_dict in refer_dict_list:
                        for refer_past_pred_conf in refer_past_pred_conf_list:
                            for div_conf in div_conf_list:
                                for model_file in model_list:
                                    for other_conf in other_conf_list:

                                        do_predict(conf, start, end, model_file, spread_conf,
                                                   mode_conf=mode_conf,
                                                   div_conf=div_conf, refer_dict=refer_dict,
                                                   change_stoploss_conf=change_stoploss_conf,
                                                   refer_past_pred_conf=refer_past_pred_conf,
                                                   other_conf=other_conf, )

    print("Processing Time(Sec)", time.perf_counter() - start_time)

    print("END!!!")
    # 終わったらメールで知らせる
    mail.send_message(host, ": testLstmFX2_rgr_limit finished!!!")
