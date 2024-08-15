import numpy as np
import redis
import json
from matplotlib import pyplot as plt
from datetime import datetime
import time
import conf_class
from util import *
import copy
from silence_tensorflow import silence_tensorflow
silence_tensorflow() #ログ抑制 import tensorflowの前におく

import tensorflow as tf
import socket
from DataSequence2 import DataSequence2

png_dir = "/app/bin_op/png/"
host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

"""
LEARNING_TYPE == "CATEGORY"
LEARNING_TYPE == "CATEGORY_BIN_UP"
LEARNING_TYPE == "CATEGORY_BIN_DW"
の場合専用
"""
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

c = None

#2秒ごとの成績を計算する場合
per_sec_flg = True

#border以上の予想パーセントをしたものから正解率と予想数と正解数を返す

def getAcc(res, border, dataY, bin_both_ind_up, bin_both_ind_dw):
    global c

    if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN":
        up_ind = np.where((res[:, 0] >= res[:, 2]) & (res[:, 0] >= border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] >= border))[0]
    elif c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        up_ind = bin_both_ind_up
        down_ind = bin_both_ind_dw
    elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
        up_ind = np.where(res[:, 0] >= border)[0]
        down_ind = []
    elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
        up_ind = []
        down_ind = np.where(res[:, 0] >= border)[0]

    x5_up = res[up_ind,:]
    if c.LEARNING_TYPE == "CATEGORY_BIN_UP":
        x5_up[:,0] = 1

    y5_up= dataY[up_ind,:]

    x5_down = res[down_ind,:]
    if c.LEARNING_TYPE == "CATEGORY_BIN_DW":
        x5_down[:,0] = 1

    y5_down= dataY[down_ind,:]

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

    return Acc, total_num, correct_num

#全体の正解率を返す(SAME予想も含める)
def getAccTotalSame(res, dataY):
    global c

    eq = np.equal(res.argmax(axis=1), dataY.argmax(axis=1))
    cor_length = int(len(np.where(eq == True)[0]))

    total_num = len(res)
    correct_num = cor_length

    if total_num ==0:
        Acc =0
    else:
        Acc = correct_num / total_num

    return Acc

def getAccPerBorder(res, dataY, border):
    global c
    shape = res[0].shape[0] #CATEGORY_BIN_BOTHなら3列、CATEGORY_BIN_UPやCATEGORY_BIN_DWなら2列
    if shape == 2:
        up_down_ind = np.where(res[:, 0] > border)[0]
        x5_up_down = res[up_down_ind, :]
        y5_up_down = dataY[up_down_ind, :]

        up_down_eq = np.equal(x5_up_down.argmax(axis=1), y5_up_down.argmax(axis=1))
        up_down_cor_length = int(len(np.where(up_down_eq == True)[0]))

        total_num = len(up_down_eq)
        correct_num = up_down_cor_length

    elif shape == 3:
        up_ind = np.where((res[:, 0] > res[:, 2]) & (res[:, 0] > border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0])  & (res[:, 2] > border))[0]

        x5_up = res[up_ind, :]
        y5_up = dataY[up_ind, :]

        x5_down = res[down_ind, :]
        y5_down = dataY[down_ind, :]

        up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
        up_cor_length = int(len(np.where(up_eq == True)[0]))
        down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
        down_cor_length = int(len(np.where(down_eq == True)[0]))

        total_num = len(up_ind) + len(down_ind)
        correct_num = up_cor_length + down_cor_length

    acc = 0

    if total_num !=0:
        acc = correct_num/total_num

    return total_num, correct_num, acc

def getAccTotal(res, dataY):
    global c
    if c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW":

        idxa = np.where(res.argmax(axis=1) == 0)[0] #1列目が最大のインデックスを返す
        idxb = np.where(dataY[idxa].argmax(axis=1) == 0)[0] #最大と予想したインデックスの中での正解のインデックスを返す

        total_num = len(idxa)
        correct_num = len(idxb)

        if total_num ==0:
            Acc =0
        else:
            Acc = correct_num / total_num
    else:
        idxa1 = np.where(res.argmax(axis=1) == 0)[0]  # 1列目が最大のインデックスを返す
        idxb1 = np.where(dataY[idxa1].argmax(axis=1) == 0)[0]  # 最大と予想したインデックスの中での正解のインデックスを返す

        idxa2 = np.where(res.argmax(axis=1) == 2)[0]  # 3列目が最大のインデックスを返す
        idxb2 = np.where(dataY[idxa2].argmax(axis=1) == 2)[0]

        total_num = len(idxa1) + len(idxa2)
        correct_num = len(idxb1) + len(idxb2)

        if total_num == 0:
            Acc = 0
        else:
            Acc = correct_num / total_num

    return str(total_num), str(Acc)

def getAccFx(res, border, dataY, change):
    global c
    if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN":
        up_ind = np.where((res[:, 0] > res[:, 2]) & (res[:, 0] > border))[0]
        down_ind = np.where((res[:, 2] > res[:, 0]) & (res[:, 2] > border))[0]

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

    c5_sum = c5_up_sum + c5_down_sum - ((len(c5_up) + len(c5_down)) * float( Decimal("0.001") * Decimal(c.SPREAD) ))
    c5_sum = c5_sum * c.FX_POSITION

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
        #最大ドローダウンを更新してしまった場合
        max_drawdown = drawdown

    if drawdown > 0:
        if max_drawdown != 0:
            max_drawdowns.append(max_drawdown)
        drawdown = 0
        max_drawdown = 0

    return max_drawdown, drawdown

def make_data(conf, start, end, target_spreads=[], spread_correct=None, sub_force = False):
    global c
    c = conf

    dataSequence2 = DataSequence2(conf, start, end, True, False, target_spread_list=target_spreads, spread_correct=spread_correct, sub_force = sub_force)
    return dataSequence2

def make_eval_data(dataSequence2):
    dataSequence2_eval = copy.deepcopy(dataSequence2)
    dataSequence2_eval.change_eval_flg(True)
    return dataSequence2_eval

def show_win_atr(col, win_list, bet_list):
    col_val_list = []
    win_rate_list = []
    profit_list = []

    tpips = float(Decimal(str(c.PIPS)) * Decimal("10"))
    if ("satr" in col) == False:
        tpips = 1

    prev_atr = 0
    for a in range(20):
        tmp_atr = float(Decimal(str(tpips)) * Decimal(str((a + 1))))

        w_cnt = len(np.where((win_list >= prev_atr) & (win_list < tmp_atr))[0])
        total_cnt = len(np.where((bet_list >= prev_atr) & (bet_list < tmp_atr))[0])

        if total_cnt != 0:
            profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
            win_rate = w_cnt / total_cnt
            output(prev_atr, "~", tmp_atr, " rate:", win_rate, " profit:", profit," total_cnt:", total_cnt)

            col_val_list.append(prev_atr)
            win_rate_list.append(win_rate)
            profit_list.append(profit)

        prev_atr = tmp_atr


    tpips = float(Decimal(str(c.PIPS)) * Decimal("100"))
    if ("satr" in col) == False:
        tpips = 10

    prev_atr = 0
    for a in range(10):
        tmp_atr = float(Decimal(str(tpips)) * Decimal(str((a + 1))))

        w_cnt = len(np.where((win_list >= prev_atr) & (win_list < tmp_atr))[0])
        total_cnt = len(np.where((bet_list >= prev_atr) & (bet_list < tmp_atr))[0])

        if total_cnt != 0:
            profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
            win_rate = w_cnt / total_cnt
            output(prev_atr, "~", tmp_atr, " rate:", win_rate, " profit:", profit, " total_cnt:", total_cnt)

        prev_atr = tmp_atr

    return [col_val_list, win_rate_list, profit_list]

def show_win_ind(conf, ind_list, train_list_index, ind, bet_ind, win_ind, show_plot, save_dir=None):

    target_ind_list = ind_list[np.where(train_list_index != -1)[0], :]
    target_ind_list = target_ind_list[ind,:]
    for i, col in enumerate(conf.IND_COLS):
        output(col + "毎の勝率")

        tmp_target_col_list = target_ind_list[:,i]

        bet_list = np.array(tmp_target_col_list[bet_ind], dtype='float32')
        win_list = np.array(tmp_target_col_list[win_ind], dtype='float32')

        if "atr" in col :
            col_val_list, win_rate_list, profit_list = show_win_atr(col, win_list, bet_list)
        else:
            #avg = np.average(bet_list)
            #std = np.std(bet_list)
            #max_ind = float('{:.4f}'.format(avg + std * 2))
            #min_ind = float('{:.4f}'.format(avg - std * 2))
            max_ind =  float('{:.4f}'.format(max(bet_list)))
            min_ind =  float('{:.4f}'.format(min(bet_list)))
            width = float('{:.4f}'.format((max_ind - min_ind) / 20))

            col_val_list = []
            win_rate_list = []
            profit_list = []
            range_start = min_ind
            for j in range(20):
                range_next = float('{:.4f}'.format(range_start + width))
                w_cnt = len(np.where((win_list >= range_start) & (win_list < range_next))[0])
                total_cnt = len(np.where((bet_list >= range_start) & (bet_list < range_next))[0])

                if total_cnt != 0:
                    profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
                    win_rate = w_cnt / total_cnt
                    output(range_start, "~", range_next, " rate:", win_rate, " profit:", profit," total_cnt:", total_cnt)

                    col_val_list.append(range_start)
                    win_rate_list.append(win_rate)
                    profit_list.append(profit)

                range_start = range_next


        if show_plot:
            fig, ax1 = plt.subplots(figsize=(6.4*0.45, 4.8*0.45))
            ax1.plot(col_val_list, win_rate_list, "b-")
            ax1.set_ylabel("win_rate")

            ax2 = ax1.twinx()

            ax2.plot(col_val_list, profit_list, "r-")
            ax2.set_ylabel("profit")

            plt.title('show_win_ind col:' + col)

            filename = save_dir + "/" + 'show_win_ind col:' + col + ".png"
            fig.savefig(filename)

def do_predict(conf, dataSequence2, target_spreads, test_eval=False):
    global c
    c = conf

    if test_eval:
        dataSequence2_eval = make_eval_data(dataSequence2)

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

    # 全ind値のリスト(Noneあり)
    """
    for i in dataSequence2.get_ind_list():
        if i == None:
            print(i)
        elif len(i) != 1:
            print(i)
    """
    ind_list = np.array(dataSequence2.get_ind_list())
    #print(ind_list.dtype, ind_list.shape)
    # 予想対象のatrのリスト
    target_atr_list = atr_list[np.where(train_list_index != -1)[0]]

    model_suffix = []
    for i in range(c.EPOCH):
        model_suffix.append(str(i + 1))


    # GBPJPY turbe spread
    cat_bin_both_models = [
            {
            2: [
                "GBPJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD2_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-36",
                "GBPJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD3_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-36",
            ],
            3: [
                "GBPJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD5_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-5",
                "GBPJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD4_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-11",
            ],
            4: [
                "GBPJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD5_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-39",
                "GBPJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD3_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-37",
            ],
            5: [
                "GBPJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD4_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-5",
                "GBPJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_SPREAD2_201001_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-sub_OD-c_IDL1_BS12288_SEED0_SHUFFLE_ub1-13",
            ],
            }
    ]

    """
    #target_spreadをkey,Valueは[UP,DW]それぞれのモデル
    #USDJPY turbo

    cat_bin_both_models = [{0: [
        "USDJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub2-28",
        "USDJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub1-40",
        ]
    }]
    """

    """
    #EURUSD
    cat_bin_both_models = [{0: [
        "USDJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub2-28",
        "USDJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub1-40",
        ]
    }]
    """

    # CATEGORY_BIN_BOTH用 スプレッドに対応する予想結果がない場合に対応させるtarget_spreadsのインデックス
    #GBPJPY for turbo spread
    idx_other = 8 + 1  # リストの最初にtarget_spread_listが入っているのでプラス1する

    #USDJPY for turbo
    #idx_other = 1 + 1
    #border_list = [   0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63,0.64, ]  # Turbo用(category_bin)
    #border_list = [   0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, ]  # Turbo用
    #border_list = [   0.53, 0.54, 0.55,  0.56, 0.57, 0.58, 0.59, 0.6, 0.61, ]  # Turbo spread2用

    #border_list = [0.52,0.54,0.56,]  # Turbo div0.1
    #border_list = [0.46,0.47, 0.48,0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,  0.56,]  # TurboSpread 1用
    #border_list = [ 0.45, 0.46,0.47, 0.48,0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, ]   # TurboSpread 1 BIN_UP,BIN_DW用
    #border_list = [0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, ]

    #border_list = [ 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52  ]   # TurboSpread 8 BIN_UP,BIN_DW用
    #border_list = [0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53 ]  # TurboSpread 8, 9 BIN_UP,BIN_DW用

    border_list_show = []

    #CATEGORY_BIN_BOTH用

    """
    #USDJPY
    border_list_both = [
        {-1: [0.50, 0.50], 0: [0.6, 0.6],  },
    ]
    """


    #GBPJPY
    border_list_both = [{-1: [0.5, 0.5], 2: [0.49, 0.49], 3: [0.47, 0.5], 4: [0.49, 0.52], 5: [0.51, 0.53]}]


    # BIN_BOTH用 それぞれのスプレッド対応モデルごとのborder(UP,DW) -1スプレッドは対応するモデルがない場合用
    border_list_show_both = border_list_both

    """
    FILE_PREFIXS = [
        #"USDJPY_CATEGORY_BIN_UP_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub2",
        "USDJPY_CATEGORY_BIN_DW_LSTM7_TYPE-LSTM1_BET2_TERM30_INPUT2-10-30_INPUT_LEN300-300-240_L-UNIT25-25-20_D-UNIT48-24-12_LSTMDO0.05_DROP0.05_BNL2_BDIV0.01_201011_202210_L-RATE0.001_LOSS-C-ENTROPY_ADAM_d1-M1_OT-d_OD-c_IDL1_BS10240_SEED0_SHUFFLE_ub1",

    ]   
    """

    #model_suffix = ["32",]
    #border_list = [ 0.5, ]
    #border_list_show = border_list

    # BIN_BOTH用以外用
    #FILE_PREFIXS = [c.FILE_PREFIX]

    # line_val = 0.505
    line_val = 0.572 #pyoutが100なので
    #line_val = 0.582 #pyoutが960なので

    show_plot = True
    if show_plot:
        # png保存用のディレクトリ作成
        save_dir = png_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
        makedirs(save_dir)
        output("PNG SAVE DIR:", save_dir)

    if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
        model_suffix = cat_bin_both_models
        border_list = border_list_both
        border_list_show = border_list_show_both
        FILE_PREFIXS = ["CATEGORY_BIN_BOTH"]  # FILE_PREFIXはCATEBIN_BOTHで使用しないので適当にいれる
        output("border_list_both", border_list_both)

    for file in FILE_PREFIXS:
        output("FILE_PREFIX:", file)
        output("line_val:", line_val)
        output("target_spreads",target_spreads)
        output("ATR_COL", c.ATR_COL)
        output("ATR", c.ATR)
        output("IND_COLS", c.IND_COLS)
        output("IND_RANGES", c.IND_RANGES)
        if c.RESTRICT_FLG:
            output("restrict_sec", c.RESTRICT_SEC)

        output("sub_force", dataSequence2.sub_force)
        if dataSequence2.sub_force == True:
            output("SPREAD", c.SPREAD,)

        output("EXCEPT_LIST", c.EXCEPT_LIST)

        total_bet_txt = []
        total_acc_txt = []
        total_money_txt = []
        total_accuacy_txt = []
        total_loss_txt = []
        max_val_suffix = {"val":0,}

        for suffix in model_suffix:
            if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                #target_spreadのリスト
                target_spreads = []
                dic = sorted(suffix.items())

                #zip関数に渡すスプレッドと予想結果のリスト
                arg_lists = [target_spread_list]
                output("target_spread_list length:", len(target_spread_list))
                for k, v in dic:
                    target_spreads.append(k)

                    #CATEGORY_BIN_UP とDWで予想した結果を合わせる
                    load_dir_up = "/app/model/bin_op/" + v[0]
                    model_up = tf.keras.models.load_model(load_dir_up)

                    load_dir_dw = "/app/model/bin_op/" + v[1]
                    model_dw = tf.keras.models.load_model(load_dir_dw)

                    # ndarrayで返って来る
                    predict_list_up = model_up.predict_generator(dataSequence2,
                                                           steps=None,
                                                           max_queue_size=c.MAX_QUEUE_SIZE * 1,
                                                           use_multiprocessing=False,
                                                           verbose=0)

                    predict_list_dw = model_dw.predict_generator(dataSequence2,
                                                           steps=None,
                                                           max_queue_size=c.MAX_QUEUE_SIZE * 1,
                                                           use_multiprocessing=False,
                                                           verbose=0)
                    #SAMEの予想結果は0とする
                    predict_list_zero = np.zeros((len(predict_list_up), 2))

                    #UP,SAME,DWの予想結果を合算する
                    all = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
                    predict_list_tmp = all[:, [0, 2, 4]]
                    arg_lists.append(predict_list_tmp)
                    #print("predict_list_tmp length:",k, len(predict_list_tmp))
                #各スプレッドに対応する予想結果をまとめる
                predict_list = []


                output("target_spreads",target_spreads)
                """
                print("arg_lists[:20]")
                print(arg_lists[:20])
                """

                for j in zip(*arg_lists): #arg_listsを展開して渡す
                    tmp_spr = j[0]
                    if tmp_spr in target_spreads:
                        #target_spreadsの何番目に予想結果が入っているか取得
                        idx = target_spreads.index(tmp_spr) + 1 #リストの最初にtarget_spread_listが入っているのでプラス1する
                        predict_list.append(j[idx].tolist()) #numpy.arrayを一旦リストにして追加
                    else:
                        #スプレッドに対応する予想結果がない場合
                        predict_list.append(j[idx_other].tolist())

                predict_list = np.array(predict_list) #listからnumpy.arrayにもどす
                #print(predict_list[:20])

            elif c.LEARNING_TYPE == "CATEGORY_BIN":
                load_dir = "/app/model/bin_op/" + file + "-" + suffix
                if not os.path.isdir(load_dir):
                    #print("model not exists:" + load_dir )
                    continue

                model = tf.keras.models.load_model(load_dir)
                if test_eval:
                    loss, accuracy = model.evaluate_generator(dataSequence2_eval,
                                                           steps=None,
                                                           max_queue_size=c.PROCESS_COUNT * 1,
                                                           use_multiprocessing=False,
                                                           verbose=0)

                    total_accuacy_txt.append(str(accuracy))
                    total_loss_txt.append(str(loss))

                # ndarrayで返って来る
                predict_list = model.predict_generator(dataSequence2,
                                                       steps=None,
                                                       max_queue_size=c.PROCESS_COUNT * 1,
                                                       use_multiprocessing=False,
                                                       verbose=0)
                #SAMEの予想結果は0とする
                predict_list_zero = np.zeros((len(predict_list), 2))
                #UP,SAME,DWの予想結果を合算する
                all = np.concatenate([predict_list, predict_list_zero,], 1)
                predict_list = all[:, [0, 2, 1]]

            else:
                #suffix = "90*" + suffix

                load_dir = "/app/model/bin_op/" + file + "-" + suffix
                if not os.path.isdir(load_dir):
                    print("model not exists:" + load_dir )
                    continue

                model = tf.keras.models.load_model(load_dir)
                if test_eval:
                    loss, accuracy = model.evaluate_generator(dataSequence2_eval,
                                                           steps=None,
                                                           max_queue_size=c.PROCESS_COUNT * 1,
                                                           use_multiprocessing=False,
                                                           verbose=0)

                    total_accuacy_txt.append(str(accuracy))
                    total_loss_txt.append(str(loss))

                # ndarrayで返って来る
                predict_list = model.predict_generator(dataSequence2,
                                                       steps=None,
                                                       max_queue_size=c.PROCESS_COUNT * 1,
                                                       use_multiprocessing=False,
                                                       verbose=0)
            #model.summary()
            #print(predict_list[:20])
            output("suffix:", suffix)

            # START tensorflow1で作成したモデル用
            """
            model_file = "/app/bin_op/model/GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5" + "." + suffix
            model = tf.keras.models.load_model(model_file)
            model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
            """
            # END tensorflow1で作成したモデル用

            under_dict = {}
            over_dict = {}


            """
            print("close", len(close_list), close_list[:10])
            print("score", len(score_list), score_list[:10])
            print("target_score", len(target_score_list), target_score_list[:10])
            print("pred_close", len(pred_close_list), pred_close_list[:10])
            """

            #print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "Predict finished!! Now Calculating")

            r = redis.Redis(host='localhost', port=6379, db=c.DB_TRADE_NO)

            betStr, accStr  = getAccTotal(predict_list, correct_list)
            total_bet_txt.append(betStr)
            total_acc_txt.append(accStr )

            for border_ind, border in enumerate(border_list):

                # 予想結果表示用テキストを保持
                result_txt = []
                result_txt_trade = []

                max_drawdown = 0
                drawdown = 0
                max_drawdowns = []

                max_drawdown_trade = 0
                drawdown_trade = 0
                max_drawdowns_trade = []

                #BIN_BOTHの場合、borderが一律でない為、先に対象indを求めておく
                bin_both_ind = []
                bin_both_ind_up = []
                bin_both_ind_dw = []
                if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":

                    for l, m in enumerate(zip(target_spread_list, predict_list)):
                        spr_t = m[0]
                        pred_t = m[1]

                        #print("spr_t",spr_t)
                        #print("border",border)
                        if spr_t in border:
                            up_border = border[spr_t][0]
                            dw_border = border[spr_t][1]
                        else:
                            up_border = border[-1][0]
                            dw_border = border[-1][1]
                        if pred_t[0] >= pred_t[2] and  pred_t[0] >= up_border:
                            bin_both_ind_up.append(l)
                            bin_both_ind.append(l)
                        elif pred_t[2] > pred_t[0] and pred_t[2] >= dw_border:
                            bin_both_ind_dw.append(l)
                            bin_both_ind.append(l)

                if c.FX == False:
                    Acc, total_num, correct_num = getAcc(predict_list, border, correct_list, bin_both_ind_up, bin_both_ind_dw)
                    profit = (c.PAYOUT * correct_num) - ((total_num - correct_num) * c.PAYOFF)
                else:
                    Acc, total_num, correct_num = getAcc(predict_list, border, correct_list, bin_both_ind_up, bin_both_ind_dw)

                #全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
                if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                    result_txt.append("Accuracy border_ind " + str(border_ind) + ":" + str(Acc))
                    result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))
                else:
                    result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                    result_txt.append("Total:" + str(total_num) + " Correct:" + str(correct_num))

                if c.FX == False:
                    result_txt.append("Earned Money:" + str(profit))

                    if line_val >= Acc:
                        if not "acc" in over_dict.keys():
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

                if (c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW" ):
                    # up,downどれかが閾値以上の予想パーセントであるもののみ抽出
                    ind = np.where(predict_list[:, 0] >= border)[0]
                elif c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                    #up,downどちらかが閾値以上
                    ind = bin_both_ind
                else:
                    # up,same,downどれかが閾値以上の予想パーセントであるもののみ抽出
                    ind = np.where(predict_list >=border)[0]

                x5 = predict_list[ind,:]
                y5 = correct_list[ind,:]
                s5 = target_score_list[ind]
                c5 = change_list[ind]
                sp5 = target_spread_list[ind]
                sc5 = pred_close_list[ind]
                ec5 = real_close_list[ind]
                dp5 = target_divide_prev_list[ind]
                da5 = target_divide_aft_list[ind]
                atr5 = target_atr_list[ind]

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
                money_trade_y = []
                money_tmp = {}
                money_trade_tmp = {}

                money = c.START_MONEY #始めの所持金
                money_trade = c.START_MONEY

                cnt_up_cor = 0 #upと予想して正解した数
                cnt_up_wrong = 0 #upと予想して不正解だった数

                cnt_down_cor = 0 #downと予想して正解した数
                cnt_down_wrong = 0 #downと予想して不正解だった数

                spread_trade = {}
                spread_win = {}
                spread_trade_up = {}
                spread_win_up = {}
                spread_trade_dw = {}
                spread_win_dw = {}

                spread_trade_real = {}
                spread_win_real = {}
                spread_trade_real_up = {}
                spread_win_real_up = {}
                spread_trade_real_dw = {}
                spread_win_real_dw = {}

                divide_trade = {}
                divide_win = {}

                divide_trade_real = {}
                divide_win_real = {}

                divide_aft_trade = {}
                divide_aft_win = {}

                divide_aft_trade_real = {}
                divide_aft_win_real = {}

                cnt = 0
                win_cnt = 0

                trade_cnt_by_day = {}

                # 2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
                # {key:[trade_num,win_num]}
                per_sec = 2
                per_sec_dict = {}
                per_sec_dict_real = {}
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

                # 時間ごとの成績
                per_hour_dict = {}
                per_hour_dict_real = {}
                for i in range(24):
                    per_hour_dict[i] = [0, 0]
                    per_hour_dict_real[i] = [0, 0]

                # 理論上の予測確率ごとの勝率 key:確率 val:{win_cnt:勝った数, lose_cnt:負けた数}
                prob_list = {}
                # 実際のトレードの予測確率ごとの勝率 key:確率 val:{win_cnt:勝った数, lose_cnt:負けた数}
                prob_real_list = {}

                true_cnt = 0

                trade_cnt = 0
                trade_win_cnt = 0
                trade_wrong_win_cnt = 0
                trade_wrong_lose_cnt = 0

                bet_ind = []
                win_ind = []

                atr_win_list = []
                atr_lose_list = []
                atr_total_list = []

                prev_trade_time = 0

                for i, (x, y, s, cl, sp, sc, ec, dp, da, atr) in enumerate(zip(x5, y5, s5, c5, sp5, sc5, ec5, dp5, da5, atr5)):

                    max = x.argmax()
                    if (c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW"):
                        max = 0

                    probe_float = x[max]
                    probe = str(x[max])
                    percent = probe[0:4]

                    if (c.LEARNING_TYPE == "CATEGORY_BIN"):
                        if max == 1:
                            max = 2

                    #予想した時間
                    predict_t = datetime.fromtimestamp(s)
                    predict_day = str(predict_t.year) + "/" + str(predict_t.month) + "/" + str(predict_t.day)

                    win_flg = False
                    win_trade_flg = False

                    startVal = "NULL"
                    endVal = "NULL"
                    result = "NULL"
                    correct = "NULL"

                    tradeReult = []

                    if ((c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW" ) and max == 0)  \
                            or (c.LEARNING_TYPE == "CATEGORY" and (max == 0 or max == 2))\
                            or (c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and (max == 0 or max == 2))\
                            or (c.LEARNING_TYPE == "CATEGORY_BIN" ):

                        # 取引間隔を開ける場合、指定時間が経過していなければスキップする
                        if c.RESTRICT_FLG:
                            #取引制限をかける
                            if prev_trade_time != 0 and s - prev_trade_time < c.RESTRICT_SEC:
                                money_tmp[s] = money
                                if c.TRADE_FLG:
                                    money_trade_tmp[s] = money_trade
                                continue

                        prev_trade_time = s

                        if c.TRADE_FLG:
                            tradeReult = r.zrangebyscore(c.DB_TRADE_NAME, s, s)
                            if len(tradeReult) == 0:
                                # 取引履歴がない場合1秒後の履歴として残っているかもしれないので取得
                                tradeReult = r.zrangebyscore(c.DB_TRADE_NAME, s + 1, s + 1)

                            if len(tradeReult) != 0:
                                trade_cnt = trade_cnt + 1
                                tmps = json.loads(tradeReult[0].decode('utf-8'))
                                startVal = tmps.get("startVal")
                                endVal = tmps.get("endVal")
                                result = tmps.get("result")
                                if result == "win":
                                    win_trade_flg = True
                                    trade_win_cnt = trade_win_cnt + 1
                                    money_trade = money_trade + c.PAYOUT
                                    max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade, drawdown_trade, c.PAYOUT)
                                else:
                                    win_trade_flg = False
                                    money_trade = money_trade - c.PAYOFF
                                    max_drawdown_trade, drawdown_trade = countDrawdoan(max_drawdowns_trade, max_drawdown_trade, drawdown_trade, c.PAYOFF * -1)

                            if c.TRADE_ONLY_FLG:
                                if len(tradeReult) == 0:
                                    #取引履歴がなければスキップ
                                    money_tmp[s] = money
                                    if c.TRADE_FLG:
                                        money_trade_tmp[s] = money_trade
                                    continue

                        cnt += 1
                        win_flg = True if max == y.argmax() else False

                        if win_flg:
                            win_cnt += 1
                            atr_win_list.append(atr)
                            win_ind.append(i)

                        else:
                            atr_lose_list.append(atr)

                        atr_total_list.append(atr)
                        bet_ind.append(i)


                        if predict_day in trade_cnt_by_day.keys():
                            trade_cnt_by_day[predict_day] += 1
                        else:
                            trade_cnt_by_day[predict_day] = 1

                        # 理論上のスプレッド毎の勝率
                        flg = False
                        for k, v in c.SPREAD_LIST.items():
                            if sp > v[0] and sp <= v[1]:
                                spread_trade[k] = spread_trade.get(k, 0) + 1
                                if win_flg:
                                    spread_win[k] = spread_win.get(k, 0) + 1

                                if max == 0:
                                    spread_trade_up[k] = spread_trade_up.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_up[k] = spread_win_up.get(k, 0) + 1
                                elif max == 2:
                                    spread_trade_dw[k] = spread_trade_dw.get(k, 0) + 1
                                    if win_flg:
                                        spread_win_dw[k] = spread_win_dw.get(k, 0) + 1

                                flg = True
                                break

                        if flg == False:
                            if sp < 0:
                                spread_trade["spread0"] = spread_trade.get("spread0", 0) + 1
                                if win_flg:
                                    spread_win["spread0"] = spread_win.get("spread0", 0) + 1

                                if max == 0:
                                    spread_trade_up["spread0"] = spread_trade_up.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_up["spread0"] = spread_win_up.get("spread0", 0) + 1
                                elif max == 2:
                                    spread_trade_dw["spread0"] = spread_trade_dw.get("spread0", 0) + 1
                                    if win_flg:
                                        spread_win_dw["spread0"] = spread_win_dw.get("spread0", 0) + 1

                            else:
                                spread_trade["spread16Over"] = spread_trade.get("spread16Over", 0) + 1
                                if win_flg:
                                    spread_win["spread16Over"] = spread_win.get("spread16Over", 0) + 1

                                if max == 0:
                                    spread_trade_up["spread16Over"] = spread_trade_up.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_up["spread16Over"] = spread_win_up.get("spread16Over", 0) + 1
                                elif max == 2:
                                    spread_trade_dw["spread16Over"] = spread_trade_dw.get("spread16Over", 0) + 1
                                    if win_flg:
                                        spread_win_dw["spread16Over"] = spread_win_dw.get("spread16Over", 0) + 1

                        # 実際のスプレッド毎の勝率
                        if len(tradeReult) != 0:
                            flg = False
                            for k, v in c.SPREAD_LIST.items():
                                if sp > v[0] and sp <= v[1]:
                                    spread_trade_real[k] = spread_trade_real.get(k, 0) + 1
                                    if win_trade_flg:
                                        spread_win_real[k] = spread_win_real.get(k, 0) + 1

                                    if max == 0:
                                        spread_trade_real_up[k] = spread_trade_real_up.get(k, 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_up[k] = spread_win_real_up.get(k, 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw[k] = spread_trade_real_dw.get(k, 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_dw[k] = spread_win_real_dw.get(k, 0) + 1

                                    flg = True
                                    break
                            if flg == False:
                                if sp < 0:
                                    spread_trade_real["spread0"] = spread_trade_real.get("spread0", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real["spread0"] = spread_win_real.get("spread0", 0) + 1

                                    if max == 0:
                                        spread_trade_real_up["spread0"] = spread_trade_real_up.get("spread0", 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_up["spread0"] = spread_win_real_up.get("spread0", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw["spread0"] = spread_trade_real_dw.get("spread0", 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_dw["spread0"] = spread_win_real_dw.get("spread0", 0) + 1

                                else:
                                    spread_trade_real["spread16Over"] = spread_trade_real.get("spread16Over", 0) + 1
                                    if win_trade_flg:
                                        spread_win_real["spread16Over"] = spread_win_real.get("spread16Over", 0) + 1

                                    if max == 0:
                                        spread_trade_real_up["spread16Over"] = spread_trade_real_up.get("spread16Over", 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_up["spread16Over"] = spread_win_real_up.get("spread16Over", 0) + 1
                                    elif max == 2:
                                        spread_trade_real_dw["spread16Over"] = spread_trade_real_dw.get("spread16Over", 0) + 1
                                        if win_trade_flg:
                                            spread_win_real_dw["spread16Over"] = spread_win_real_dw.get("spread16Over", 0) + 1

                        # 理論上の秒毎の勝率
                        if per_sec_flg:
                            per_sec_dict[predict_t.second][0] += 1
                            if win_flg:
                                per_sec_dict[predict_t.second][1] += 1

                            # 実際の秒毎の勝率
                            if len(tradeReult) != 0:
                                per_sec_dict_real[predict_t.second][0] += 1
                                if win_trade_flg:
                                    per_sec_dict_real[predict_t.second][1] += 1

                        # 分ごと及び、分秒ごとのbetした回数と勝ち数を保持
                        per_min_dict[predict_t.minute][0] += 1
                        # per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][0] += 1
                        if win_flg:
                            per_min_dict[predict_t.minute][1] += 1
                            # per_minsec_dict[str(predict_t.minute) + "-" + str(predict_t.second)][1] += 1
                        if len(tradeReult) != 0:
                            per_min_dict_real[predict_t.minute][0] += 1
                            if win_trade_flg:
                                per_min_dict_real[predict_t.minute][1] += 1

                        # 時間ごとのbetした回数と勝ち数を保持
                        per_hour_dict[predict_t.hour][0] += 1
                        if win_flg:
                            per_hour_dict[predict_t.hour][1] += 1
                        if len(tradeReult) != 0:
                            per_hour_dict_real[predict_t.hour][0] += 1
                            if win_trade_flg:
                                per_hour_dict_real[predict_t.hour][1] += 1

                        # 理論上の直近変化率毎の勝率
                        flg = False
                        for k, v in c.DIVIDE_LIST.items():
                            if dp > v[0] and dp <= v[1]:
                                divide_trade[k] = divide_trade.get(k, 0) + 1
                                if win_flg:
                                    divide_win[k] = divide_win.get(k, 0) + 1
                                flg = True
                                break

                        # 実際の直近変化率毎の勝率
                        if len(tradeReult) != 0:
                            flg = False
                            for k, v in c.DIVIDE_LIST.items():
                                if dp > v[0] and dp <= v[1]:
                                    divide_trade_real[k] = divide_trade_real.get(k, 0) + 1
                                    if win_trade_flg:
                                        divide_win_real[k] = divide_win_real.get(k, 0) + 1
                                    flg = True
                                    break

                        """
                        # 理論上の正解までの変化率毎の勝率
                        flg = False
                        for k, v in DIVIDE_AFT_LIST.items():
                            if da > v[0] and da <= v[1]:
                                divide_aft_trade[k] = divide_aft_trade.get(k, 0) + 1
                                if win_flg:
                                    divide_aft_win[k] = divide_aft_win.get(k, 0) + 1
                                flg = True
                                break

                        if flg == False:
                            #変化率が9以上の場合
                            divide_aft_trade["divide13over"] = divide_aft_trade.get("divide13over", 0) + 1
                            if win_flg:
                                divide_aft_win["divide13over"] = divide_aft_win.get("divide13over", 0) + 1

                        # 実際の正解までの変化率毎の勝率
                        if len(tradeReult) != 0:
                            flg = False
                            for k, v in DIVIDE_AFT_LIST.items():
                                if da > v[0] and da <= v[1]:
                                    divide_aft_trade_real[k] = divide_aft_trade_real.get(k, 0) + 1
                                    if win_trade_flg:
                                        divide_aft_win_real[k] = divide_aft_win_real.get(k, 0) + 1
                                    flg = True
                                    break

                            if flg == False:
                                # 変化率が9以上の場合
                                divide_aft_trade_real["divide13over"] = divide_aft_trade_real.get("divide13over", 0) + 1
                                if win_trade_flg:
                                    divide_aft_win_real["divide13over"] = divide_aft_win_real.get("divide13over", 0) + 1

                        """

                        tmp_prob_cnt_list = {}
                        tmp_prob_real_cnt_list = {}
                        # 確率ごとのトレード数および勝率を求めるためのリスト
                        if percent in prob_list.keys():
                            tmp_prob_list = prob_list[percent]
                            if win_flg:
                                tmp_prob_cnt_list["win_cnt"] = tmp_prob_list["win_cnt"] + 1
                                tmp_prob_cnt_list["lose_cnt"] = tmp_prob_list["lose_cnt"]
                            else:
                                tmp_prob_cnt_list["win_cnt"] = tmp_prob_list["win_cnt"]
                                tmp_prob_cnt_list["lose_cnt"] = tmp_prob_list["lose_cnt"] + 1
                            prob_list[percent] = tmp_prob_cnt_list
                        else:
                            if win_flg:
                                tmp_prob_cnt_list["win_cnt"] = 1
                                tmp_prob_cnt_list["lose_cnt"] = 0
                            else:
                                tmp_prob_cnt_list["win_cnt"] = 0
                                tmp_prob_cnt_list["lose_cnt"] = 1
                            prob_list[percent] = tmp_prob_cnt_list
                        # トレードした場合
                        if len(tradeReult) != 0:
                            # 確率ごとのトレード数および勝率を求めるためのリスト
                            if percent in prob_real_list.keys():
                                tmp_prob_real_list = prob_real_list[percent]
                                if win_trade_flg:
                                    tmp_prob_real_cnt_list["win_cnt"] = tmp_prob_real_list["win_cnt"] + 1
                                    tmp_prob_real_cnt_list["lose_cnt"] = tmp_prob_real_list["lose_cnt"]
                                else:
                                    tmp_prob_real_cnt_list["win_cnt"] = tmp_prob_real_list["win_cnt"]
                                    tmp_prob_real_cnt_list["lose_cnt"] = tmp_prob_real_list["lose_cnt"] + 1
                                prob_real_list[percent] = tmp_prob_real_cnt_list
                            else:
                                if win_flg:
                                    tmp_prob_real_cnt_list["win_cnt"] = 1
                                    tmp_prob_real_cnt_list["lose_cnt"] = 0
                                else:
                                    tmp_prob_real_cnt_list["win_cnt"] = 0
                                    tmp_prob_real_cnt_list["lose_cnt"] = 1
                                prob_real_list[percent] = tmp_prob_real_cnt_list

                    if (c.LEARNING_TYPE == "CATEGORY" and max == 0) or (c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and max == 0) or (c.LEARNING_TYPE == "CATEGORY_BIN_UP" and max == 0) :
                        # Up predict

                        if c.FX:
                            profit = (cl - float( Decimal("0.001") * Decimal(c.SPREAD) )) * c.FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                        if max == y.argmax():
                            if c.FX == False:
                                money = money + c.PAYOUT
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOUT)

                            cnt_up_cor = cnt_up_cor + 1

                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    #理論上の結果と実トレードの結果がおなじ
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1
                                else:
                                    correct = "FALSE"
                                    trade_wrong_lose_cnt += 1
                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(ec) + "," + "UP" + ","
                                              + probe + "," + "win" + "," + str(startVal) + "," + str(endVal)
                                              + "," + result + "," + correct)

                        else :
                            if c.FX == False:
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

                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(
                                    ec) + "," + "UP" + ","
                                                  + probe + "," + "lose" + "," + str(startVal) + "," + str(endVal)
                                                  + "," + result + "," + correct)

                    elif (c.LEARNING_TYPE == "CATEGORY" and max == 2) or (c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" and max == 2)  or (c.LEARNING_TYPE == "CATEGORY_BIN_DW" and max == 0) :
                        #Down predict
                        if c.FX:
                            #cはaft-befなのでdown予想の場合の利益として-1を掛ける
                            profit = ((cl * -1)  - float( Decimal("0.001") * Decimal(c.SPREAD) )) * c.FX_POSITION
                            money = money + profit
                            max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, profit)

                        if max == y.argmax():
                            if c.FX == False:
                                money = money + c.PAYOUT
                                max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, c.PAYOUT)

                            cnt_down_cor = cnt_down_cor + 1


                            if len(tradeReult) != 0:
                                if win_trade_flg:
                                    #理論上の結果と実トレードの結果がおなじ
                                    correct = "TRUE"
                                    true_cnt = true_cnt + 1
                                else:
                                    correct = "FALSE"
                                    trade_wrong_lose_cnt += 1
                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(ec) + "," + "DOWN" + ","
                                              + probe + "," + "win" + "," + str(startVal) + "," + str(endVal)
                                              + "," + result + "," + correct)

                        else:
                            if c.FX == False:
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

                                result_txt_trade.append(predict_t.strftime('%Y-%m-%d %H:%M:%S') + "," + str(sc) + "," + str(
                                    ec) + "," + "DOWN" + ","
                                                  + probe + "," + "lose" + "," + str(startVal) + "," + str(endVal)
                                                  + "," + result + "," + correct)

                    money_tmp[s] = money
                    if c.TRADE_FLG:
                        money_trade_tmp[s] = money_trade

                prev_money = c.START_MONEY
                prev_trade_money = c.START_MONEY

                output("cnt:", cnt)

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




                for txt in result_txt_trade:
                    res = txt.find("FALSE")
                    if res != -1:
                        print(txt)

                # print('\n'.join(result_txt))

                if trade_cnt != 0:
                    output("trade cnt: " + str(trade_cnt))
                    output("trade correct: " + str(true_cnt / trade_cnt))
                    output("trade wrong cnt: " + str(trade_cnt - true_cnt))
                    output("trade wrong win cnt: " + str(trade_wrong_win_cnt))
                    output("trade wrong lose cnt: " + str(trade_wrong_lose_cnt))
                    output("trade accuracy: " + str(trade_win_cnt / trade_cnt))
                    output("trade money: " + str(prev_trade_money))
                    output("trade cnt rate: " + str(trade_cnt / total_num))

                output("predict money: " + str(prev_money))
                max_drawdowns.sort()


                result_txt = []
                if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                    Acc = 0 if cnt == 0 else win_cnt/cnt

                    result_txt.append("Accuracy border_ind " + str(border_ind) + ":" + str(Acc))
                    result_txt.append("Total:" + str(cnt) + " Correct:" + str(win_cnt))
                else:
                    result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                    result_txt.append("Total:" + str(cnt) + " Correct:" + str(win_cnt))

                result_txt.append("Earned Money / MaxDrawDown:" + str((prev_money - c.START_MONEY) / max_drawdowns[0]))
                for i in result_txt:
                    output(i)

                if len(c.IND_COLS) != 0:
                    show_win_ind(conf, ind_list, train_list_index, ind, bet_ind, win_ind, show_plot, save_dir)

                if c.ATR_COL != "":

                    target_atr_list_sorted = np.sort(np.array(atr_total_list))
                    atr_win_arr = np.array(atr_win_list)
                    #atr_lose_arr = np.array(atr_lose_list)

                    output("ATR毎の勝率")
                    tpips = float(Decimal(str(c.PIPS)) * Decimal("10"))
                    if ("satr" in conf.ATR_COL) == False:
                        tpips = 0.1

                    for i in range(10):
                        #小数点以下
                        if i !=9 :
                            w_cnt = len(np.where((atr_win_arr > (tpips * i)) & (atr_win_arr <= (tpips * (i+1))))[0])
                            #l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                            total_cnt = len(np.where((target_atr_list_sorted > (tpips * i)) & (target_atr_list_sorted <= (tpips * (i+1))))[0])
                            if total_cnt != 0:
                                profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
                                output(tpips * i, "~", tpips * (i+1), " rate:", w_cnt / total_cnt, " profit:", profit, " total_cnt:", total_cnt)

                    tpips = float(Decimal(str(c.PIPS)) * Decimal("100"))
                    if ("satr" in conf.ATR_COL) == False:
                        tpips = 1

                    for i in range(10):
                        if i !=9 :
                            w_cnt = len(np.where((atr_win_arr > (tpips * i)) & (atr_win_arr <= (tpips * (i+1))))[0])
                            #l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                            total_cnt = len(np.where((target_atr_list_sorted > (tpips * i)) & (target_atr_list_sorted <= (tpips * (i+1))))[0])
                            if total_cnt != 0:
                                profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
                                output((tpips * i), "~", (tpips * (i+1)), " rate:", w_cnt / total_cnt, " profit:", profit, " total_cnt:", total_cnt)
                        else:
                            w_cnt = len(np.where(atr_win_arr > (tpips * i))[0])
                            # l_cnt = len(np.where((atr_lose_arr >= under_val) & (atr_lose_arr < over_val))[0])
                            total_cnt = len(np.where(target_atr_list_sorted > (tpips* i))[0])
                            if total_cnt != 0:
                                profit = (c.PAYOUT * w_cnt) - ((total_cnt - w_cnt) * c.PAYOFF)
                                output((tpips * i), "~", " rate:", w_cnt / total_cnt, " profit:", profit, " total_cnt:", total_cnt)


                output("理論上のスプレッド毎の勝率(全体)")
                for k, v in sorted(c.SPREAD_LIST.items()):
                    if spread_trade.get(k, 0) != 0:
                        output(k, " cnt:", spread_trade.get(k, 0),
                              " profit:", spread_win.get(k, 0) * c.PAYOUT - (spread_trade.get(k) - spread_win.get(k, 0)) * c.PAYOFF,
                              " win rate:", spread_win.get(k, 0) / spread_trade.get(k))
                    else:
                        output(k, " cnt:", spread_trade.get(k, 0))

                output("理論上のスプレッド毎の勝率(UP)")
                for k, v in sorted(c.SPREAD_LIST.items()):
                    if spread_trade_up.get(k, 0) != 0:
                        output(k, " cnt:", spread_trade_up.get(k, 0),
                              " profit:", spread_win_up.get(k, 0) * c.PAYOUT - (spread_trade_up.get(k) - spread_win_up.get(k, 0)) * c.PAYOFF,
                              " win rate:", spread_win_up.get(k, 0) / spread_trade_up.get(k))
                    else:
                        output(k, " cnt:", spread_trade_up.get(k, 0))

                output("理論上のスプレッド毎の勝率(DW)")
                for k, v in sorted(c.SPREAD_LIST.items()):
                    if spread_trade_dw.get(k, 0) != 0:
                        output(k, " cnt:", spread_trade_dw.get(k, 0),
                              " profit:", spread_win_dw.get(k, 0) * c.PAYOUT - (spread_trade_dw.get(k) - spread_win_dw.get(k, 0)) * c.PAYOFF,
                              " win rate:", spread_win_dw.get(k, 0) / spread_trade_dw.get(k))
                    else:
                        output(k, " cnt:", spread_trade_dw.get(k, 0))

                if c.TRADE_FLG:
                    output("実トレード上のスプレッド毎の勝率(全体)")
                    for k, v in sorted(c.SPREAD_LIST.items()):
                        if spread_trade_real.get(k, 0) != 0:
                            output(k, " cnt:", spread_trade_real.get(k, 0),
                                  " profit:", spread_win_real.get(k, 0) * c.PAYOUT - (spread_trade_real.get(k) - spread_win_real.get(k, 0)) * c.PAYOFF,
                                  " win rate:", spread_win_real.get(k, 0) / spread_trade_real.get(k),)
                        else:
                            output(k, " cnt:", spread_trade_real.get(k, 0))

                    output("実トレード上のスプレッド毎の勝率(UP)")
                    for k, v in sorted(c.SPREAD_LIST.items()):
                        if spread_trade_real_up.get(k, 0) != 0:
                            output(k, " cnt:", spread_trade_real_up.get(k, 0),
                                  " profit:", spread_win_real_up.get(k, 0) * c.PAYOUT - (spread_trade_real_up.get(k) - spread_win_real_up.get(k, 0)) * c.PAYOFF,
                                  " win rate:", spread_win_real_up.get(k, 0) / spread_trade_real_up.get(k),)
                        else:
                            output(k, " cnt:", spread_trade_real_up.get(k, 0))

                    output("実トレード上のスプレッド毎の勝率(DW)")
                    for k, v in sorted(c.SPREAD_LIST.items()):
                        if spread_trade_real_dw.get(k, 0) != 0:
                            output(k, " cnt:", spread_trade_real_dw.get(k, 0),
                                  " profit:", spread_win_real_dw.get(k, 0) * c.PAYOUT - (spread_trade_real_dw.get(k) - spread_win_real_dw.get(k, 0)) * c.PAYOFF,
                                  " win rate:", spread_win_real_dw.get(k, 0) / spread_trade_real_dw.get(k),)
                        else:
                            output(k, " cnt:", spread_trade_real_dw.get(k, 0))

                    output("スプレッド毎の約定率")
                    for k, v in sorted(c.SPREAD_LIST.items()):
                        if spread_trade_real.get(k, 0) != 0:
                            output(k, " cnt:", spread_trade_real.get(k, 0), " rate:",
                                  spread_trade_real.get(k) / spread_trade.get(k))


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

                # 理論上の時間ごとの勝率
                per_hour_winrate_dict = {}
                for i in per_hour_dict.keys():
                    if per_hour_dict[i][0] != 0:
                        win_rate = per_hour_dict[i][1] / per_hour_dict[i][0]
                        per_hour_winrate_dict[i] = (win_rate,per_hour_dict[i][0])
                    else:
                        per_hour_winrate_dict[i] = (0,0)

                output("理論上の時間毎の勝率悪い順:")
                worst_sorted = sorted(per_hour_winrate_dict.items(), key=lambda x: x[1][0])
                for i in worst_sorted:
                    output(i[0], i[1][0], i[1][1])

                if c.TRADE_FLG:
                    # 実際の分ごとの勝率
                    per_hour_winrate_dict_real = {}
                    for i in per_hour_dict_real.keys():
                        if per_hour_dict_real[i][0] != 0:
                            win_rate = per_hour_dict_real[i][1] / per_hour_dict_real[i][0]
                            per_hour_winrate_dict_real[i] = (win_rate, per_hour_dict_real[i][0])
                        else:
                            per_hour_winrate_dict_real[i] = (0, 0)

                    output("実際の時間毎の勝率悪い順:")
                    worst_sorted = sorted(per_hour_winrate_dict_real.items(), key=lambda x: x[1][0])
                    for i in worst_sorted:
                        output(i[0], i[1][0], i[1][1])

                output("理論上の直近変化率毎の勝率")
                for k, v in sorted(c.DIVIDE_LIST.items()):
                    if divide_trade.get(k, 0) != 0:
                        output(k, " cnt:", divide_trade.get(k, 0), " win rate:", divide_win.get(k, 0) / divide_trade.get(k))
                    else:
                        output(k, " cnt:", divide_trade.get(k, 0))

                if c.TRADE_FLG:
                    output("実際の直近変化率毎の勝率")
                    for k, v in sorted(c.DIVIDE_LIST.items()):
                        if divide_trade_real.get(k, 0) != 0:
                            output(k, " cnt:", divide_trade_real.get(k, 0), " win rate:", divide_win_real.get(k, 0) / divide_trade_real.get(k))
                        else:
                            output(k, " cnt:", divide_trade_real.get(k, 0))

                """
                print("理論上の正解までの変化率毎の勝率")
                for k, v in sorted(DIVIDE_AFT_LIST.items()):
                    if divide_aft_trade.get(k, 0) != 0:
                        print(k, " cnt:", divide_aft_trade.get(k, 0), " win rate:", divide_aft_win.get(k, 0) / divide_aft_trade.get(k))
                    else:
                        print(k, " cnt:", divide_aft_trade.get(k, 0))

                if TRADE_FLG:
                    print("実際の正解までの変化率毎の勝率")
                    for k, v in sorted(DIVIDE_AFT_LIST.items()):
                        if divide_aft_trade_real.get(k, 0) != 0:
                            print(k, " cnt:", divide_aft_trade_real.get(k, 0), " win rate:", divide_aft_win_real.get(k, 0) / divide_aft_trade_real.get(k))
                        else:
                            print(k, " cnt:", divide_aft_trade_real.get(k, 0))
                """


                output("MAX DrawDowns(理論上のドローダウン)")
                output(max_drawdowns[0:10])

                drawdown_cnt = {}
                for i in max_drawdowns:
                    for k, v in c.DRAWDOWN_LIST.items():
                        if i < v[0] and i >= v[1]:
                            drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                            break
                for k, v in sorted(c.DRAWDOWN_LIST.items()):
                    output(k, drawdown_cnt.get(k,0))

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


                for k, v in sorted(prob_list.items()):
                    # 勝率
                    win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                    output("理論上の確率:" + k + " 勝ち:" + str(v["win_cnt"]) + " 負け:" + str(v["lose_cnt"]) + " 勝率:" + str(win_rate))


                if c.TRADE_FLG:
                    for k, v in sorted(prob_real_list.items()):
                        # トレードできた確率
                        trade_rate = (v["win_cnt"] + v["lose_cnt"]) / (prob_list[k]["win_cnt"] + prob_list[k]["lose_cnt"])
                        # 勝率
                        win_rate = v["win_cnt"] / (v["win_cnt"] + v["lose_cnt"])
                        output("実トレード上の確率:" + k + " トレードできた割合:" + str(trade_rate) + " 勝率:" + str(win_rate))

                output("日毎の取引回数が多い順:")
                trade_cnt_sorted = sorted(trade_cnt_by_day.items(), key=lambda x: x[1],reverse=True)
                for i, trd in enumerate(trade_cnt_sorted):
                    if i < 50:
                        output(trd[0], trd[1])

                """
                if per_sec_flg:
                    # 理論上の秒ごとの勝率
                    for i in per_sec_dict.keys():
                        if per_sec_dict[i][0] != 0:
                            win_rate = per_sec_dict[i][1] / per_sec_dict[i][0]
                            print("理論上の秒毎の確率:" + str(i) + " トレード数:" + str(per_sec_dict[i][0]) + " 勝率:" + str(win_rate))
    
                    if TRADE_FLG:
                        # 実際の秒ごとの勝率
                        for i in per_sec_dict_real.keys():
                            if per_sec_dict_real[i][0] != 0:
                                win_rate = per_sec_dict_real[i][1] / per_sec_dict_real[i][0]
                                print("実際の秒毎の確率:" + str(i) + " トレード数:" + str(per_sec_dict_real[i][0]) + " 勝率:" + str(win_rate))
                """

                if show_plot:
                    fig = plt.figure(figsize=(6.4*0.7, 4.8*0.7))
                    # 価格の遷移
                    ax1 = fig.add_subplot(111)

                    ax1.plot(close_list, 'g')

                    ax2 = ax1.twinx()
                    ax2.plot(money_y)
                    if c.TRADE_FLG:
                        ax2.plot(money_trade_y, "r")

                    if c.FX == True:
                        plt.title('border:' + str(border) + " position:" + str(c.FX_POSITION) + " spread:" + str(
                            c.SPREAD) + " money:" + str(money))
                    else:
                        if c.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            plt.title(
                                'border_ind:' + str(border_ind) + " payout:" + str(c.PAYOUT) + " spread:" + str(
                                    c.SPREAD) + " money:" + str(
                                    money))
                        else:
                            plt.title('border:' + str(border) + " payout:" + str(c.PAYOUT) + " spread:" + str(
                                c.SPREAD) + " money:" + str(money))

                    #plt.show()

                    filename = save_dir + "/" + 'testLstm2_total' + ".png"
                    fig.savefig(filename)

            if c.LEARNING_TYPE != "CATEGORY_BIN_BOTH":
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

        if c.LEARNING_TYPE != "CATEGORY_BIN_BOTH":
            if "suffix" in max_val_suffix.keys():
                output("max_val_suffix:", max_val_suffix["suffix"])

            output("total_bet_txt")
            for i in total_bet_txt:
                output(i)

            output("total_acc_txt")
            for i in total_acc_txt:
                output(i)

            if test_eval:
                output("total_accuacy_txt")
                for i in total_accuacy_txt:
                    output(i)

                output("total_loss_txt")
                for i in total_loss_txt:
                    output(i)

            output("total_money_txt")
            for i in total_money_txt:
                if i != "":
                    output(int(i))
                else:
                    output("")

if __name__ == "__main__":
    start_time = time.perf_counter()
    #print("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)
    #do_predict()
    conf = conf_class.ConfClass()
    conf.change_real_spread_flg(True)

    test_eval = False

    start_ends = [
        [datetime(2023, 4, 1), datetime(2024, 6, 8)],
    ]

    #spread_lists = [[]] #confからではなくことなったスプレッドでテストしたい場合
    spread_lists = [[2,3,4,5], ]


    restrict_secs = []

    for start_end in start_ends:
        start, end = start_end
        output(start, end)
        for list in spread_lists:
            dataSequence2 = make_data(conf,start, end, target_spreads=list, spread_correct=1, sub_force = True)
            if len(restrict_secs)!=0:
                for sec in restrict_secs:
                    conf.change_restrict_sec(sec)
                    do_predict(conf, dataSequence2, list, test_eval)
            else:
                do_predict(conf, dataSequence2, list, test_eval)
    print("Processing Time(Sec)", time.perf_counter() - start_time )
