import json
import numpy as np
import os

import psutil
import redis
import time
import gc
import math
from decimal import Decimal
from util import *
import talib
import csv
import pandas as pd
import socket
from datetime import  datetime
import send_mail as mail
"""
各間隔のcloseデータをもとにRSIやストキャスなどのインデックスを計算して既存DBに追加する
"""

host = "127.0.0.1"

pred_terms = [1800]  # 正解までの秒数

foot = []

# 同じ分足のDBを同時に作成
#hor_db_names = {'USDJPY_60_1440_0.01_HOR':5,} #DB名:上下幾つ分のデータを登録するか
hor_db_names = {}
hor_cols = [] # 登録用のカラム名

for hor_db_name,v in hor_db_names.items():
    start_v = -1 * v
    while True:
        if start_v > v:
            break

        # "USDJPY_60_360_0.01_HOR"
        hor_sec = hor_db_name.split("_")[1]
        hor_his = hor_db_name.split("_")[2]
        hor_width = hor_db_name.split("_")[3]
        hor_col_name = hor_sec + "-hor-" + hor_his + "-" + hor_width + "-" + str(start_v)
        hor_cols.append(hor_col_name)

        start_v += 1

#highlow_db_names = {'USDJPY_60_60_24_HIGHLOW':24,} #DB名:幾つ分のデータを登録するか
highlow_db_names = {}
highlow_cols = [] # 登録用のカラム名

for highlow_db_name,v in highlow_db_names.items():
    start_v = 1
    while True:
        if start_v > v:
            break

        # "USDJPY_60_60_24_HIGHLOW"
        hor_sec = highlow_db_name.split("_")[1]
        data_term = highlow_db_name.split("_")[2]
        data_length = int(get_decimal_multi(data_term, start_v))
        highlow_col_name = hor_sec + "-hl-" + str(data_term) + "-" + str(data_length)
        highlow_cols.append(highlow_col_name + "_h")
        highlow_cols.append(highlow_col_name + "_l")

        start_v += 1

# 0が現在レートが属するレンジ,-1がその一つ下
oanda_ords = []
oanda_poss = []
# oanda_ords = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
# oanda_poss = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]

ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 501, 10)]
#ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 101, 10)] + [i for i in range(160, 601, 60)] + [i for i in range(900, 7201, 300)]
#ds = [i for i in range(510, 1001, 10)]
subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 501, 10)]

#satrs = [5]
satrs = [i for i in range(5, 101, 5)]

smas = [i for i in range(5, 101, 5)]

atrs = [i for i in range(5, 101, 5)]

subds = [i + 1 for i in range(100)]

rsis = [5, 15, 30, 50, 75, 100]

sbbs = [5] + [i for i in range(10, 201, 10)]  # 5,10,20,30・・・200

adxs = [5]
adxs.extend([i for i in range(10, 201, 10)])  # 5,10,20,30・・・200

dips = [5] + [i for i in range(10, 101, 10)]

dims = [5] + [i for i in range(10, 101, 10)]

dxs = [5] + [i for i in range(10, 101, 10)]

d1s = [1]

dl01s = [i + 1 for i in range(1)]

dl1s = [i + 1 for i in range(1)]

dms = []

smamss = []
wmamss = []

smamrs = []
smams = [1]
wmams = []

ssmams = []

bbs = [5, 25]
bbs_col = ["bbu3", ]

bbcs = []
bbcs_col = ["bbcu3", "bbcl3"]

bbms = []
bbms_col = ["bbmu3", "bbml3"]

sbbms = []
sbbms_col = ["sbbmu3", "sbbml3"]

sbbs_col = ["sbbu3", "sbbu2", "sbbu1", "sbbm", "sbbl1", "sbbl2", "sbbl3"]


def make_bb(close_data):
    return_list = []
    for bb in bbs:
        upper1, middle, lower1 = talib.BBANDS(close_data, timeperiod=bb, nbdevup=1, nbdevdn=1, matype=0)
        upper2, middle, lower2 = talib.BBANDS(close_data, timeperiod=bb, nbdevup=2, nbdevdn=2, matype=0)
        upper3, middle, lower3 = talib.BBANDS(close_data, timeperiod=bb, nbdevup=3, nbdevdn=3, matype=0)
        tmp_list = []
        for tmp_col in bbs_col:

            if tmp_col == "bbu3":
                target = get_divide_arr(upper3, middle, math_log=False)
            elif tmp_col == "bbu2":
                target = get_divide_arr(upper2, middle, math_log=False)
            elif tmp_col == "bbu1":
                target = get_divide_arr(upper1, middle, math_log=False)
            elif tmp_col == "bbl1":
                target = get_divide_arr(lower1, middle, math_log=False)
            elif tmp_col == "bbl2":
                target = get_divide_arr(lower2, middle, math_log=False)
            elif tmp_col == "bbl3":
                target = get_divide_arr(lower3, middle, math_log=False)

            tmp_list.append(target.reshape(len(target), -1))

        return_list.append(np.concatenate(tmp_list, 1))

    return_list = np.concatenate(return_list, 1)
    return_list = return_list.reshape(9, len(bbs), len(bbs_col))

    return return_list


def make_atr(high_data, low_data, close_data, ):
    return_list = []
    for atr in atrs:
        atr_data = talib.ATR(high_data, low_data, close_data, timeperiod=atr)
        # atr_data_bef= np.roll(atr_data, r)
        # target = get_divide_arr(atr_data_bef, atr_data, math_log=False)
        # target[:r] = None
        target = get_divide_arr(close_data, (atr_data + close_data))
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_satr(high_data, low_data, close_data, ):
    return_list = []
    for satr in satrs:
        atr_data = talib.ATR(high_data, low_data, close_data, timeperiod=satr)
        # atr_data_bef= np.roll(atr_data, r)
        # target = get_divide_arr(atr_data_bef, atr_data, math_log=False)
        # target[:r] = None
        target = get_sub_arr(close_data, (atr_data + close_data))
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_sma(close_data):
    return_list = []
    for sma in smas:
        sma_data = talib.SMA(close_data, timeperiod=sma)

        s_bef = np.roll(sma_data, 1)
        target = get_divide_arr(s_bef, sma_data)
        target[:1] = None
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_smac(close_data):
    return_list = []
    for sma in smas:
        sma_data = talib.SMA(close_data, timeperiod=sma)

        target = get_divide_arr(sma_data, close_data)
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_smam_raw(base_close_data, base_score_data, score_data, term, bet_term):
    return_list = []
    for smamr in smamrs:
        period = int(get_decimal_divide(get_decimal_multi(term, smamr), bet_term))
        sma_data = talib.SMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        return_list.append(base_sma)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_smam(base_close_data, base_score_data, score_data, term, bet_term):
    return_list = []
    for smam in smams:
        period = int(get_decimal_divide(get_decimal_multi(term, smam), bet_term))
        sma_data = talib.SMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        s_bef = np.roll(base_sma, smam)
        target = get_divide_arr(s_bef, base_sma)
        target[:smam] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_smam_sec(base_close_data, base_score_data, score_data, term, bet_term):
    return_list = []

    for smams in smamss:
        period = int(get_decimal_divide(smams, bet_term))
        sma_data = talib.SMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        s_bef = np.roll(base_sma, 1)
        target = get_divide_arr(s_bef, base_sma)
        target[:1] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_wmam_sec(base_close_data, base_score_data, score_data, term, bet_term):
    return_list = []

    for wmams in wmamss:
        period = int(get_decimal_divide(wmams, bet_term))
        sma_data = talib.WMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        s_bef = np.roll(base_sma, 1)
        target = get_divide_arr(s_bef, base_sma)
        target[:1] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_smam_by_sec(base_close_data, base_score_data, score_data, term, bet_term, secs):
    return_list = []

    for sec in secs:
        period = int(get_decimal_divide(sec, bet_term))
        sma_data = talib.SMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        return_list.append(base_sma)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_smam_a_by_sec(base_close_data, base_score_data, score_data, term, bet_term, secs):
    return_list = []

    for sec in secs:
        period = int(get_decimal_divide(sec, bet_term))
        roll_period = int(get_decimal_divide(get_decimal_divide(sec, bet_term), "2"))
        if roll_period % 2 != 0:
            # 奇数の場合はエラー　適切な位置にrollできないため
            print("sec should be even, not odd")
            exit(1)

        sma_data = talib.SMA(base_close_data, timeperiod=period)
        sma_data = np.roll(sma_data, -1 * roll_period)  # 目的のscoreを中心として平均を取るのでシフトさせる
        sma_data[-1 * roll_period:] = None
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        return_list.append(base_sma)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_wmam(base_close_data, base_score_data, score_data, term, bet_term):
    return_list = []
    for wmam in wmams:
        period = int(get_decimal_divide(get_decimal_multi(term, wmam), bet_term))
        sma_data = talib.WMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        s_bef = np.roll(base_sma, 1)
        target = get_divide_arr(s_bef, base_sma)
        target[:1] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_wmam_by_sec(base_close_data, base_score_data, score_data, term, bet_term, secs):
    return_list = []
    for sec in secs:
        period = int(get_decimal_divide(sec, bet_term))
        sma_data = talib.WMA(base_close_data, timeperiod=period)
        base_dict = dict(zip(base_score_data, sma_data))

        base_sma = []
        for sc in score_data:
            try:
                base_sma.append(base_dict.get(sc + term - bet_term))
            except Exception as ex:
                print(tracebackPrint(ex))
                exit(1)

        return_list.append(base_sma)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


"""
def make_smam(mean_data):
    return_list = []
    for smam in smams:

        sma_data = talib.SMA(mean_data, timeperiod=smam)

        s_bef = np.roll(sma_data, 1)
        target = get_divide_arr(s_bef, sma_data)
        target[:1] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list
"""


def make_ssmam(mean_data):
    return_list = []
    for ssmam in ssmams:
        ssma_data = talib.SMA(mean_data, timeperiod=ssmam)

        s_bef = np.roll(ssma_data, 1)
        target = get_sub_arr(s_bef, ssma_data)
        target[:1] = None
        return_list.append(target)

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_d(close_data, ):
    return_list = []
    for d in ds:
        d_bef = np.roll(close_data, d)
        target = get_divide_arr(d_bef, close_data)
        target[:d] = None
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_d1(close_data):
    return_list = []
    d_bef = np.roll(close_data, 1)
    target = get_divide_arr(d_bef, close_data, math_log=False)
    target[0] = None
    for d1 in d1s:
        target_tmp = np.roll(target, d1 - 1)
        return_list.append(target.reshape(len(target_tmp), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_sub(close_data, multi=1):
    return_list = []
    for sub in subs:
        sub_bef = np.roll(close_data, sub)
        target = get_sub_arr(sub_bef, close_data, multi=multi)
        target[:sub] = None
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_subd(close_data):
    return_list = []
    d_bef = np.roll(close_data, 1)
    target = get_sub_arr(d_bef, close_data)
    target[0] = None
    for d1 in subds:
        return_list.append(np.roll(target, d1 - 1))

    return_list = np.array(np.array(return_list).T.tolist())
    return return_list


def make_dm(mean_data):
    return_list = []
    for dm in dms:
        d_bef = np.roll(mean_data, dm)
        target = get_divide_arr(d_bef, mean_data, math_log=False)
        target[:dm] = None
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_rsi(close_data):
    return_list = []
    for rsi in rsis:
        target = talib.RSI(close_data, timeperiod=rsi)
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_dip(high_data, low_data, close_data):
    return_list = []
    for dip in dips:
        target = talib.PLUS_DI(high_data, low_data, close_data, timeperiod=dip)
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_dim(high_data, low_data, close_data):
    return_list = []
    for dim in dims:
        target = talib.MINUS_DI(high_data, low_data, close_data, timeperiod=dim)
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_dx(high_data, low_data, close_data):
    return_list = []
    for dx in dxs:
        p = talib.PLUS_DI(high_data, low_data, close_data, timeperiod=dx)
        m = talib.MINUS_DI(high_data, low_data, close_data, timeperiod=dx)
        target = p - m
        return_list.append(target.reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_answer(close_data, term):
    return_list = []
    return_list_abs = []
    for pred_term in pred_terms:
        pred_length = int(Decimal(str(pred_term)) / Decimal(str(term)))
        pred_length_minus = int(Decimal("-1") * Decimal(str(pred_length)))

        close_data_after = np.roll(close_data, pred_length_minus)
        answer_data = get_divide_arr(close_data, close_data_after, math_log=False)
        answer_abs_data = np.abs(answer_data)  # 絶対値を取る

        # ずらした分、最後の方は正解がないはずなのでNone(nan となるのでnp.isnan()で判定する)をいれる
        answer_data[pred_length_minus:] = None
        answer_abs_data[pred_length_minus:] = None

        return_list.append(answer_data)
        return_list_abs.append(answer_abs_data)
    """
    if len(pred_terms) == 0:
        # もしアンサーを作らない場合は0だけのダミーの配列をつくる　後にzipでループさせるため
        return_list.append(np.zeros(len(close_data)))
        return_list_abs.append(np.zeros(len(close_data)))
    """

    return_list = np.array(return_list).T.tolist()  # 行列を入れ替える
    return_list_abs = np.array(return_list_abs).T.tolist()  # 行列を入れ替える

    return return_list, return_list_abs


def make_sanswer(close_data, term):
    return_list = []
    return_list_abs = []
    for pred_term in pred_terms:
        pred_length = int(Decimal(str(pred_term)) / Decimal(str(term)))
        pred_length_minus = int(Decimal("-1") * Decimal(str(pred_length)))

        close_data_after = np.roll(close_data, pred_length_minus)
        answer_data = get_sub_arr(close_data, close_data_after, )
        answer_abs_data = np.abs(answer_data)  # 絶対値を取る

        # ずらした分、最後の方は正解がないはずなのでNone(nan となるのでnp.isnan()で判定する)をいれる
        answer_data[pred_length_minus:] = None
        answer_abs_data[pred_length_minus:] = None

        return_list.append(answer_data)
        return_list_abs.append(answer_abs_data)
    """
    if len(pred_terms) == 0:
        # もしアンサーを作らない場合は0だけのダミーの配列をつくる　後にzipでループさせるため
        return_list.append(np.zeros(len(close_data)))
        return_list_abs.append(np.zeros(len(close_data)))
    """

    return_list = np.array(return_list).T.tolist()  # 行列を入れ替える
    return_list_abs = np.array(return_list_abs).T.tolist()  # 行列を入れ替える

    return return_list, return_list_abs


def make_foot(score_data, close_data, foot_dict, term):
    return_list = []

    for foot_name in foot:
        foot_data = foot_dict[foot_name]
        foot_sec = foot_name.split("-")[0]
        target = []
        for s, c in zip(score_data, close_data):
            # 直近の足のindを取得
            now_score_tmp = s + term
            tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(foot_sec)))
            tmp_val = foot_data.get(tmp_score)
            if "sma" in foot_name:
                if tmp_val != None and np.isnan(tmp_val) == False:
                    tmp_val = get_divide(tmp_val, c)

            target.append(tmp_val)

        return_list.append(np.array(target).reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list


def make_hor(score_data, close_data, hor_dict, term):
    return_list = []

    for hor_db_name,v in hor_db_names.items():
        # "USDJPY_60_360_0.01_HOR"
        hor_sec = int(hor_db_name.split("_")[1])
        hor_his = int(hor_db_name.split("_")[2])
        hor_width = float(hor_db_name.split("_")[3])

        hor_d = hor_dict[hor_db_name]

        no_data_tmp = [] #データなしの場合のデータをつくっておく
        tmp_start_v = -1 * v
        while True:
            if tmp_start_v > v:
                break
            no_data_tmp.append(None)
            tmp_start_v += 1

        target = []
        for s, c in zip(score_data, close_data):
            # 直近の足のindを取得
            now_score_tmp = s + term
            tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(str(hor_sec))))
            hor_data = hor_d.get(tmp_score)

            if hor_data != None:
                tmp_dict = {}
                for tmp_data in hor_data.split(","):
                    tmp_rate, hit_cnt = tmp_data.split(":")

                    hit_cnt = int(hit_cnt)
                    tmp_dict[tmp_rate] = hit_cnt
                hor_val = []
                start_v = -1 * v
                base = get_decimal_sub(c, Decimal(str(c)) % Decimal(str(hor_width)))
                while True:
                    if start_v > v:
                        break
                    t = str(get_decimal_add(base, get_decimal_multi(start_v, hor_width)))
                    tmp_hit_cnt = tmp_dict.get(t)
                    if tmp_hit_cnt == None:
                        tmp_hit_cnt = 0
                    else:
                        tmp_hit_cnt = tmp_hit_cnt - 1  # ヒットした数が2以上のものしかDBにない、ヒットしなかったら0なので合わせるためにヒットする数から1マイナスする

                    hor_val.append(tmp_hit_cnt)
                    start_v += 1
                target.append(hor_val)
            else:
                target.append(no_data_tmp)

        return_list.append(np.array(target).reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list

def make_highlow(score_data, close_data, highlow_dict, term):
    return_list = []

    for highlow_db_name,v in highlow_db_names.items():
        hor_sec = highlow_db_name.split("_")[1]
        data_term = int(highlow_db_name.split("_")[2])

        highlow_d = highlow_dict[highlow_db_name]

        no_data_tmp = [] #データなしの場合のデータをつくっておく
        tmp_start_v = 1
        while True:
            if tmp_start_v > v:
                break
            no_data_tmp.append(None)#highの分
            no_data_tmp.append(None)#lowの分
            tmp_start_v += 1

        target = []
        for s, c in zip(score_data, close_data):
            # 直近の足のindを取得
            now_score_tmp = s + term
            tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(str(hor_sec))))
            highlow_data = highlow_d.get(tmp_score)
            #print(highlow_data)
            if highlow_data != None:
                data_tmp = []
                start_v = 1
                while True:
                    if start_v > v:
                        break

                    data_length = str(data_term * start_v)
                    high = highlow_data.get(data_length + "_h")
                    low = highlow_data.get(data_length + "_l")

                    if high != None:
                        data_tmp.append(get_decimal_sub(high, c))
                    else:
                        data_tmp.append(None)

                    if low != None:
                        data_tmp.append(get_decimal_sub(low, c))
                    else:
                        data_tmp.append(None)

                    start_v += 1

                target.append(data_tmp)

            else:
                target.append(no_data_tmp)

        return_list.append(np.array(target).reshape(len(target), -1))

    return_list = np.concatenate(return_list, 1)
    return return_list

def make_oanda_ord(score_data, close_data, oanda_ord_dict, term):
    return_list = []

    for s, c in zip(score_data, close_data):
        # 直近の足のindを取得
        now_score_tmp = s + term
        tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal("300")))
        t_val = oanda_ord_dict.get(tmp_score)

        # データない場合
        if t_val == None or t_val == "":
            t_list = []
            for ord in oanda_ords:
                t_list.append(None)
            return_list.append(t_list)
            continue
        else:
            wid = float(t_val[0])
            ord_data = t_val[1]

            tmp_val_list = []
            mid_ind = None

            for k, tmp_data in enumerate(ord_data.split(",")):
                tmp_rate, tmp_val = tmp_data.split(":")
                tmp_rate = float(tmp_rate)
                tmp_val = float(tmp_val)
                tmp_val_list.append(tmp_val)

                if mid_ind == None and tmp_rate <= c and c < tmp_rate + wid:
                    # 現在レートが属するレンジが何番目か特定
                    mid_ind = k

            if mid_ind == None:
                # 該当レンジがないのでスキップ
                t_list = []
                for ord in oanda_ords:
                    t_list.append(None)
                return_list.append(t_list)
                continue
            else:
                t_list = []
                for ord in oanda_ords:
                    try:
                        t_list.append(tmp_val_list[mid_ind + ord])
                    except Exception as e:
                        # 該当レンジがない
                        t_list.append(None)
                return_list.append(t_list)
                continue

    return return_list


def make_oanda_pos(score_data, close_data, oanda_pos_dict, term):
    return_list = []

    for s, c in zip(score_data, close_data):
        # 直近の足のindを取得
        now_score_tmp = s + term
        tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal("300")))
        t_val = oanda_pos_dict.get(tmp_score)

        # データない場合
        if t_val == None or t_val == "":
            t_list = []
            for ord in oanda_poss:
                t_list.append(None)
            return_list.append(t_list)
            continue
        else:
            wid = float(t_val[0])
            ord_data = t_val[1]

            tmp_val_list = []
            mid_ind = None

            for k, tmp_data in enumerate(ord_data.split(",")):
                tmp_rate, tmp_val = tmp_data.split(":")
                tmp_rate = float(tmp_rate)
                tmp_val = float(tmp_val)
                tmp_val_list.append(tmp_val)

                if mid_ind == None and tmp_rate <= c and c < tmp_rate + wid:
                    # 現在レートが属するレンジが何番目か特定
                    mid_ind = k

            if mid_ind == None:
                # 該当レンジがないのでスキップ
                t_list = []
                for ord in oanda_poss:
                    t_list.append(None)
                return_list.append(t_list)
                continue
            else:
                t_list = []
                for ord in oanda_poss:
                    try:
                        t_list.append(tmp_val_list[mid_ind + ord])
                    except Exception as e:
                        # 該当レンジがない
                        t_list.append(None)
                return_list.append(t_list)
                continue

    return return_list


def make_dict_data(csv_dict, news):
    # timeは登録しない
    if "time" in news.keys():
        del news["time"]

    for tmp_col in news.keys():
        if tmp_col in csv_dict:
            csv_dict[tmp_col].append(news[tmp_col])
        else:
            csv_dict[tmp_col] = [news[tmp_col]]


def make_ind_db(start, end, org_db_no, new_db_no, bet_term, term, symbol, org_db, new_db, mode, use_old_data):
    # symbol = "EURUSD"

    db_source = "DUKA"  # DBの作成元データソース

    # bet_term = 2

    secs = []  # smaの変化率などを正解とする場合、smaの間隔秒数を指定

    # インデックスの追加方法
    # normal:既存DBに追記 new:新規にインデックスだけのDBを作成する csv:csvファイルに書き込む
    # new-csv:インデックスだけのDBに書き込みつつ、csvファイルにも書き込む
    # mode = "csv"

    # use_old_data = False  # mode=csv or new-csv の場合にDBの既存データを使用してCSV出力する場合

    new_db_name = symbol + "_" + str(bet_term) + "_IND"

    # term_convert.pyで作成したtick用DBを取り込む場合
    merge_tick_flg = False
    merge_tick_db = symbol + "_" + str(bet_term) + "_0" + "_TICK"

    merge_pred_flg = False
    merge_pred_db = symbol + "_" + str(bet_term) + "_PREDICT"

    # 元のデータから取り込む値(closeやsprなど)
    copy_cols = []

    # csvの場合終値を取りこむ
    if mode == "csv" or mode == "new-csv":
        if ("c" in copy_cols) == False:
            print("csv must include close!!!")
            copy_cols.append("c")
            # exit(1)

    # 元のデータから削除する値
    delete_cols = []
    redis_db_org = redis.Redis(host=org_db, port=6379, db=org_db_no, decode_responses=True)
    redis_db_new = redis.Redis(host=new_db, port=6379, db=new_db_no, decode_responses=True)

    start_stp = int(time.mktime(start.timetuple()))
    end_stp = int(time.mktime(end.timetuple())) - 1

    print(datetime.now())
    print("START", start, "END", end)
    start_time = time.perf_counter()


    db_list = make_db_list(symbol, term, bet_term)
    """
    db_list = []
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))
    else:
        db_list.append(symbol + "_" + str(term) + "_0")
    """

    # CSVに書き込む為に保持するデータ
    csv_dict = {}

    # smaなどの移動平均系を求めるためにbet_termのデータ配列を作成しておく
    base_close_data, base_score_data = [], []
    base_data = redis_db_org.zrangebyscore(symbol + "_5_0", start_stp, end_stp + 60 * 60 * 24,
                                           withscores=True)  # end_stpは多めにとっておく

    for line in base_data:
        body = line[0]
        score = float(line[1])
        tmps = json.loads(body)

        base_close_data.append(tmps.get("c"))
        base_score_data.append(score)

    base_close_data = np.array(base_close_data)
    base_score_data = np.array(base_score_data)

    if len(foot) != 0:
        foot_dict = {}
        for foot_name in foot:
            foot_dict[foot_name] = {}

        result_tmp = redis_db_org.zrangebyscore(symbol + "_2_IND_FOOT", start_stp, end_stp + 60 * 60 * 24,
                                                withscores=True)  # end_stpは多めにとっておく

        for res in result_tmp:
            body = res[0]
            score = float(res[1])
            tmps = json.loads(body)

            for foot_name in foot:
                foot_dict[foot_name][score] = tmps.get(foot_name)

    if len(hor_db_names) != 0:
        hor_dict = {}
        for hor_db_name,v in hor_db_names.items():
            hor_dict[hor_db_name] = {}

            result_tmp = redis_db_org.zrangebyscore(hor_db_name, start_stp, end_stp + 60 * 60 * 24,
                                                    withscores=True)  # end_stpは多めにとっておく

            for res in result_tmp:
                body = res[0]
                score = float(res[1])
                tmps = json.loads(body)

                hor_dict[hor_db_name][score] = tmps.get("data")

    if len(highlow_db_names) != 0:
        highlow_dict = {}
        for highlow_db_name,v in highlow_db_names.items():
            highlow_dict[highlow_db_name] = {}

            result_tmp = redis_db_org.zrangebyscore(highlow_db_name, start_stp, end_stp + 60 * 60 * 24,
                                                    withscores=True)  # end_stpは多めにとっておく

            for res in result_tmp:
                body = res[0]
                score = float(res[1])
                tmps = json.loads(body)

                start_v = 1
                while True:
                    if start_v > v:
                        break

                    data_term = highlow_db_name.split("_")[2]
                    data_length = str(int(get_decimal_multi(data_term, start_v)))
                    if start_v == 1:
                        highlow_dict[highlow_db_name][score] = {
                            data_length + "_h": tmps.get(data_length + "_h"),
                            data_length + "_l": tmps.get(data_length + "_l"),
                        }
                    else:
                        highlow_dict[highlow_db_name][score][data_length + "_h"] = tmps.get(data_length + "_h")
                        highlow_dict[highlow_db_name][score][data_length + "_l"] = tmps.get(data_length + "_l")
                    start_v += 1

    if len(oanda_ords) != 0:
        oanda_ord_dict = {}

        result_tmp = redis_db_org.zrangebyscore(symbol + "_OANDA_ORD", start_stp, end_stp + 60 * 60 * 24,
                                                withscores=True)  # end_stpは多めにとっておく

        for res in result_tmp:
            body = res[0]
            score = float(res[1])
            tmps = json.loads(body)

            oanda_ord_dict[score] = [tmps.get("wid"), tmps.get("data")]

    if len(oanda_poss) != 0:
        oanda_pos_dict = {}

        result_tmp = redis_db_org.zrangebyscore(symbol + "_OANDA_POS", start_stp, end_stp + 60 * 60 * 24,
                                                withscores=True)  # end_stpは多めにとっておく

        for res in result_tmp:
            body = res[0]
            score = float(res[1])
            tmps = json.loads(body)

            oanda_pos_dict[score] = [tmps.get("wid"), tmps.get("data")]

    for db in db_list:
        print(db)

        close_data, high_data, low_data = [], [], []
        mean_data = []
        wmean_data = []
        score_data = []
        result_data = redis_db_org.zrangebyscore(db, start_stp, end_stp, withscores=True)
        print("result_data length:" + str(len(result_data)))

        for line in result_data:
            body = line[0]
            score = float(line[1])
            tmps = json.loads(body)

            close_data.append(tmps.get("c"))
            if term == 2:
                high_data.append(tmps.get("c"))
                low_data.append(tmps.get("c"))
            else:
                high_data.append(tmps.get("eh"))
                low_data.append(tmps.get("el"))
            mean_data.append(tmps.get("m"))
            wmean_data.append(tmps.get("wm"))
            score_data.append(score)

        close_data = np.array(close_data)
        high_data = np.array(high_data)
        low_data = np.array(low_data)
        mean_data = np.array(mean_data)
        wmean_data = np.array(wmean_data)
        score_data = np.array(score_data)

        print("close_data length:", len(close_data))

        # cmo_data = talib.CMO(close_data, timeperiod=14) #-100 ～ 100 の間で推移し、50を上回ると買われすぎ、-50を下回ると売られすぎ
        """
        ppo_data = talib.PPO(close_data) #(12-period EMA−26-period EMA)/26-period EMA * 100   -100%から100%の値をとる +なら上昇傾向、-なら下降傾向にある
        slowk_data, slowd_data = talib.STOCH(high_data, low_data, close_data, fastk_period=5, slowk_period=3,slowd_period=3) #Slow％Dが0～20％にある時は、売られすぎ 80～100％にある時は、買われすぎ
        willr_data = talib.WILLR(high_data, low_data, close_data, timeperiod=14) #買われすぎが0%～-20%、売られすぎが-80%～-100%

        dm_data = make_dm(mean_data)
        """
        #d_data = make_d(close_data)
        eh_data = make_d1(high_data)
        el_data = make_d1(low_data)
        #wm_data = make_d1(wmean_data)
        #m_data = make_d1(mean_data)

        #sub_data = make_sub(close_data)

        #satr_data = make_satr(high_data, low_data, close_data)
        #smac_data = make_smac(close_data)
        #atr_data = make_atr(high_data, low_data, close_data)

        # f_data = make_foot(score_data, close_data, foot_dict, term)

        #highlow_data = make_highlow(score_data, close_data, highlow_dict, term)
        #hor_data = make_hor(score_data, close_data, hor_dict, term)

        # oanda_ord_data = make_oanda_ord(score_data, close_data, oanda_ord_dict, term)
        # oanda_pos_data = make_oanda_ord(score_data, close_data, oanda_pos_dict, term)

        # rsi_data = make_rsi(close_data)

        # dx_data = make_dx(high_data, low_data, close_data)
        # dip_data = make_dip(high_data, low_data, close_data)
        # dim_data = make_dim(high_data, low_data, close_data)

        # bb_data = make_bb(close_data)
        # sanswer_data, sanswer_abs_data = make_sanswer(close_data)

        # adx_data = make_adx(high_data, low_data, close_data)

        # subd_data = make_subd(close_data)
        # hsub_data = make_sub(high_data, hsubs)
        # lsub_data = make_sub(low_data, lsubs)
        # sbbm_data = make_sbbm(mean_data, close_data)


        # d1_data = make_d1(close_data)
        # md_data = make_d(ds, mean_data, 10000)

        # smam_data = make_smam(base_close_data, base_score_data, score_data)
        # smamr_data = make_smam_raw(base_close_data, base_score_data, score_data)
        # wmam_data = make_wmam(base_close_data, base_score_data, score_data)
        # smam_answer = make_smam_by_sec(base_close_data, base_score_data, score_data)
        # wmam_answer = make_wmam_by_sec(base_close_data, base_score_data, score_data)

        # smams_data = make_smam_sec(base_close_data, base_score_data, score_data, )
        # wmams_data = make_wmam_sec(base_close_data, base_score_data, score_data, )

        # bbm_data = make_bbm(mean_data)

        # bbc_data = make_bbc(close_data)

        # ssmam_data = make_ssmam(mean_data)

        # dl1_data = make_dl1(close_data)
        # dl01_data = make_dl01(close_data)

        # answer_data, answer_abs_data = make_answer(close_data)

        # print(rsi_data_14[15:])
        # print(answer_abs_data[15:])

        start5 = time.perf_counter()

        # DBに登録する
        cnt = 0
        for line, eh, el in zip(result_data,  eh_data, el_data ):
            # for line in result_data:
            cnt += 1
            body = line[0]
            score = float(line[1])
            orgs = json.loads(body)

            suffix = str(term) + "-" if (mode == "new" or mode == "csv" or mode == "new-csv") else ""

            tmps = {}
            # tmps[suffix + "cmo"] = cmo
            """
            tmps[suffix + "ppo"] = ppo
            tmps[suffix + "slowd"] = slowd
            tmps[suffix + "willr"] = willr


            for i, tmp in enumerate(dms):
                tmps[suffix + "dm-" + str(tmp)] = dm[i]

            for i, tmp in enumerate(pred_terms):
                tmps[str(tmp) + "-a"] = ad[i]
                tmps[str(tmp)  + "-aa"] = aad[i]

            for b1, col in zip(sbb, sbbs_col):
                for bs, b2 in zip(sbbs, b1):
                    tmps[suffix + col + "-" + str(bs)] = b2

            for i, tmp in enumerate(atrs):
                tmps[suffix + "satr-" + str(tmp)] = atr[i]

            for i, tmp in enumerate(adxs):
                tmps[suffix + "adx-" + str(tmp)] = adx[i]

            for i, tmp in enumerate(pred_terms):
                tmps[str(tmp) + "-a"] = sanswer[i]
                tmps[str(tmp)  + "-aa"] = sanswer_abs[i]

            for i, tmp in enumerate(subds):
                tmps[suffix + "subd-" + str(tmp)] = subd[i]

            for bb in return_list:
                for bb_1,bbs_1 in zip(bb, bbs):
                    for bb_2, bbs_col_1 in zip(bb_1, bbs_col):
                        #tmps[suffix + bbs_col_1 + "-" + str(bbs_1)] = bb_2

            for b1, col in zip(sbbm, sbbms_col):
                for bs, b2 in zip(sbbms, b1):
                    tmps[suffix + col + "-" + str(bs)] = b2

            for i, tmp in enumerate(ssmams):
                tmps[suffix + "ssmam-" + str(tmp)] = ssmam[i]

            for i, tmp in enumerate(ds):
                tmps[suffix + "md-" + str(tmp)] = md[i]



            for i, tmp in enumerate(ds):
                tmps[suffix + "md-" + str(tmp)] = md[i]

            for b1, col in zip(bbm, bbms_col):
                for bs, b2 in zip(bbms, b1):
                    tmps[suffix + col + "-" + str(bs)] = b2 

            for i, tmp in enumerate(atrs):
                tmps[suffix + "atr-" + str(tmp)] = atr[i]

            for i, tmp in enumerate(ds):
                tmps[suffix + "md1-" + str(tmp)] = md[i]

            for i, tmp in enumerate(secs):
                tmps[suffix + "wmam" + str(tmp)] = wmam_a[i]

            for i, tmp in enumerate(wmams):
                tmps[suffix + "wmam1-" + str(tmp)] = wmam[i]



            for i, tmp in enumerate(secs):
                tmps[suffix + "smam" + str(tmp)] = smam_a[i]

            for i, tmp in enumerate(smamss):
                tmps[suffix + "smams" + str(tmp) + "-1"] = smams_d[i]

            for i, tmp in enumerate(wmamss):
                tmps[suffix + "wmams" + str(tmp) + "-1"] = wmams_d[i]

            for i, tmp in enumerate(smams):
                tmps[suffix + "smam1-" + str(tmp)] = smam[i]

            for b1, col in zip(bb, bbs_col):
                for bs, b2 in zip(bbs, b1):
                    tmps[suffix + col + "-" + str(bs)] = b2



            for i, tmp in enumerate(rsis):
                tmps[suffix + "rsi-" + str(tmp)] = rsi[i]

            for i, tmp in enumerate(dips):
                tmps[suffix + "dip-" + str(tmp)] = dip[i]

            for i, tmp in enumerate(dims):
                tmps[suffix + "dim-" + str(tmp)] = dim[i]        

            for i, tmp in enumerate(dxs):
                tmps[suffix + "dx-" + str(tmp)] = dx[i]

            for i, foot_name in enumerate(foot):
                tmps[foot_name] = f[i]    

            for i, tmp in enumerate(oanda_ords):
                tmps["ord" + str(tmp)] = ord[i]

            for i, tmp in enumerate(oanda_poss):
                tmps["pos" + str(tmp)] = pos[i]

            for i, tmp in enumerate(satrs):
                tmps[suffix + "satr-" + str(tmp)] = satr[i]

            for i, tmp in enumerate(smas):
                tmps[suffix + "smac-" + str(tmp)] = smac[i]

            for i, tmp in enumerate(atrs):
                tmps[suffix + "atr-" + str(tmp)] = atr[i]   
                
            for i, tmp in enumerate(ds):
                tmps[suffix + "ehd-" + str(tmp)] = eh[i]

            for i, tmp in enumerate(ds):
                tmps[suffix + "eld-" + str(tmp)] = el[i]   
                
             for i, hor_col in enumerate(hor_cols):
                tmps[hor_col] = hor[i]    
                           
            for i, highlow_col in enumerate(highlow_cols):
                tmps[highlow_col] = highlow[i]
                
            for i, tmp in enumerate(subs):
                tmps[suffix + "sub-" + str(tmp)] = sub[i]      

            for i, tmp in enumerate(d1s):
                tmps[suffix + "md1-" + str(tmp)] = m[i]

            for i, tmp in enumerate(d1s):
                tmps[suffix + "wmd1-" + str(tmp)] = wm[i]
                
            for i, tmp in enumerate(ds):
                tmps[suffix + "d-" + str(tmp)] = d[i]

            """
            for i, tmp in enumerate(d1s):
                tmps[suffix + "ehd1-" + str(tmp)] = eh[i]

            for i, tmp in enumerate(d1s):
                tmps[suffix + "eld1-" + str(tmp)] = el[i]


            """
            if cnt < 30:
                print(score + term)
                print(atr)
            """
            if mode == "new" or mode == "csv" or mode == "new-csv":
                # 新規のDBを作成する場合
                new_score = get_decimal_add(score, term)  # 新しく追加する場合のスコアは実際に予想するときのスコアとする lgbmで使用することを考慮してのこと

                # 元のデータからコピーしたい値がある場合
                for copy_col in copy_cols:
                    if copy_col in orgs.keys():
                        if copy_col == "c":
                            # コピーしたいのがcloseなら新しく追加する場合のスコアは実際に予想するときのスコアとなるのでopenにする
                            tmps["o"] = orgs.get(copy_col)
                        else:
                            tmps[copy_col] = orgs.get(copy_col)

                # tick情報を取り込む場合
                if merge_tick_flg:
                    tick_data = redis_db_org.zrangebyscore(merge_tick_db, get_decimal_sub(new_score, bet_term), get_decimal_sub(new_score, bet_term),
                                                           withscores=True)
                    if len(tick_data) == 1:
                        line = tick_data[0]  # 1件だけのはずなので最初のデータだけ取得
                        body_tick = line[0]
                        score_tick = float(line[1])
                        body = json.loads(body_tick)
                        # 前の足のオープンからクローズまでのtick情報をもつ
                        tmps["tk"] = body.get("tk")

                    elif len(tick_data) == 0:
                        # 存在しないのはおかしいのでエラー
                        print("cannot find tick_data!!!", get_decimal_sub(new_score, bet_term))
                        exit()

                # predict情報を取り込む場合
                if merge_pred_flg:
                    pred_data = redis_db_org.zrangebyscore(merge_pred_db, new_score, new_score, withscores=True)
                    if len(pred_data) == 1:
                        line = pred_data[0]  # 1件だけのはずなので最初のデータだけ取得
                        body_pred = line[0]
                        body = json.loads(body_pred)

                        tmps["UP"] = body.get("UP")
                        tmps["DW"] = body.get("DW")

                    elif len(pred_data) == 0:
                        continue

                tmps["score"] = new_score
                # tmps['time'] = datetime.fromtimestamp(new_score).strftime("%Y-%m-%d %H:%M:%S")

                if use_old_data:
                    result_data_new = redis_db_new.zrangebyscore(new_db_name, new_score, new_score, withscores=True)
                    if len(result_data_new) == 1:
                        line = result_data_new[0]  # 1件だけのはずなので最初のデータだけ取得

                        body_new = line[0]
                        score_new = float(line[1])
                        news = json.loads(body_new)

                        # 削除したい値があれば削除する
                        for col in delete_cols:
                            if col in news.keys():
                                del news[col]

                        # 既存の値を追加する
                        for k in news.keys():
                            tmps[k] = news[k]

                        if mode == "new" or mode == "new-csv":
                            rm_cnt = redis_db_new.zremrangebyscore(new_db_name, new_score, new_score)  # 削除した件数取得
                            if rm_cnt != 1:
                                # 削除できなかったらおかしいのでエラーとする
                                print("cannot remove!!!", new_score)
                                exit()

                            redis_db_new.zadd(new_db_name, json.dumps(tmps), new_score)

                            if mode == "new-csv":
                                # csv_dictのそれぞれのカラムごとの配列に値を足していく。なければ配列を作成する
                                make_dict_data(csv_dict, tmps)
                                # csv_list.append(news)
                        elif mode == "csv":
                            make_dict_data(csv_dict, tmps)
                            # csv_list.append(news)

                    elif len(result_data_new) > 1:
                        print("new data exists error!!! cnt:", len(result_data_new))
                        print("score", new_score)
                    else:
                        if mode == "new":
                            redis_db_new.zadd(new_db_name, json.dumps(tmps), new_score)
                        else:
                            # 既存データがない場合
                            # 使用した既存データがない場合とある場合ではCSVに出力するカラム数が変わってきてしまうので無ければ一律データ出力しない
                            continue

                else:
                    if mode == "new" or mode == "new-csv":

                        # 既存レコードがあるばあい、削除して追加
                        tmp_val = redis_db_new.zrangebyscore(new_db_name, new_score, new_score)
                        if len(tmp_val) >= 1:
                            rm_cnt = redis_db_new.zremrangebyscore(new_db_name, new_score, new_score)  # 削除した件数取得
                            if rm_cnt != 1:
                                # 削除できなかったらおかしいのでエラーとする
                                print("cannot remove!!!", score)
                                exit()

                        redis_db_new.zadd(new_db_name, json.dumps(tmps), new_score)

                        if mode == "new-csv":
                            make_dict_data(csv_dict, tmps)
                            # csv_list.append(tmps)
                    elif mode == "csv":
                        # pass
                        make_dict_data(csv_dict, tmps)
                        # csv_list.append(tmps)

            elif mode == "normal":
                for k, v in tmps.items():
                    # 既存の値にインデックス情報を追加する
                    orgs[k] = v

                # 削除したい値があれば削除する
                for col in delete_cols:
                    if col in orgs.keys():
                        del orgs[col]

                rm_cnt = redis_db_org.zremrangebyscore(db, score, score)  # 削除した件数取得
                if rm_cnt != 1:
                    # 削除できなかったらおかしいのでエラーとする
                    print("cannot remove!!!", score)
                    exit()

                redis_db_org.zadd(db, json.dumps(orgs), score)

            # if cnt % 100000 == 0:
            #    print(datetime.now(), cnt)

        print("time5:", time.perf_counter() - start5)
        print("cnt:", cnt)
        print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        """
        del close_data, high_data, low_data, result_data,

        try:
            del rsi_data
        except NameError:
            pass

        try:
            del smac_data
        except NameError:
            pass


        print("after del memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        gc.collect()

        print("after gc memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        """

    if mode == "csv" or mode == "new-csv":
        # 列名を取得
        # csv_list = sorted(csv_list, key=lambda x: x['score'])
        # tmp_data = csv_list[0]#最初のデータのみ取得
        # csv_regist_cols = list(tmp_data.keys())
        # csv_regist_cols.remove("time")#timeはいらないので削除
        print("memory1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        df_dict = pd.DataFrame(csv_dict)
        print("memory2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        """
        tmp_columns = df_dict.columns.tolist()
        tmp_columns.remove("score")
        tmp_dict = {}
        for col in tmp_columns:
            tmp_dict[col] = "float16"
        """

        print(df_dict[:100])
        del csv_dict
        gc.collect()
        print("memory3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        csv_regist_cols = df_dict.columns.tolist()
        csv_regist_cols.sort()  # カラムを名前順にする

        # o,scoreは最後にしたいので削除後、追加
        if 'o' in csv_regist_cols:
            csv_regist_cols.remove('o')
            csv_regist_cols.append('o')
        if 'score' in csv_regist_cols:
            csv_regist_cols.remove('score')
            csv_regist_cols.append('score')

        df_dict = df_dict.sort_values('score', ascending=True)  # scoreの昇順　古い順にする
        print(df_dict[:100])
        print(df_dict[-100:])
        input_name = list_to_str(csv_regist_cols, "@")
        end_tmp = end + timedelta(days=-1)

        tmp_file_name = symbol + "_B" + str(bet_term) + "_IN-" + input_name + "_" + \
                        date_to_str(start, format='%Y%m%d') + "-" + date_to_str(end_tmp,
                                                                                format='%Y%m%d') + "_" + socket.gethostname() + "_" + db_source
        db_name_file = "INPUT_FILE_NO_" + symbol
        # win2のDBを参照してモデルのナンバリングを行う
        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
        if len(result) == 0:
            print("CANNOT GET INPUT_FILE_NO")
            exit(1)
        else:
            newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

            for line in result:
                body = line[0]
                score = float(line[1])
                tmps = json.loads(body)
                tmp_name = tmps.get("input_name")
                if tmp_name == tmp_file_name:
                    # 同じファイルがないが確認
                    print("The File Already Exists!!!")
                    exit(1)

            # DBにモデルを登録
            child = {
                'input_name': tmp_file_name,
                'no': newest_no
            }
            r.zadd(db_name_file, json.dumps(child), newest_no)

        # csv_dir = "/db2/csv/" + symbol + "_bt" + str(bet_term) + "_" + "2subd100-30satr5-pred" + "_db" + str(new_db_no) + "/"
        csv_path = "/db2/lgbm/" + symbol + "/input_file/" + "IF" + str(newest_no)
        csv_file_name = csv_path + ".csv"
        pickle_file_name = csv_path + ".pickle"

        print("file_name", "IF" + str(newest_no))
        print("input_name", tmp_file_name)
        #print(df_dict.info)
        # print("csv_regist_cols", csv_regist_cols)

        tmp_dict = {}
        for col in csv_regist_cols:
            if col != 'o' and ("hor" in col) and ("hl" in col) == False:
                tmp_dict[col] = 'float32'
        # 型変換
        df_dict = df_dict.astype(tmp_dict, copy=False)
        print(df_dict.info)

        # scoreを型変換し軽くする
        # df = df.astype({"score": "int32"})
        print("memory6", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        df_dict.to_pickle(pickle_file_name)

    print(datetime.now())
    print("FINISH")
    print("Processing Time(Sec)", time.perf_counter() - start_time)


if __name__ == '__main__':

    # start, end, org_db_no, new_db_no, bet_term, term(抽出元のレコード間隔秒数), symbol, org_db, new_db, mode, use_old_dataをリストで保持させる
    # mode
    # normal:既存DBに追記 new:新規にインデックスだけのDBを作成する csv:csvファイルに書き込む
    # new-csv:インデックスだけのDBに書き込みつつ、csvファイルにも書き込む

    # use_old_data: true or false
    # mode=csv or new-csv の場合にDBの既存データを使用してCSV出力する場合
    dates = [
        #[datetime(2024, 6, 30), datetime(2024, 8, 10), 2, 2, 2, 10, 'USDJPY', 'localhost', 'localhost', 'normal', True],
        #[datetime(2024, 6, 30), datetime(2024, 8, 10), 2, 2, 2, 60, 'USDJPY', 'localhost', 'localhost', 'normal', True],
        #[datetime(2024, 6, 30), datetime(2024, 8, 10), 2, 2, 2, 300, 'USDJPY', 'localhost', 'localhost', 'normal', True],

        [datetime(2024, 2, 1), datetime(2024, 8, 10), 2, 2, 2, 10, 'USDJPY', 'localhost', 'localhost', 'normal', True],
        [datetime(2024, 2, 1), datetime(2024, 8, 10), 2, 2, 2, 60, 'USDJPY', 'localhost', 'localhost', 'normal', True],
        [datetime(2024, 2, 1), datetime(2024, 8, 10), 2, 2, 2, 300, 'USDJPY', 'localhost', 'localhost', 'normal', True],

        #[datetime(2022, 1, 1), datetime(2023, 4, 1), 3, 3, 1, 5, 'USDJPY', 'localhost', 'localhost', 'normal', True],
        #[datetime(2022, 1, 1), datetime(2023, 4, 1), 3, 3, 1, 30, 'USDJPY', 'localhost', 'localhost', 'normal', True],

        #[datetime(2024, 5, 5), datetime(2024, 6, 30), 2, 2, 2, 2, 'USDJPY', 'localhost', 'localhost', 'csv', False],

        #[datetime(2024, 6, 30), datetime(2024, 8, 10), 2, 2, 2, 2, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2023, 1, 1), datetime(2023, 4, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2022, 1, 1), datetime(2023, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2021, 1, 1), datetime(2022, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2020, 1, 1), datetime(2021, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2019, 1, 1), datetime(2020, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2018, 1, 1), datetime(2019, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2017, 1, 1), datetime(2018, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2016, 1, 1), datetime(2017, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2015, 1, 1), datetime(2016, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2014, 1, 1), datetime(2015, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2013, 1, 1), datetime(2014, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2012, 1, 1), datetime(2013, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2011, 1, 1), datetime(2012, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2010, 1, 1), datetime(2011, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2009, 1, 1), datetime(2010, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2008, 1, 1), datetime(2009, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2007, 1, 1), datetime(2008, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2006, 1, 1), datetime(2007, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2005, 1, 1), datetime(2006, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
        #[datetime(2004, 1, 1), datetime(2005, 1, 1), 3, 3, 1, 1, 'USDJPY', 'localhost', 'localhost', 'csv', False],
    ]

    for start, end, org_db_no, new_db_no, bet_term, term, symbol, org_db, new_db, mode, use_old_data in dates:
        make_ind_db(start, end, org_db_no, new_db_no, bet_term, term, symbol, org_db, new_db, mode, use_old_data)

    # 終わったらメールで知らせる
    mail.send_message(host, ": make_ind_db finished!!!")