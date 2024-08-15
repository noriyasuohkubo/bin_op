import json
import pickle

import numpy as np
import redis
from datetime import datetime
import time
from decimal import Decimal
import pandas as pd
from util import  *
"""
testLstmFX2_rgr_limit.pyで作成した取引履歴をもとに
以前の取引の途中成績と今回の取引の成績の関係を調査する
以前の途中成績が悪ければ、今回の取引も悪いと想定。
"""

dir = "/app/fx/deal_history/"
file =   "USDJPY_LT1_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_FDB1800-300-5-d1_BNL2_BDIV0.01_201001_202210_L-RATE0.0005_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS5120_SD0_SHU1_EL20-21-22_ub1_MN311-30_BORDER_0.54_EXTBORDER_0.5.pickle"
file_name = dir + file

times = [18]
borders = [-0.01,]
good_borders = [0.01,]

div_sec = 12


db_no = 2
start = datetime(2022, 11, 1)
end = datetime(2023, 10, 1)
score_rate_dict_file_name = dir + "score_rate_dict-" + start.strftime('%Y%m%d') + "-" + end.strftime('%Y%m%d') + ".pickle"

#レート情報がpickleで存在しなければ取得してpickle作成し保存
if not os.path.isfile(score_rate_dict_file_name):

    start_stp = int(time.mktime(start.timetuple()))
    end_stp = int(time.mktime(end.timetuple()))

    redis_db = redis.Redis(host="localhost", port=6379, db=db_no, decode_responses=True)
    base_data = redis_db.zrangebyscore("USDJPY_2_0", start_stp, end_stp + 60 * 60 * 24,withscores=True)  # end_stpは多めにとっておく

    score_rate_dict = {}

    for line in base_data:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)
        score_rate_dict[score - 2] = tmps.get("c")

    with open(score_rate_dict_file_name, 'wb') as f:
        pickle.dump(score_rate_dict, f)
        print("score_rate_dict make finish")

else:
    with open(score_rate_dict_file_name, 'rb') as f:
        score_rate_dict = pickle.load(f)
        print("score_rate_dict laod finish")

with open(file_name, 'rb') as f:
    deal_dict = pickle.load(f)
    print("deal_dict laod finish")

print("file:", file)
print("deal length:", len(deal_dict))
print("times:",times)
print("borders:", borders)

# 古い順にならびかえ
sorted_d = sorted(deal_dict.items(), key=lambda x: x[0], reverse=False)

profit_sum_normal = [] #条件に合致しない場合の取引利益
profit_sum_bad = [] #条件に合致た場合の取引利益
profit_sum_good = [] #条件に合致た場合の取引利益
profit_prev = []
profit_prev_real = []

types_total = []
div_total = []
profit_total = []

for k, v in sorted_d:
    this_sprice = v["sprice"]
    this_type = v["type"]
    this_profit = v["profit_pips"]

    prev_rate = score_rate_dict.get(k - div_sec)
    div = get_divide(prev_rate, this_sprice)

    types_total.append(this_type)
    div_total.append(div)
    profit_total.append(this_profit)

    continue_flg = False
    bad_flgs = []
    good_flgs = []
    tmp_prev_profits = []
    tmp_prev_profits_real = []
    for i,t in enumerate(times):
        target_sc = k - t
        target_deal = deal_dict.get(target_sc)
        if target_deal != None:
            target_deal_type = target_deal["type"]
            target_deak_real_profit = target_deal["profit_pips"]
            if target_deal_type != this_type:
                #取引タイプが異なるなら集計対象外
                continue_flg =True
                break
            else:

                target_deal_sprice = target_deal["sprice"]
                if this_type == "BUY":
                    target_deal_profit = float(Decimal(str(this_sprice)) - Decimal(str(target_deal_sprice)))
                else:
                    target_deal_profit = float(Decimal(str(target_deal_sprice)) - Decimal(str(this_sprice)))

                tmp_prev_profits.append(target_deal_profit)
                tmp_prev_profits_real.append(target_deak_real_profit)

                if target_deal_profit <= borders[i]:
                    bad_flgs.append(True)
                else:
                    bad_flgs.append(False)

                if target_deal_profit >= good_borders[i]:
                    good_flgs.append(True)
                else:
                    good_flgs.append(False)

        else:
            # 該当する以前の取引がないなら集計対象外
            continue_flg = True
            break

    if continue_flg:
        continue

    if good_flgs.count(False) == 0:
        #全ての以前の取引の途中成績が良い場合
        profit_sum_good.append(this_profit)

    if bad_flgs.count(False) == 0:
        #全ての以前の取引の途中成績が悪い場合
        profit_sum_bad.append(this_profit)
        profit_prev = profit_prev + tmp_prev_profits
        profit_prev_real = profit_prev_real + tmp_prev_profits_real
    else:
        #以前の取引のうち、一つでも成績が悪くなかった取引があった場合
        profit_sum_normal.append(this_profit)

total = profit_sum_normal + profit_sum_bad

print("total length:", len(total), " avg profit:", np.average(np.array(total)))
print("normal length:", len(profit_sum_normal), " avg profit:", np.average(np.array(profit_sum_normal)))
print("bad length:", len(profit_sum_bad), " avg profit:", np.average(np.array(profit_sum_bad)))
print("profit_prev length:",len(profit_prev), " avg profit:", np.average(np.array(profit_prev)))
print("profit_prev_real length:",len(profit_prev_real), " avg profit:", np.average(np.array(profit_prev_real)))
print("good length:", len(profit_sum_good), " avg profit:", np.average(np.array(profit_sum_good)))
"""
profit_total = np.array(profit_total)
types_total = np.array(types_total)
div_total = np.array(div_total)

type_buy_ind = np.where( types_total == "BUY" )[0]
type_sell_ind = np.where( types_total == "SELL" )[0]


buy_profit_arr =  profit_total[type_buy_ind]
sell_profit_arr =  profit_total[type_sell_ind]

buy_div_arr = div_total[type_buy_ind]
sell_div_arr = div_total[type_sell_ind]

print("buy_div_arr length:", len(buy_div_arr))
print("sell_div_arr length:", len(sell_div_arr))

start = -10
end = 10
len_cnt = 0
print("BUYのdivごとの利益")
while True:
    next = get_decimal_add(start, 0.05)
    if next > end:
        break

    ind = np.where((start <= buy_div_arr) & (buy_div_arr < next))[0]
    len_cnt += len(ind)
    if len(ind) >0:
        tmp_profit = buy_profit_arr[ind]
        print(start, "~", next, "avg pips:", np.average(tmp_profit), " length:", len(tmp_profit))

    start = next
print("len_cnt:", len_cnt)

print("")
print("SELLのdivごとの利益")
start = -10
end = 10

while True:
    next = get_decimal_add(start, 0.05)
    if next > end:
        break

    ind = np.where((start <= sell_div_arr) & (sell_div_arr < next))[0]
    if len(ind) >0:
        tmp_profit = sell_profit_arr[ind]
        print(start, "~", next, "avg pips:", np.average(tmp_profit), " length:", len(tmp_profit))

    start = next

"""
