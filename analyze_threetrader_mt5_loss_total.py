import os
import signal
import sys
import time
from datetime import datetime, timedelta, date
from chk_summertime import *
import redis
import json
import numpy as np
from util import *


symbol = "USDJPY"
except_hours = []

#レコード対象期間
#2025/3/4からすべて100万通貨
start_dt = datetime(2026, 2, 1,   )
end_dt = datetime(2026, 3, 1,  )

start_stp = int(time.mktime(start_dt.timetuple()))
end_stp = int(time.mktime(end_dt.timetuple()))

#broker = "THREETRADER"
broker = "VANTAGE"
#broker = "EXNESS_ZERO"
#broker = "EXNESS_PRO"
#broker = "EXNESS_PRO_IE"

if broker == "THREETRADER":
    usd_fee_flg = False #True:ドル建ての手数料の場合
    #fee = 0.004 - 0.0005 #手数料 - キャッシュバック
    fee = 0.004
    host = "192.168.1.14" #win3
    symbol = "USDJPY.raw"
elif broker == "VANTAGE":
    usd_fee_flg = False #True:ドル建ての手数料の場合
    #fee = 0.0 - 0.0013 #手数料
    fee = 0.0
    host = "192.168.1.15" #win5
elif broker == "EXNESS_ZERO":
    usd_fee_flg = True #True:ドル建ての手数料の場合
    #fee = 7 - 1#手数料 USD
    fee = 7
    host = "win5"
    symbol = "USDJPYz"
elif broker == "EXNESS_PRO":
    usd_fee_flg = False #True:ドル建ての手数料の場合
    #fee = 0.0 - 0.0013#手数料
    fee = 0.0
    host = "win3"
elif broker == "EXNESS_PRO_IE":
    usd_fee_flg = False #True:ドル建ての手数料の場合
    #fee = 0.0 - 0.0013#手数料
    fee = 0.0
    host = "win6"

#対象の注文方法、決済方法
#target_order_type = "STOP" #DEAL:成行 STOP:逆指値
#target_deal_type = "STOP"
target_order_type = "DEAL" #DEAL:成行 STOP:逆指値
target_deal_type = "DEAL"

#対象の最低予想閾値
target_predict_min = 0.5

#対象のLOT
lot_condition = None #None:指定なし
#lot_condition = 1.0 #None:指定なし

#対象の最高予想閾値
target_predict_max = None #None:指定なし
#target_predict_max = 0.64

#損失表示対象のpips
target_loss_pips = -0.05

if target_order_type == "DEAL" and target_deal_type == "DEAL":
    db_name = symbol + "_MT5_" + broker  + "_TRADE_ORDER"
    db_name_history = symbol + "_MT5_" + broker + "_TRADE_HISTORY"

elif target_order_type == "STOP" and target_deal_type == "DEAL":
    db_name = symbol + "_MT5_" + broker  + "_STOP_DEAL_TRADE_ORDER"
    db_name_history = symbol + "_MT5_" + broker + "_STOP_DEAL_TRADE_HISTORY"

elif target_order_type == "STOP" and target_deal_type == "STOP":
    db_name = symbol + "_MT5_" + broker  + "_STOP_TRADE_ORDER"
    db_name_history = symbol + "_MT5_" + broker + "_STOP_TRADE_HISTORY"

db_no = 8
print(db_name,db_name_history)

pip = 0.001


redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

#先に取引履歴DBを取得
result_data_history = redis_db.zrangebyscore(db_name_history, start_stp, end_stp, withscores=True)

#open_scoreとstoplossをキーとして辞書作成
#print("history_cnt:", len(result_data_history))

history_dict = {}
history_pips_list = []
history_win_cnt = 0
history_lose_cnt = 0

for i, v in enumerate(result_data_history):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    position = int(tmps.get("position_id"))
    open_rate = float(tmps.get("start_rate"))
    close_rate = float(tmps.get("end_rate"))

    open_score = float(tmps.get("order_score"))
    close_score = float(tmps.get("deal_score"))

    bet_type = str(tmps.get("sign"))

    pips = get_decimal_sub(close_rate, open_rate)
    if bet_type == "2":
        pips = pips * -1

    # 手数料考慮
    if usd_fee_flg:
        real_fee = (open_rate * fee) /100000
        pips = get_decimal_sub(pips, real_fee)
    else:
        pips = get_decimal_sub(pips, fee)
    #print(pips)
    history_pips_list.append(pips)

    if pips >= 0:
        history_win_cnt += 1
    elif pips < 0:
        history_lose_cnt += 1

    key = position

    tmp_dict = {
        "position": position,
        "open_rate": open_rate,
        "close_rate": close_rate,
        "open_score": open_score,
        "close_score": close_score,
        "pips": pips,
    }

    history_dict[key] = tmp_dict

print("history_cnt:", len(history_dict))
if len(history_dict) != 0:
    print("history_win_rate:", history_win_cnt/len(history_dict))
    print("history_win_cnt:", history_win_cnt)
    print("history_lose_cnt:", history_lose_cnt)
    print("history_avg pips:", np.average(np.array(history_pips_list)))
    print("history_sum pips:", sum(history_pips_list))
result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)

trade_cnt = len(result_data)


pips_list = [] #発注時のレートで約定したと仮定した場合の利益
pips_real_list = [] #実際約定したレートでの利益
pips_order_list = []  # 注文直前のレートでの利益

delay_start_list = []
delay_end_list = []

loss_start_rate_list = []
loss_end_rate_list = []

loss_start_rate_order_list = []
loss_end_rate_order_list = []

win_cnt = 0
lose_cnt = 0
same_cnt = 0

win_real_cnt = 0
lose_real_cnt = 0
same_real_cnt = 0

lot_order_dict = {}
probe_order_dict={}
delay_start_order_dict={}
delay_end_order_dict={}
hour_order_dict = {}

open_spread_dict = {}
close_spread_dict = {}

history_match_cnt = 0

for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    tmp_dt = datetime.fromtimestamp(score)
    if tmp_dt.hour in except_hours:
        continue

    order_type = tmps.get("order_type")
    deal_type = tmps.get("deal_type")

    if target_order_type != order_type or target_deal_type != deal_type:
        #対象外の発注、決済方法は除外
        continue

    position = int(tmps.get("position"))

    order_score = float(tmps.get("order_score"))
    position_score = float(tmps.get("position_score"))
    deal_score = float(tmps.get("deal_score"))

    sign = int(tmps.get("sign"))
    probe = float(tmps.get("probe"))

    if probe < target_predict_min:
        #対象予想閾値より低いなら対象外
        continue

    start_rate = float(tmps.get("start_rate"))
    #start_rate_tm = float(tmps.get("start_rate_local"))
    end_rate = float(tmps.get("end_rate"))

    start_rate_order = tmps.get("start_rate_order")
    end_rate_order = tmps.get("end_rate_order")

    volume = tmps.get("volume")

    if lot_condition != None and float(volume) != lot_condition:
        continue

    if target_predict_max != None and target_predict_max <= probe :
        continue
    #以下２つの項目は後から足したので入っていない場合はスプレッド0として処理する
    open_spread = tmps.get("open_spread")
    close_spread = tmps.get("close_spread")

    o_spread = get_decimal_divide(open_spread, 2)
    c_spread = get_decimal_divide(close_spread, 2)

    if sign == 0:
        start_rate = get_decimal_add(start_rate, get_decimal_multi(pip, o_spread)) #spreadを足す
        end_rate = get_decimal_sub(end_rate, get_decimal_multi(pip, c_spread))
        pips = get_decimal_sub(end_rate, start_rate)
    elif sign == 2:
        start_rate = get_decimal_sub(start_rate, get_decimal_multi(pip, o_spread))
        end_rate = get_decimal_add(end_rate, get_decimal_multi(pip, c_spread))
        pips = get_decimal_sub(start_rate, end_rate)

    # 手数料考慮
    if usd_fee_flg:
        real_fee = (open_rate * fee) /100000
        pips = get_decimal_sub(pips, real_fee)
    else:
        pips = get_decimal_sub(pips, fee)

    key = position
    if (key in history_dict.keys()) == False:

        print("key not exists:", tmps.get("position_score"))

    elif key in history_dict.keys():
        history_match_cnt += 1
        tmp_dict = history_dict[key]
        start_rate_tm = tmp_dict["open_rate"]
        end_rate_tm = tmp_dict["close_rate"]
        close_score = tmp_dict["close_score"]
        pips_real = tmp_dict["pips"]

        if open_spread in open_spread_dict.keys():
            open_spread_dict[open_spread] += 1
        else:
            open_spread_dict[open_spread] = 1

        if close_spread in close_spread_dict.keys():
            close_spread_dict[close_spread] += 1
        else:
            close_spread_dict[close_spread] = 1

        # 発注から約定までかかった秒数
        delay_start = position_score - order_score

        delay_end = close_score - deal_score
        if delay_end >= 8:
            # 決済に8秒以上かかっている場合は正常に決済されなかったので分析対象外とする
            continue

        pips_list.append(pips)
        pips_real_list.append(pips_real)
        delay_start_list.append(delay_start)
        delay_end_list.append(delay_end)

        if pips >= 0:
            win_cnt += 1
        elif pips < 0:
            lose_cnt += 1

        if pips_real >= 0:
            win_real_cnt += 1
        elif pips_real < 0:
            lose_real_cnt += 1

        if sign == 0:
            loss_start_rate = get_decimal_sub(start_rate_tm, start_rate)  # 早く約定していれば安く買えたのに、約定が遅かった分あがってしまったレート差
            loss_end_rate = get_decimal_sub(end_rate, end_rate_tm)  # 早く約定していれば高く売れたのに、約定が遅かった分さがってしまったレート差

        elif sign == 2:
            loss_start_rate = get_decimal_sub(start_rate, start_rate_tm)  # 早く約定していれば高く買えたのに、約定が遅かった分さがってしまったレート差
            loss_end_rate = get_decimal_sub(end_rate_tm, end_rate)

        # 実際に約定したときのレートと注文直前のレートの差を求める
        if start_rate_order != None and end_rate_order != None:
            if sign == 0:
                loss_start_rate_order = get_decimal_sub(start_rate_tm, start_rate_order)
                loss_end_rate_order = get_decimal_sub(end_rate_order, end_rate_tm)
                pips_order = get_decimal_sub(end_rate_order, start_rate_order)
            elif sign == 2:
                loss_start_rate_order = get_decimal_sub(start_rate_order, start_rate_tm)
                loss_end_rate_order = get_decimal_sub(end_rate_tm, end_rate_order)
                pips_order = get_decimal_sub(start_rate_order, end_rate_order)

            # 手数料考慮
            if usd_fee_flg:
                real_fee = (open_rate * fee) / 100000
                pips_order = get_decimal_sub(pips_order, real_fee)
            else:
                pips_order = get_decimal_sub(pips_order, fee)

            pips_order_list.append(pips_order)
            loss_start_rate_order_list.append(loss_start_rate_order)
            loss_end_rate_order_list.append(loss_end_rate_order)

        loss_start_rate_list.append(loss_start_rate)
        loss_end_rate_list.append(loss_end_rate)

        order_dict = {
            "order_score": order_score,
            "position_score": position_score,
            "deal_score": deal_score,
            "sign": sign,
            "probe": probe,
            "pips": pips,
            "pips_real": pips_real,
            "pips_order": pips_order,
            "delay_start": delay_start,
            "delay_end": delay_end,
            "loss_start_rate": loss_start_rate,
            "loss_end_rate": loss_end_rate,
            "loss_start_rate_order": loss_start_rate_order,
            "loss_end_rate_order": loss_end_rate_order,
        }


        if volume in lot_order_dict.keys():
            lot_order_dict[volume].append(order_dict)
        else:
            lot_order_dict[volume] = [order_dict]


        probe_str = str(probe)[:4]

        if probe_str in probe_order_dict.keys():
            probe_order_dict[probe_str].append(order_dict)
        else:
            probe_order_dict[probe_str] = [order_dict]

        delay_start_str = str(delay_start)[:1]

        if delay_start_str in delay_start_order_dict.keys():
            delay_start_order_dict[delay_start_str].append(order_dict)
        else:
            delay_start_order_dict[delay_start_str] = [order_dict]

        delay_end_str = str(delay_end)[:1]

        if delay_end_str in delay_end_order_dict.keys():
            delay_end_order_dict[delay_end_str].append(order_dict)
        else:
            delay_end_order_dict[delay_end_str] = [order_dict]

        hour = datetime.fromtimestamp(position_score).hour
        if hour in hour_order_dict.keys():
            hour_order_dict[hour].append(order_dict)
        else:
            hour_order_dict[hour] = [order_dict]

print("order_cnt:", trade_cnt)

print("open_spread")
total_open_spread = 0
total_open_spread_cnt = 0
for k,v in open_spread_dict.items():
    print("spread:",k, " 件数:",v, " %", v/history_match_cnt*100)
    total_open_spread = total_open_spread + int(get_decimal_multi(k, v))
    total_open_spread_cnt = total_open_spread_cnt + v

if total_open_spread_cnt != 0:
    print("avg open_spread:", total_open_spread/total_open_spread_cnt)

print("")
print("close_spread")
total_close_spread = 0
total_close_spread_cnt = 0
for k,v in close_spread_dict.items():
    print("spread:",k, " 件数:",v, " %", v/history_match_cnt*100)
    total_close_spread = total_close_spread + int(get_decimal_multi(k, v))
    total_close_spread_cnt = total_close_spread_cnt + v

if total_close_spread_cnt != 0:
    print("avg close_spread:", total_close_spread/total_close_spread_cnt)

print("")

if total_open_spread_cnt != 0 and total_close_spread_cnt != 0:
    print("avg spread:", (total_open_spread + total_close_spread)/(total_open_spread_cnt + total_close_spread_cnt))

print("全体")
print("history_match_cnt:", history_match_cnt)
print("avg pips:", np.average(np.array(pips_list)))
print("avg real pips:", np.average(np.array(pips_real_list)))
print("loss pips:", "{:.8f}".format(np.average(np.array(pips_list)) - np.average(np.array(pips_real_list))))
print("avg loss start rate:", "{:.8f}".format(np.average(np.array(loss_start_rate_list))))
print("avg loss end rate:", "{:.8f}".format(np.average(np.array(loss_end_rate_list))))
print("avg delay start:", np.average(np.array(delay_start_list)))
print("avg delay end:", np.average(np.array(delay_end_list)))

if len(pips_order_list) != 0:
    print("avg order pips:", np.average(np.array(pips_order_list)))
    print("loss pips order:",
          "{:.8f}".format(np.average(np.array(pips_order_list)) - np.average(np.array(pips_real_list))))
    print("avg loss start rate order:", "{:.8f}".format(np.average(np.array(loss_start_rate_order_list))))
    #for i in loss_start_rate_order_list:
    #    print(i)
    print("avg loss end rate order:", "{:.8f}".format(np.average(np.array(loss_end_rate_order_list))))
    #for i in loss_end_rate_order_list:
    #    print(i)

if len(pips_real_list) != 0:
    print("")
    print("under:", target_loss_pips)
    p_cnt = 0
    for p in pips_real_list:
        if p <= target_loss_pips:
            p_cnt += 1
            print(p)

    print("count:", p_cnt)

#LOT毎の統計
lot_order_dict_sorted = sorted(lot_order_dict.items())

print("")
print("LOT毎の統計")

for lot, order_list in lot_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    pips_order_list = []
    delay_start_list = []
    delay_end_list = []
    loss_start_rate_list = []
    loss_end_rate_list = []
    loss_start_rate_order_list = []
    loss_end_rate_order_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        pips_order_list.append(order_dict["pips_order"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_start_rate_list.append(order_dict["loss_start_rate"])
        loss_end_rate_list.append(order_dict["loss_end_rate"])
        loss_start_rate_order_list.append(order_dict["loss_start_rate_order"])
        loss_end_rate_order_list.append(order_dict["loss_end_rate_order"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]
        pips_order = order_dict["pips_order"]

        if pips>= 0:
            win_cnt += 1
        elif pips < 0:
            lose_cnt += 1

        if pips_real>= 0:
            win_real_cnt += 1
        elif pips_real < 0:
            lose_real_cnt += 1

    tmp_trade_cnt = len(order_list)

    print("")
    print("LOT:",lot)
    print("trade_cnt:", tmp_trade_cnt)
    print("")
    print("avg pips:", np.average(np.array(pips_list)))
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("loss pips:", "{:.8f}".format(np.average(np.array(pips_list)) - np.average(np.array(pips_real_list))))
    print("avg loss start rate:", "{:.8f}".format(np.average(np.array(loss_start_rate_list))))
    print("avg loss end rate:", "{:.8f}".format(np.average(np.array(loss_end_rate_list))))
    print("avg delay start:", np.average(np.array(delay_start_list)))
    print("avg delay end:", np.average(np.array(delay_end_list)))

    if len(pips_order_list) != 0:
        print("avg order pips:", np.average(np.array(pips_order_list)))
        print("loss pips order:",
              "{:.8f}".format(np.average(np.array(pips_order_list)) - np.average(np.array(pips_real_list))))
        print("avg loss start rate order:", "{:.8f}".format(np.average(np.array(loss_start_rate_order_list))))
        #for i in loss_start_rate_order_list:
        #    print(i)
        print("avg loss end rate order:", "{:.8f}".format(np.average(np.array(loss_end_rate_order_list))))
        #for i in loss_end_rate_order_list:
        #    print(i)

#予想確率毎の統計
probe_order_dict_sorted = sorted(probe_order_dict.items())


print("")
print("予想確率毎の統計")
print("")

for prb, order_list in probe_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    pips_order_list = []
    delay_start_list = []
    delay_end_list = []
    loss_start_rate_list = []
    loss_end_rate_list = []
    loss_start_rate_order_list = []
    loss_end_rate_order_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        pips_order_list.append(order_dict["pips_order"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_start_rate_list.append(order_dict["loss_start_rate"])
        loss_end_rate_list.append(order_dict["loss_end_rate"])
        loss_start_rate_order_list.append(order_dict["loss_start_rate_order"])
        loss_end_rate_order_list.append(order_dict["loss_end_rate_order"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]
        pips_order = order_dict["pips_order"]

        if pips>= 0:
            win_cnt += 1
        elif pips < 0:
            lose_cnt += 1

        if pips_real>= 0:
            win_real_cnt += 1
        elif pips_real < 0:
            lose_real_cnt += 1

    tmp_trade_cnt = len(order_list)

    print("")
    print(prb)
    print("trade_cnt:", tmp_trade_cnt)
    print("")
    print("avg pips:", np.average(np.array(pips_list)))
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("loss pips:", "{:.8f}".format(np.average(np.array(pips_list)) - np.average(np.array(pips_real_list))))
    print("avg loss start rate:", "{:.8f}".format(np.average(np.array(loss_start_rate_list))))
    print("avg loss end rate:", "{:.8f}".format(np.average(np.array(loss_end_rate_list))))
    print("avg delay start:", np.average(np.array(delay_start_list)))
    print("avg delay end:", np.average(np.array(delay_end_list)))

    if len(pips_order_list) != 0:
        print("avg order pips:", np.average(np.array(pips_order_list)))
        print("loss pips order:",
              "{:.8f}".format(np.average(np.array(pips_order_list)) - np.average(np.array(pips_real_list))))
        print("avg loss start rate order:", "{:.8f}".format(np.average(np.array(loss_start_rate_order_list))))
        #for i in loss_start_rate_order_list:
        #    print(i)
        print("avg loss end rate order:", "{:.8f}".format(np.average(np.array(loss_end_rate_order_list))))
        #for i in loss_end_rate_order_list:
        #    print(i)