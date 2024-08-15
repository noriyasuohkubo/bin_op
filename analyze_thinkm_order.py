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

"""
thinkmarketsの発注履歴から勝率と平均獲得pipsを算出する
"""

"""
取引履歴
USDJPY
5/27.28.29 win5(母) amt 30000
5/30.31 win3(自分) amt 30000
6/3 win3(自分) amt 40000
6/4 win3(自分) amt 50000
6/5 win3(自分) amt 60000
6/6 win3(自分) amt 70000
6/7 win3(自分) amt 80000
6/10 win3(自分) amt 90000
6/11,12 win3(自分) amt 99000

EURUSD
5/27.28.29 win3(自分) amt 30000
5/30.31 win5(母) amt 30000
6/3,4,5,6,7 win5(母) amt 30000
"""


symbol = "USDJPY"
#symbol = "EURUSD"
#except_hours = [16,17,18,19,]
except_hours = []


#DB情報
db_name = symbol + "_TM_ORDER"
db_name_history = symbol + "_TM_HISTORY"

db_no = 8
if symbol == "USDJPY":
    host = "win8"
elif symbol == "EURUSD":
    host = "win3" #win5

pip = 0.001 if symbol == "USDJPY" else 0.00001

#レコード削除対象期間
#start = datetime(2024, 6, 30, 23)
start = datetime(2024, 8, 12, 23 )
#start = datetime(2024, 3, 12, 12)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2024, 8, 13, 23)
end_stp = int(time.mktime(end.timetuple()))

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

    open_rate = float(tmps.get("open_rate"))
    close_rate = float(tmps.get("close_rate"))

    stoploss_rate = tmps.get("stoploss_rate")
    try:
        stoploss_rate = float(stoploss_rate)
    except Exception as e:
        #stoploss_rateが空の場合はとばす
        continue

    open_score = int(tmps.get("open_score"))
    close_score = int(tmps.get("close_score"))

    bet_type = tmps.get("bet_type")

    pips = close_rate - open_rate
    if bet_type == "sell":
        pips = pips * -1

    history_pips_list.append(pips)

    if pips>= 0:
        history_win_cnt += 1
    elif pips < 0:
        history_lose_cnt += 1

    #key = str(open_score) + "_" + str(stoploss_rate)
    key = str(open_score)

    tmp_dict = {
        "open_rate": open_rate,
        "close_rate":close_rate,
        "close_score":close_score,
        "pips": pips,
    }

    history_dict[key] = tmp_dict

print("history_cnt:", len(history_dict))
if len(history_dict) != 0:
    print("history_win_rate:", history_win_cnt/len(history_dict))
    print("history_win_cnt:", history_win_cnt)
    print("history_lose_cnt:", history_lose_cnt)
    print("history_avg pips:", np.average(np.array(history_pips_list)))

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)

trade_cnt = len(result_data)


pips_list = [] #発注時のレートで約定したと仮定した場合の利益
pips_real_list = [] #実際約定したレートでの利益
delay_start_list = []
delay_end_list = []
loss_start_rate_list = []
loss_end_rate_list = []

win_cnt = 0
lose_cnt = 0
same_cnt = 0

win_real_cnt = 0
lose_real_cnt = 0
same_real_cnt = 0

probe_order_dict={}
delay_start_order_dict={}
delay_end_order_dict={}
hour_order_dict = {}

close_spread_dict = {}

history_match_cnt = 0

for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    tmp_dt = datetime.fromtimestamp(score)
    if tmp_dt.hour in except_hours:
        continue

    order_score = int(tmps.get("order_score"))
    position_score = int(tmps.get("position_score"))
    deal_score = int(tmps.get("deal_score"))

    sign = int(tmps.get("sign"))
    probe = float(tmps.get("probe"))

    start_rate = float(tmps.get("start_rate"))
    start_rate_tm = float(tmps.get("start_rate_tm"))
    end_rate = float(tmps.get("end_rate"))

    stoploss = float(tmps.get("stoploss"))

    try:
        #以下２つの項目は後から足したので入っていない場合はスプレッド0として処理する
        open_spread = tmps.get("open_spread")
        close_spread = tmps.get("close_spread")
        if open_spread < 1:
            #以前は0.1のような形でDB登録していたのでその場合は10倍する
            open_spread = get_decimal_multi(open_spread, 10)
        if close_spread < 1:
            #以前は0.1のような形でDB登録していたのでその場合は10倍する
            close_spread = get_decimal_multi(close_spread, 10)
    except Exception as e:
        open_spread = 0
        close_spread = 0

    o_spread = get_decimal_divide(open_spread, 2)
    c_spread = get_decimal_divide(close_spread, 2)

    if sign == 0:
        start_rate = get_decimal_add(start_rate, pip * o_spread) #spreadを足す
        end_rate = get_decimal_sub(end_rate, pip * c_spread)
        pips = end_rate - start_rate
    elif sign == 2:
        start_rate = get_decimal_sub(start_rate, pip * o_spread)
        end_rate = get_decimal_add(end_rate, pip * c_spread)
        pips = start_rate - end_rate


    """
    pips_real = end_rate - start_rate_tm
    if sign == 2:
        pips_real = pips_real * -1
    """

    #key = str(position_score) + "_" + str(stoploss)
    key = str(position_score)

    if key in history_dict.keys():
        history_match_cnt += 1
        tmp_dict = history_dict[key]
        start_rate_tm = tmp_dict["open_rate"]
        end_rate_tm = tmp_dict["close_rate"]
        close_score = tmp_dict["close_score"]
        pips_real = tmp_dict["pips"]

        if close_spread in close_spread_dict.keys():
            close_spread_dict[close_spread] += 1
        else:
            close_spread_dict[close_spread] = 1

        # 発注から約定までかかった秒数
        delay_start = position_score - order_score

        delay_end = close_score - deal_score
        if delay_end >= 8:
            #決済に8秒以上かかっている場合は正常に決済されなかったので分析対象外とする
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

            loss_start_rate = get_decimal_sub(start_rate_tm, start_rate) #早く約定していれば安く買えたのに、約定が遅かった分あがってしまったレート差
            loss_end_rate = get_decimal_sub(end_rate, end_rate_tm) #早く約定していれば高く売れたのに、約定が遅かった分さがってしまったレート差
        elif sign == 2:
            loss_start_rate = get_decimal_sub(start_rate, start_rate_tm) #早く約定していれば高く買えたのに、約定が遅かった分さがってしまったレート差
            loss_end_rate = get_decimal_sub(end_rate_tm, end_rate)

        loss_start_rate_list.append(loss_start_rate)
        loss_end_rate_list.append(loss_end_rate)

        order_dict = {
            "order_score": order_score,
            "position_score": position_score,
            "deal_score": deal_score,
            "sign": sign,
            "probe": probe,
            "pips":pips,
            "pips_real":pips_real,
            "delay_start":delay_start,
            "delay_end": delay_end,
            "loss_start_rate": loss_start_rate,
            "loss_end_rate": loss_end_rate,
        }

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


print("close_spread")
for k,v in close_spread_dict.items():
    print("spread:",k, " 件数:",v, " %", v/history_match_cnt*100)

print("全体")
print("history_match_cnt:", history_match_cnt)
print("")
print("win_rate:", win_cnt/history_match_cnt)
print("win_cnt:", win_cnt)
print("lose_cnt:", lose_cnt)
print("avg pips:", np.average(np.array(pips_list)))
print("")
print("win_real_rate:", win_real_cnt/history_match_cnt)
print("win_real_cnt:", win_real_cnt)
print("lose_real_cnt:", lose_real_cnt)
print("avg real pips:", np.average(np.array(pips_real_list)))
print("")
print("loss pips:", "{:.8f}".format(np.average(np.array(pips_list)) - np.average(np.array(pips_real_list))))

print("")
print("avg loss start rate:", "{:.8f}".format(np.average(np.array(loss_start_rate_list))))
print("avg loss end rate:", "{:.8f}".format(np.average(np.array(loss_end_rate_list))))
print("")
print("avg delay start:", np.average(np.array(delay_start_list)))
print("avg delay end:", np.average(np.array(delay_end_list)))


#予想確率毎の統計
probe_order_dict_sorted = sorted(probe_order_dict.items())

print("")
print("予想確率毎の統計")
print("")

for prb, order_list in probe_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    delay_start_list = []
    delay_end_list = []
    loss_start_rate_list = []
    loss_end_rate_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_start_rate_list.append(order_dict["loss_start_rate"])
        loss_end_rate_list.append(order_dict["loss_end_rate"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]

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
    print("win_rate:", win_cnt/tmp_trade_cnt)
    print("win_cnt:", win_cnt)
    print("lose_cnt:", lose_cnt)
    print("avg pips:", np.average(np.array(pips_list)))
    print("")
    print("win_real_rate:", win_real_cnt/tmp_trade_cnt)
    print("win_real_cnt:", win_real_cnt)
    print("lose_real_cnt:", lose_real_cnt)
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("")
    print("loss pips:", np.average(np.array(pips_list)) - np.average(np.array(pips_real_list)))
    print("")
    print("avg loss start rate:", np.average(np.array(loss_start_rate_list)))
    print("avg loss end rate:", np.average(np.array(loss_end_rate_list)))
    print("")
    print("avg delay start:", np.average(np.array(delay_start_list)))
    print("avg delay end:", np.average(np.array(delay_end_list)))


#時間毎の統計
hour_order_dict_sorted = sorted(hour_order_dict.items())
print("")
print("時間毎の統計")
print("")

for h, order_list in hour_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    delay_start_list = []
    delay_end_list = []
    loss_start_rate_list = []
    loss_end_rate_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_start_rate_list.append(order_dict["loss_start_rate"])
        loss_end_rate_list.append(order_dict["loss_end_rate"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]

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
    print(h)
    print("trade_cnt:", tmp_trade_cnt)
    print("")
    print("win_rate:", win_cnt/tmp_trade_cnt)
    print("win_cnt:", win_cnt)
    print("lose_cnt:", lose_cnt)
    print("avg pips:", np.average(np.array(pips_list)))
    print("")
    print("win_real_rate:", win_real_cnt/tmp_trade_cnt)
    print("win_real_cnt:", win_real_cnt)
    print("lose_real_cnt:", lose_real_cnt)
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("")
    print("loss pips:", np.average(np.array(pips_list)) - np.average(np.array(pips_real_list)))
    print("")
    print("avg loss start rate:", np.average(np.array(loss_start_rate_list)))
    print("avg loss end rate:", np.average(np.array(loss_end_rate_list)))
    print("")
    print("avg delay start:", np.average(np.array(delay_start_list)))
    print("avg delay end:", np.average(np.array(delay_end_list)))


print("")
print("delay_start毎の統計")
print("")
#delay_start毎の統計
delay_start_order_dict_sorted = sorted(delay_start_order_dict.items())

for delay_start, order_list in delay_start_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    delay_start_list = []
    delay_end_list = []
    loss_start_rate_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_start_rate_list.append(order_dict["loss_start_rate"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]

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
    print(delay_start)
    print("trade_cnt:", tmp_trade_cnt)
    print("")
    print("win_rate:", win_cnt/tmp_trade_cnt)
    print("win_cnt:", win_cnt)
    print("lose_cnt:", lose_cnt)
    print("avg pips:", np.average(np.array(pips_list)))
    print("")
    print("win_real_rate:", win_real_cnt/tmp_trade_cnt)
    print("win_real_cnt:", win_real_cnt)
    print("lose_real_cnt:", lose_real_cnt)
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("")
    print("loss pips:", np.average(np.array(pips_list)) - np.average(np.array(pips_real_list)))
    print("")
    print("avg loss start rate:", np.average(np.array(loss_start_rate_list)))

print("")
print("delay_end毎の統計")
print("")
#delay_end毎の統計
delay_end_order_dict_sorted = sorted(delay_end_order_dict.items())

for delay_end, order_list in delay_end_order_dict_sorted:

    pips_list = []
    pips_real_list = []
    delay_start_list = []
    delay_end_list = []
    loss_end_rate_list = []

    win_cnt = 0
    lose_cnt = 0

    win_real_cnt = 0
    lose_real_cnt = 0

    for order_dict in order_list:
        pips_list.append(order_dict["pips"])
        pips_real_list.append(order_dict["pips_real"])
        delay_start_list.append(order_dict["delay_start"])
        delay_end_list.append(order_dict["delay_end"])
        loss_end_rate_list.append(order_dict["loss_end_rate"])

        pips = order_dict["pips"]
        pips_real = order_dict["pips_real"]

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
    print(delay_end)
    print("trade_cnt:", tmp_trade_cnt)
    print("")
    print("win_rate:", win_cnt/tmp_trade_cnt)
    print("win_cnt:", win_cnt)
    print("lose_cnt:", lose_cnt)
    print("avg pips:", np.average(np.array(pips_list)))
    print("")
    print("win_real_rate:", win_real_cnt/tmp_trade_cnt)
    print("win_real_cnt:", win_real_cnt)
    print("lose_real_cnt:", lose_real_cnt)
    print("avg real pips:", np.average(np.array(pips_real_list)))
    print("")
    print("loss pips:", np.average(np.array(pips_list)) - np.average(np.array(pips_real_list)))
    print("")
    print("avg loss end rate:", np.average(np.array(loss_end_rate_list)))