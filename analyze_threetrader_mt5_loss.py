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
threetrader(MT5)の発注履歴から理論上と実際取引のレートのスリップを日毎に取得する
"""

symbol = "USDJPY"
except_hours = []

#レコード対象期間
start_dt = datetime(2024, 11, 30, 23 )
end_dt = datetime(2025, 1, 1, 23)

#broker = "THREETRADER"
broker = "VANTAGE"

if broker == "THREETRADER":
    fee = 0.004 #手数料
elif broker == "VANTAGE":
    fee = 0.0#手数料

#対象の注文方法、決済方法
#target_order_type = "STOP" #DEAL:成行 STOP:逆指値
#target_deal_type = "STOP"
target_order_type = "DEAL" #DEAL:成行 STOP:逆指値
target_deal_type = "DEAL"

if target_order_type == "DEAL" and target_deal_type == "DEAL":
    db_name = symbol + "_MT5_" + broker  + "_TRADE_ORDER"
    db_name_history = symbol + "_MT5_" + broker + "_TRADE_HISTORY"

elif target_order_type == "STOP" and target_deal_type == "STOP":
    db_name = symbol + "_MT5_" + broker  + "_TRADE_STOP_ORDER"
    db_name_history = symbol + "_MT5_" + broker + "_TRADE_STOP_HISTORY"

db_no = 8

host = "192.168.1.14"

pip = 0.001 if symbol == "USDJPY" else 0.01

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

while True:
    if start_dt > end_dt:
        break

    tmp_end_dt = start_dt + timedelta(days=1)

    start_stp = int(time.mktime(start_dt.timetuple()))
    end_stp = int(time.mktime(tmp_end_dt.timetuple()))

    #先に取引履歴DBを取得
    result_data_history = redis_db.zrangebyscore(db_name_history, start_stp, end_stp, withscores=True)

    #print("history_cnt:", len(result_data_history))

    history_dict = {}
    history_pips_list = []
    history_win_cnt = 0
    history_lose_cnt = 0
    #print("result_data_history length:", len(result_data_history))

    if len(result_data_history) == 0:
        start_dt = start_dt + timedelta(days=1)
        continue

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

        #手数料考慮
        pips = get_decimal_sub(pips, fee)

        history_pips_list.append(pips)

        if pips>= 0:
            history_win_cnt += 1
        elif pips < 0:
            history_lose_cnt += 1

        key = position

        tmp_dict = {
            "position": position,
            "open_rate": open_rate,
            "close_rate":close_rate,
            "open_score":open_score,
            "close_score":close_score,
            "pips": pips,
        }

        history_dict[key] = tmp_dict

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    #print("result_data length:", len(result_data))

    if len(result_data) == 0:
        start_dt = start_dt + timedelta(days=1)
        continue

    trade_cnt = len(result_data)

    pips_list = [] #発注時のレートで約定したと仮定した場合の利益
    pips_real_list = [] #実際約定したレートでの利益
    pips_order_list = [] #注文直前のレートでの利益
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

        start_rate = float(tmps.get("start_rate"))
        #start_rate_tm = float(tmps.get("start_rate_local"))
        end_rate = float(tmps.get("end_rate"))

        start_rate_order = tmps.get("start_rate_order")
        end_rate_order = tmps.get("end_rate_order")

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

        pips = get_decimal_sub(pips, fee)

        key = position

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

            #実際に約定したときのレートと注文直前のレートの差を求める
            if start_rate_order != None and end_rate_order !=None:
                if sign == 0:
                    loss_start_rate_order = get_decimal_sub(start_rate_tm,start_rate_order)
                    loss_end_rate_order = get_decimal_sub(end_rate_order, end_rate_tm)
                    pips_order = get_decimal_sub(end_rate_order, start_rate_order)
                elif sign == 2:
                    loss_start_rate_order  = get_decimal_sub(start_rate_order,start_rate_tm)
                    loss_end_rate_order  = get_decimal_sub(end_rate_tm, end_rate_order)
                    pips_order = get_decimal_sub(start_rate_order, end_rate_order)

                pips_order = get_decimal_sub(pips_order, fee)
                pips_order_list.append(pips_order)
                loss_start_rate_order_list.append(loss_start_rate_order)
                loss_end_rate_order_list.append(loss_end_rate_order)

            loss_start_rate_list.append(loss_start_rate)
            loss_end_rate_list.append(loss_end_rate)

            """
            print("key",key)
            print("order_score", order_score)
            print("sign", sign)
            print("loss_start_rate", loss_start_rate)
            print("loss_end_rate", loss_end_rate)
            
            if start_rate_order != None and end_rate_order != None:
                print("loss_start_rate_order", loss_start_rate_order)
                print("loss_end_rate_order", loss_end_rate_order)
            """

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

    if history_match_cnt == 0:
        start_dt = start_dt + timedelta(days=1)
        continue

    """
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
    """

    print(start_dt,tmp_end_dt)
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
        print("avg loss end rate order:", "{:.8f}".format(np.average(np.array(loss_end_rate_order_list))))

    print("")

    start_dt = start_dt + timedelta(days=1)

