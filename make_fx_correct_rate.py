from datetime import datetime
from datetime import timedelta
import time
import redis
import json
from decimal import Decimal
import traceback
import os
import gc
import numpy as np
import math
import sys
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

"""
thinkmarketsなどでの取引データからaxioryなど他の会社のレートを参照してアップデートする
また、約定までの時間差がなかった場合のレートもアップデートする
"""

update_flg = True #True:レートの更新を行う
show_plot = True #True:取引履歴のplot表示を行う

start_day = "2023/09/01 00:00:00" #この時間含む(以上)
end_day = "2023/10/01 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

db_no_trade= 0 #もとの取引履歴が入っているDB
db_name_trade = 'USDJPY_60_FXTF_HISTORY'

db_no_ref = 0 #参照するレートが入っているDB
db_name_ref = 'AXIORY_USDJPY_S1'

loop_term = 4 #取引する際に設定しているループ間隔

redis_db_trade = redis.Redis(host='localhost', port=6379, db=db_no_trade, decode_responses=True)
redis_db_ref = redis.Redis(host='localhost', port=6379, db=db_no_ref, decode_responses=True)


result_data_ref = redis_db_ref.zrangebyscore(db_name_ref, start_stp, end_stp, withscores=True)
print("result_data_ref length:" + str(len(result_data_ref)))

#参照先のレートをスコアをキーに保持
correct_rate_dict = {}
close_history = []
for i, line in enumerate(result_data_ref):
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)

    ask = tmps.get('ask')
    bid = tmps.get('bid')

    mid = float((Decimal(str(ask)) + Decimal(str(bid))) / Decimal("2")) #仲値
    correct_rate_dict[score] = mid

if update_flg:
    result_data_trade = redis_db_trade.zrangebyscore(db_name_trade, start_stp, end_stp, withscores=True)
    print("result_data_trade length:" + str(len(result_data_trade)))

    for i, line in enumerate(result_data_trade):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)
        open_score = tmps.get('open_score')
        close_score = tmps.get('close_score')

        #正しいレートを取得する もし該当する参照レートがない場合はNoneで入れる
        tmps["open_rate_correct"] = correct_rate_dict.get(open_score)
        tmps["close_rate_correct"] = correct_rate_dict.get(close_score)

        #発注と約定のタイムラグがなかったとした場合のレートを入れる
        open_score_correct = open_score - (open_score % loop_term)
        close_score_correct = close_score - (close_score % loop_term)

        tmps["open_rate_no_lag"] = correct_rate_dict.get(open_score_correct)
        tmps["close_rate_no_lag"] = correct_rate_dict.get(close_score_correct)

        # 既存レコードを削除して追加

        rm_cnt = redis_db_trade.zremrangebyscore(db_name_trade, score, score)  # 削除した件数取得
        if rm_cnt != 1:
            # 削除できなかったらおかしいのでエラーとする
            print("cannot remove!!!", score)
            exit(1)

        redis_db_trade.zadd(db_name_trade, json.dumps(tmps), score)


if show_plot:
    #元のレートでの取引結果と正しいレートでの取引結果を表示

    up_org_correct_pips_list = []
    up_correct_no_lag_pips_list = []

    dw_org_correct_pips_list = []
    dw_correct_no_lag_pips_list = []

    up_open_org_correct_list = [] #upの開始時のorgとcorrectでのレートの差
    up_close_org_correct_list = []

    dw_open_org_correct_list = []
    dw_close_org_correct_list = []

    up_open_correct_no_lag_list = []  # upの開始時のcorrectとno_lagでのレートの差
    up_close_correct_no_lag_list = []

    dw_open_correct_no_lag_list = []
    dw_close_correct_no_lag_list = []

    result_data_trade = redis_db_trade.zrangebyscore(db_name_trade, start_stp, end_stp, withscores=True)
    print("result_data_trade length:" + str(len(result_data_trade)))

    trade_result_dict = {}

    for i, line in enumerate(result_data_trade):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        trade_result_dict[score] = tmps


    money_org = 0
    money_history_org = []

    money_correct = 0
    money_history_correct = []

    money_no_lag = 0
    money_history_no_lag = []

    close_history = []

    sorted = sorted(correct_rate_dict.items()) #スコアの昇順にする
    trade_cnt = 0
    for score, close in sorted:
        tmp_profit = 0

        close_history.append(close)

        trade_result = trade_result_dict.get(score)
        if trade_result  != None:

            try:
                open_rate = trade_result["open_rate"]
                close_rate = trade_result["close_rate"]
                open_rate_correct = trade_result["open_rate_correct"]
                close_rate_correct = trade_result["close_rate_correct"]
                open_rate_no_lag = trade_result["open_rate_no_lag"]
                close_rate_no_lag = trade_result["close_rate_no_lag"]
                bet_type = trade_result["bet_type"]

                if open_rate == None or close_rate == None or open_rate_correct == None or \
                    close_rate_correct == None or open_rate_no_lag == None or close_rate_no_lag == None :
                    #どれかにNoneが入っている場合
                    money_history_org.append(money_org)
                    money_history_correct.append(money_correct)
                    money_history_no_lag.append(money_no_lag)
                else:
                    trade_cnt += 1
                    profit_org = float(Decimal(close_rate) - Decimal(open_rate))
                    profit_correct = float(Decimal(close_rate_correct) - Decimal(open_rate_correct))
                    profit_no_lag = float(Decimal(close_rate_no_lag) - Decimal(open_rate_no_lag))

                    if bet_type == "buy":
                        money_org += profit_org
                        money_correct += profit_correct
                        money_no_lag += profit_no_lag

                        up_org_correct_pips_list.append(profit_org - profit_correct)
                        up_correct_no_lag_pips_list.append(profit_correct - profit_no_lag)

                        up_open_org_correct_list.append(float(Decimal(open_rate) - Decimal(open_rate_correct)))
                        up_open_correct_no_lag_list.append(float(Decimal(open_rate_correct) - Decimal(open_rate_no_lag)))

                        up_close_org_correct_list.append(float(Decimal(close_rate) - Decimal(close_rate_correct)))
                        up_close_correct_no_lag_list.append(float(Decimal(close_rate_correct) - Decimal(close_rate_no_lag)))

                    elif bet_type == "sell":
                        money_org += profit_org * -1
                        money_correct += profit_correct * -1
                        money_no_lag += profit_no_lag * -1

                        dw_org_correct_pips_list.append(profit_org * -1 - profit_correct * -1)
                        dw_correct_no_lag_pips_list.append(profit_correct * -1 - profit_no_lag * -1)

                        dw_open_org_correct_list.append(float(Decimal(open_rate) - Decimal(open_rate_correct)))
                        dw_open_correct_no_lag_list.append(float(Decimal(open_rate_correct) - Decimal(open_rate_no_lag)))

                        dw_close_org_correct_list.append(float(Decimal(close_rate) - Decimal(close_rate_correct)))
                        dw_close_correct_no_lag_list.append(float(Decimal(close_rate_correct) - Decimal(close_rate_no_lag)))

                    money_history_org.append(money_org)
                    money_history_correct.append(money_correct)
                    money_history_no_lag.append(money_no_lag)

            except Exception as e :
                #値が取得できない場合
                    money_history_org.append(money_org)
                    money_history_correct.append(money_correct)
                    money_history_no_lag.append(money_no_lag)
        else:
            #取引履歴がない場合
            money_history_org.append(money_org)
            money_history_correct.append(money_correct)
            money_history_no_lag.append(money_no_lag)

    print("money_org:",money_org)
    print("money_correct:",money_correct)
    print("money_no_lag:",money_no_lag)

    print("1トレードあたりの本当の利益と実際の利益の差:",(money_no_lag - money_org)/trade_cnt)

    up_org_correct_pips_array = np.array(up_org_correct_pips_list)
    up_correct_no_lag_pips_array = np.array(up_correct_no_lag_pips_list)

    print("up_org_correct_pips_array avg:", '{:f}'.format(up_org_correct_pips_array.mean()))
    print("up_correct_no_lag_pips_array avg:", '{:f}'.format(up_correct_no_lag_pips_array.mean()))

    up_open_org_correct_array  = np.array(up_open_org_correct_list)
    up_open_correct_no_lag_array  = np.array(up_open_correct_no_lag_list)
    up_close_org_correct_array  = np.array(up_close_org_correct_list)
    up_close_correct_no_lag_array  = np.array(up_close_correct_no_lag_list)

    print("up_open_org_correct_array avg:", up_open_org_correct_array.mean())
    print("up_open_correct_no_lag_array avg:", up_open_correct_no_lag_array.mean())
    print("up_close_org_correct_array avg:", up_close_org_correct_array.mean())
    print("up_close_correct_no_lag_array avg:", up_close_correct_no_lag_array.mean())

    print("")

    dw_org_correct_pips_array = np.array(dw_org_correct_pips_list)
    dw_correct_no_lag_pips_array = np.array(dw_correct_no_lag_pips_list)

    print("dw_org_correct_pips_array avg:", '{:f}'.format(dw_org_correct_pips_array.mean()))
    print("dw_correct_no_lag_pips_array avg:", '{:f}'.format(dw_correct_no_lag_pips_array.mean()))

    dw_open_org_correct_array  = np.array(dw_open_org_correct_list)
    dw_open_correct_no_lag_array  = np.array(dw_open_correct_no_lag_list)
    dw_close_org_correct_array  = np.array(dw_close_org_correct_list)
    dw_close_correct_no_lag_array  = np.array(dw_close_correct_no_lag_list)

    print("dw_open_org_correct_array avg:", dw_open_org_correct_array.mean())
    print("dw_open_correct_no_lag_array avg:", dw_open_correct_no_lag_array.mean())
    print("dw_close_org_correct_array avg:", dw_close_org_correct_array.mean())
    print("dw_close_correct_no_lag_array avg:", dw_close_correct_no_lag_array.mean())

    fig, ax1 = plt.subplots()
    ax1.plot(close_history, "g-")
    ax1.set_ylabel("close")

    ax2 = ax1.twinx()

    ax2.plot(money_history_org, "r-")
    ax2.plot(money_history_correct, "b-")
    ax2.plot(money_history_no_lag, "y-")


    plt.show()