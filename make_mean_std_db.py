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


"""
1秒データをもとに指定秒数間での平均レートとその変化率および
標準偏差と平均レートとの変化率と終値などを保持する

"""

start_day = "2004/01/01 00:00:00" #この時間含む(以上)
end_day = "2012/01/01 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

#開始日時が終了日時より前であるかチェック
if start_day_dt >= end_day_dt:

    print("Error:開始日時が終了日時より前です！！！")
    exit()

#変更する間隔(sec)
terms = [300, ]


#term以外の秒数でデータ集計する場合
input_lengths = []

#DBのもとのレコード秒間隔
org_term = 1

bet_term = 5

#何秒後の予想をするか
pred_term = 1800

#正解を入れる
answer_flg = False

#closeの値をdbレコードに含める
close_flg = True

#集計情報(md,msdなど)を含める
sum_flg = False

#high lowの値を含める
highlow_flg = True

#spreadの値をdbレコードに含める
spread_flg = False

#tkの値のみのDBを作成する
tick_flg = False

db_no_old = 3
db_no_new = 3

#取得元DB
symbol = "USDJPY"
db_name_old = "USDJPY"

redis_db_old = redis.Redis(host='ub3', port=6379, db=db_no_old, decode_responses=True)
redis_db_new = redis.Redis(host='localhost', port=6379, db=db_no_new, decode_responses=True)

def get_divide(bef, aft):
    divide = aft / bef

    if aft == bef:
        divide = 1

    divide = 10000 * (divide - 1)

    return divide

def get_round(target):
    return round(target, 10)

def check_data_chain(dicts, score, bef_term, aft_term):
    ok_flg = True
    # 時刻がつながっていないものは除外 たとえば日付またぎなど
    try:
        bef_start = dicts[score - bef_term]
        bef_start_idx = bef_start["idx"]
        aft_end = dicts[score + aft_term - 1]
        aft_end_idx = aft_end["idx"]
        if aft_end_idx - bef_start_idx != bef_term + aft_term - 1:
            ok_flg = False
    except Exception:
        # bef_start aft_endがないのでスキップ
        ok_flg = False

    return ok_flg

def get_bef_data(dicts, score, term):
    bef_data = []
    # 自分より前のデータを集めていく
    for j in range(term):
        tmp_data = dicts[score - j - 1]
        bef_data.append(tmp_data)

    bef_data.reverse()  # 古いデータ順に並び替え

    return  bef_data

def get_aft_data(dicts, score, term, ipt_len):
    aft_data = []
    # 自分よりあとのデータを集めていく
    for j in range(ipt_len):
        tmp_data = dicts[score + j + (term - ipt_len)]
        aft_data.append(tmp_data)

    return  aft_data

def get_close(list):
    tmp_closes = []
    for dict in list:
        tmp_closes.append(dict["close"])

    return tmp_closes

def get_highlow(list):
    tmp_highs = []
    tmp_lows = []
    for dict in list:
        tmp_highs.append(dict["high"])
        tmp_lows.append(dict["low"])

    return max(tmp_highs), min(tmp_lows)

def convert():
    np.set_printoptions(suppress=True)
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    dicts = {}
    for i, line in enumerate(result_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        val_dict = {}
        val_dict["close"] = tmps.get("close")
        if spread_flg == True:
            val_dict["spr"] = tmps.get("spr")
        if tick_flg == True:
            val_dict["tk"] = tmps.get("tk")
        if highlow_flg:
            val_dict["high"] = tmps.get("high")
            val_dict["low"] = tmps.get("low")

        val_dict["idx"] = i

        dicts[score] = val_dict

    del result_data

    for term in terms:
        # 1分間隔でつくるとして15なら00:01:15 00:02:15 00:03:15と位相をずらす
        shift_list = []
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            shift_list.append(term - ((i + 1) * bet_term))

        print(shift_list)
        if len(input_lengths) == 0:
            local_input_lengths = [term]
        else:
            local_input_lengths = input_lengths

        for shift in shift_list:
            cnt = 0
            if tick_flg:
                db_name_new = symbol + "_" + str(term) + "_" + str(shift) + "_TICK"
            else:
                db_name_new = symbol + "_" + str(term) + "_" + str(shift)
            print("shift:", shift)

            for score, val in dicts.items():
                cnt += 1

                #間隔に当たっていたら1秒まえのレコードを取得
                if score % term == shift:
                    child = {}
                    if check_data_chain(dicts, score, term, term) == False:
                        continue

                    # 自分より前のデータを集めていく
                    bef_data = get_bef_data(dicts, score, term)
                    bef_close = bef_data[-1]["close"]

                    aft_data = []
                    tmp_spr = 0
                    tmp_tk = []

                    # 自分よりあとのデータを集めていく
                    for j in range(term):
                        tmp_data = dicts[score + j]
                        aft_data.append(tmp_data)
                        if tick_flg:
                            tick_spr_list = tmp_data["tk"].split(",")
                            tmp_tk.extend(tick_spr_list)
                        if spread_flg:
                            tmp_spr = tmp_data["spr"]  # 古い順に入れていくので、一番最後のsprが登録すべきspr

                    aft_close = aft_data[-1]["close"]
                    close_div = get_divide(bef_close, aft_close)

                    #aft_mean_close_div = get_divide(aft_mean, aft_close)
                    child["d"] = close_div

                    if close_flg:
                        child["c"] = aft_close
                    if spread_flg:
                        child["s"] = tmp_spr
                    if tick_flg:
                        tmp_tk_str = ""
                        for i, t in enumerate(tmp_tk):
                            if tmp_tk_str == "":
                                tmp_tk_str = t
                            else:
                                if t != tmp_tk[i - 1]:  # 前のティックと異なる場合のみ登録　メモリ節約
                                    tmp_tk_str = tmp_tk_str + "," + t
                        # tick情報だけ登録
                        child = {}
                        child["tk"] = tmp_tk_str

                    if highlow_flg:
                        aft_high, aft_low = get_highlow(aft_data)
                        child["h"] = aft_high
                        child["l"] = aft_low

                    child["time"] = datetime.fromtimestamp(score).strftime("%Y-%m-%d %H:%M:%S")

                    if sum_flg:
                        continue_flg = False
                        for ipt_len in local_input_lengths:
                            if check_data_chain(dicts, score, ipt_len, term) == False:
                                continue_flg = True
                                break
                            bef_data = get_bef_data(dicts, score, ipt_len)
                            bef_data_close = get_close(bef_data)

                            bef_data_np = np.array(bef_data_close)
                            bef_mean = np.mean(bef_data_np)
                            bef_mid = np.median(bef_data_np)  # 中央値

                            # 自分よりあとのデータを集めていく
                            aft_data = get_aft_data(dicts, score, term, ipt_len)
                            aft_data_close = get_close(aft_data)

                            aft_close = aft_data_close[-1]
                            aft_data_np = np.array(aft_data_close)
                            aft_mean = np.mean(aft_data_np)
                            aft_std = np.std(aft_data_np)
                            aft_std_rate = aft_mean + aft_std
                            aft_mean_std_div = get_divide(aft_mean, aft_std_rate) #平均とシグマ1との変化率を取得

                            mean_div = get_divide(bef_mean, aft_mean)
                            aft_mean_close_div = get_divide(aft_mean, aft_close)


                            aft_mid = np.median(aft_data_np) #中央値
                            Mid_div = get_divide(bef_mid, aft_mid)

                            prefix = "" if term == ipt_len else str(ipt_len)
                            child[prefix + "md"] = mean_div
                            child[prefix + "msd"] = aft_mean_std_div
                            child[prefix + "mcd"] = aft_mean_close_div
                            child[prefix + "Md"] = Mid_div

                            if highlow_flg:
                                bef_high, bef_low = get_highlow(bef_data)
                                aft_high, aft_low = get_highlow(aft_data)

                                #child[prefix + "hd"] = get_divide(bef_high, aft_high)
                                #child[prefix + "ld"] = get_divide(bef_low, aft_low)
                                #child[prefix + "mhd"] = get_divide(aft_mean, aft_high)
                                #child[prefix + "mld"] = get_divide(aft_mean, aft_low)
                                child[prefix + "hld"] = get_divide(aft_low, aft_high)
                                child[prefix + "chd"] = get_divide(aft_close, aft_high)
                                child[prefix + "cld"] = get_divide(aft_close, aft_low)
                                #child[prefix + "h"] = aft_high
                                #child[prefix + "l"] = aft_low

                        if continue_flg:
                            continue

                    if answer_flg:

                        if check_data_chain(dicts, score, pred_term, pred_term) == False:
                            continue

                        bef_data = get_bef_data(dicts, score, pred_term)
                        bef_data_close = get_close(bef_data)
                        bef_data_np = np.array(bef_data_close)
                        bef_mean = np.mean(bef_data_np)
                        bef_std = np.std(bef_data_np)
                        bef_med = np.median(bef_data_np)

                        # 自分よりあとのデータを集めていく
                        aft_data = get_aft_data(dicts, score, pred_term, pred_term)
                        aft_data_close = get_close(aft_data)
                        aft_data_np = np.array(aft_data_close)
                        aft_mean = np.mean(aft_data_np)
                        aft_std = np.std(aft_data_np)
                        aft_med = np.median(aft_data_np)
                        #予想時の平均と偏差および、正解の平均と偏差をカンマ区切りで入れる

                        child["bm"] = bef_mean
                        child["bs"] = bef_std
                        child["am"] = aft_mean
                        child["as"] = aft_std
                        child["bM"] = bef_med
                        child["aM"] = aft_med

                        """
                        if highlow_flg:
                            bef_high, bef_low = get_highlow(bef_data)
                            aft_high, aft_low = get_highlow(aft_data)
                            child[str(pred_term) + "bh"] = bef_high
                            child[str(pred_term) + "bl"] = bef_low
                            child[str(pred_term) + "ah"] = aft_high
                            child[str(pred_term) + "al"] = aft_low
                        """

                    """
                    tmp_val = redis_db_new.zrangebyscore(db_name_new, score, score)
                    if len(tmp_val) == 0:
                        # レコードなければ登録
                        ret = redis_db_new.zadd(db_name_new, json.dumps(child), score)
                        # もし登録できなかった場合
                        if ret == 0:
                            print(child)
                            print(score)
                    """
                    ret = redis_db_new.zadd(db_name_new, json.dumps(child), score)

                if cnt % 10000000 == 0:
                    dt_now = datetime.now()
                    print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

