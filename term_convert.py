import socket
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
from util import  get_decimal_add, get_decimal_sub,get_decimal_multi,get_decimal_divide,get_decimal_mod, omit_zero_point_str
import  send_mail as mail

"""
make_close_dbで作成した2秒間隔のレコードを任意の間隔に拡張しなおす。
例)秒から1分

変更間隔が1分なら00:00:00のレコードの場合00:00:55のレコードの終値を登録することになる

"""
start_day = "2024/02/01 00:00:00" #この時間含む(以上)
end_day = "2024/08/10 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない


start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

#開始日時が終了日時より前であるかチェック
if start_day_dt >= end_day_dt:

    print("Error:開始日時が終了日時より前です！！！")
    exit()

#変更する間隔(sec)
terms = [2]


#DBのもとのレコード秒間隔
org_term = 1

bet_term = 2

highlow_flg = False

ask_bid_flg = False

#closeからhigh lowを求める
easy_highlow_flg = False

div_flg = True

#平均を取得する
#直近のデータから数えて幾つ分を計算するかリストに保持 すべての平均を取る場合は0
means = []
#means = [0]

#加重平均を取得する
wmeans = []
#wmeans = [0]

#volumeの値をdbレコードに含める
volume_flg = False

#変化率のlogをとる
math_log = False

#closeの値をdbレコードに含める
close_flg = True

#spreadの値をdbレコードに含める
spread_flg = True

tick_flg = False

jpy_flg = False

sp_flg = False

db_no_old = 2
db_no_new = 2

#取得元DB
symbol = "USDJPY"
db_name_old = symbol
#db_name_old = symbol + "_1_0"
#db_name_old = symbol + "_2_0_TICK"

redis_db_old = redis.Redis(host='localhost', port=6379, db=db_no_old, decode_responses=True)
redis_db_new = redis.Redis(host='localhost', port=6379, db=db_no_new, decode_responses=True)


def get_wma(value):
    # 加重移動平均
    weight = np.arange(len(value)) + 1
    wma = np.sum(weight * value) / weight.sum()

    return wma

def get_divide(bef, aft):
    divide = aft / bef
    if aft == bef:
        divide = 1

    if math_log:
        divide = 10000 * math.log(divide, math.e * 0.1)
    else:
        divide = 10000 * (divide - 1)

    return divide

def convert():
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db_old.zrangebyscore(db_name_old, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    #key:score val:dict(cs,cm,etc)
    lists = []
    score_index = {} #scoreとlistsでのindexのひもづけ

    for i, line in enumerate(result_data):
        body = line[0]
        score = float(line[1])
        tmps = json.loads(body)

        #t = tmps.get("t")
        # 秒に変換
        #hour = int(t[11:13]) * 3600
        #min = int(t[14:16]) * 60
        #sec = int(t[17:19])
        #t_val = hour + min + sec

        val_dict = {}
        val_dict["c"] = float(tmps.get("c"))
        if ask_bid_flg:
            val_dict["ac"] = tmps.get("ac")
            val_dict["bc"] = tmps.get("bc")
        if spread_flg == True:
            val_dict["s"] = tmps.get("s")
        if volume_flg == True:
            val_dict["v"] = tmps.get("v")
        if highlow_flg == True:
            val_dict["h"] = tmps.get("high")
            val_dict["l"] = tmps.get("low")

            if ask_bid_flg:
                val_dict["ah"] = tmps.get("ah")
                val_dict["al"] = tmps.get("al")

                val_dict["bh"] = tmps.get("bh")
                val_dict["bl"] = tmps.get("bl")

        if tick_flg:
            val_dict["tk"] = tmps.get("tk")

        if jpy_flg == True:
            val_dict["jpy"] = tmps.get("jpy")

        if sp_flg == True:
            val_dict["sp"] = tmps.get("sp")

        #val_dict["t"] = tmps.get("t")
        val_dict["idx"] = i
        val_dict["sc"] = score

        lists.append(val_dict)
        score_index[score] = i

    del result_data

    for term in terms:
        # 1分間隔でつくるとして15なら00:01:15 00:02:15 00:03:15と位相をずらす
        shift_list = []
        for i in range(int(get_decimal_divide(term, bet_term))):
            shift_list.append(get_decimal_sub(term, get_decimal_multi((i + 1), bet_term)))

        print(shift_list)

        for shift in shift_list:
            cnt = 0
            #shiftが.0の場合は小数点を省く
            db_name_new = symbol + "_" + str(term) + "_" + omit_zero_point_str(shift)

            if tick_flg:
                db_name_new = db_name_new + "_TICK"

            print("shift:", shift)

            #for score, val in lists.items():
            for j, val in enumerate(lists):
                cnt += 1
                #t_val = val["t_val"]
                score = val["sc"]
                #間隔に当たっていたら2秒まえのレコードを取得
                # cdを計算するため
                #if t_val % term == shift:
                if get_decimal_mod(score, term) == shift:
                    #自分より後ろの必要なデータの長さ
                    #need_len = int(Decimal(str(term)) / Decimal(str(org_term))) - 1
                    need_len = int(get_decimal_divide(term, org_term)) - 1
                    #2秒前のレコード
                    if get_decimal_sub(score, org_term) in score_index:
                        tmp_idx = score_index[get_decimal_sub(score, org_term)]
                        prev = lists[tmp_idx]
                    else:
                        continue
                        #なければとばす

                    if get_decimal_add(score, get_decimal_sub(term, org_term)) in score_index:
                        tmp_idx = score_index[get_decimal_add(score, get_decimal_sub(term, org_term))]
                        after = lists[tmp_idx]
                        start_idx = val["idx"]
                        end_idx = after["idx"]

                        if end_idx - start_idx !=  need_len:
                            #データが続いていなければとばす
                            continue

                        child = {#'c': after["c"],
                                 'sc': score
                                 }

                        if div_flg == True:
                            child["d1"] = get_divide(prev["c"], after["c"])
                            if ask_bid_flg:
                                child["ad-1"] = get_divide(prev["ac"], after["ac"])
                                child["bd-1"] = get_divide(prev["bc"], after["bc"])

                        if spread_flg == True:
                            #もしスプレッドがNoneなら0をいれておく(dukaデータ用)
                            child["s"] = after.get("s") if after.get("s") != None else 0

                        if close_flg == True:
                            child["c"] = after["c"]
                            if ask_bid_flg:
                                child["ac"] = after["ac"]
                                child["bc"] = after["bc"]

                        if jpy_flg == True:
                            child["jpy"] = after["jpy"]

                        if sp_flg == True:
                            child["sp"] = after["sp"]

                        if volume_flg or highlow_flg or tick_flg or len(means) != 0 or len(wmeans) != 0 or easy_highlow_flg:
                            #後ろのデータを集める
                            aft_data = lists[start_idx:end_idx+1]
                            #for j in range(need_len + 1):
                            #    aft_data.append(lists[score + (j * org_term)])
                            c_list = []
                            for aft in aft_data:
                                c_list.append(aft["c"])

                            if highlow_flg:
                                high_list = []
                                low_list = []

                                ask_high_list = []
                                ask_low_list = []
                                bid_high_list = []
                                bid_low_list = []

                                for aft in aft_data:
                                    high_list.append(aft["h"])
                                    low_list.append(aft["l"])

                                    if ask_bid_flg:
                                        ask_high_list.append(aft["ah"])
                                        ask_low_list.append(aft["al"])
                                        bid_high_list.append(aft["bh"])
                                        bid_low_list.append(aft["bl"])

                                child["h"] = max(high_list)
                                child["l"] = min(low_list)

                                if ask_bid_flg:
                                    child["ah"] = max(ask_high_list)
                                    child["al"] = min(ask_low_list)
                                    child["bh"] = max(bid_high_list)
                                    child["bl"] = min(bid_low_list)

                            if len(means) != 0:
                                for m in means:
                                    if m == 0:
                                        child["m"] = np.mean(np.array(c_list))
                                    else:
                                        child["m" + str(m)] = np.mean(np.array(c_list[(-1 * m) : ]))

                            if len(wmeans) != 0:
                                for m in wmeans:
                                    if m == 0:
                                        child["wm"] = get_wma(np.array(c_list))
                                    else:
                                        child["wm" + str(m)] = get_wma(np.array(c_list[(-1 * m) : ]))


                            if easy_highlow_flg:
                                child["eh"] = np.max(np.array(c_list))
                                child["el"] = np.min(np.array(c_list))

                            if volume_flg:
                                v_sum = 0
                                for aft in aft_data:
                                    v_sum += int(aft["v"])
                                child["v"] = v_sum

                            if tick_flg:
                                tmp_tk = []
                                # 自分よりあとのデータを集めていく
                                for aft in aft_data:
                                    if aft.get("tk") != None:
                                        tmp_tk.extend(aft.get("tk").split(","))

                                tmp_tk_str = ""
                                for i, t in enumerate(tmp_tk):
                                    if tmp_tk_str == "":
                                        tmp_tk_str = t
                                    else:
                                        if t != tmp_tk[i - 1]:  # 前のティックと異なる場合のみ登録　メモリ節約
                                            tmp_tk_str = tmp_tk_str + "," + t

                                #tkデータがない場合(dukaデータ用)
                                if tmp_tk_str == "":
                                    child["tk"] = str(child["c"]) + ":0"
                                else:
                                    child["tk"] = tmp_tk_str

                        """
                        #既存レコードがあるばあい、削除して追加
                        tmp_val = redis_db_new.zrangebyscore(db_name_new, score, score)
                        if len(tmp_val) >= 1:
                            rm_cnt = redis_db_new.zremrangebyscore(db_name_new, score, score)  # 削除した件数取得
                            if rm_cnt != 1:
                                # 削除できなかったらおかしいのでエラーとする
                                print("cannot remove!!!", score)
                                exit()
                        """
                        ret = redis_db_new.zadd(db_name_new, json.dumps(child), score)

                    else:
                        continue

                if cnt % 10000000 == 0:
                    dt_now = datetime.now()
                    print(dt_now, " ", cnt)

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    convert()

    #redis_db_new.save()

    # 終わったらメールで知らせる
    mail.send_message(socket.gethostname(), ": term_convert finished!!!")