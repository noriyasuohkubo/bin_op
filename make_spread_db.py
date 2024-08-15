import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
from util import *
import send_mail as mail
"""
dukasのデータをもとに
他のデータからスプレッド情報だけを取得して新たにレコード作成する
"""

db_list = ["USDJPY_2_0"]
db_tick_list = ["USDJPY_2_0_TICK"]
#spread情報をもつコピー元DB名
#db_name_sprad = "THREETRADER_USDJPY_S1"
#db_name_sprad = "VANTAGE_USDJPY.p_S1"
#db_name_sprad = "USDJPY_4_LION"
db_name_sprad = "USDJPY_60_MONEYPARTNERS"

db_no_org = 2
#db_no_sprad = 5
#db_no_sprad = 0
db_no_sprad = 8

term = 2 #データ間隔秒

default_spread = -1

host = "127.0.0.1"
#host_spread = "win2"
#host_spread = "win5"
#host_spread = "win6"
host_spread = "localhost"

#True:スプレッド取得元がthreetraderなどMetaTraderのようなopenデータの場合
#False:Lionなどの実トレードでのデータ
open_flg = False

#start = datetime.datetime(2023, 5, 1)
#start = datetime.datetime(2024, 2, 1)
#start = datetime.datetime(2024, 3, 1)
start = datetime.datetime(2024, 6, 12)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2024, 8, 10)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no_org, decode_responses=True)
redis_db_spread = redis.Redis(host=host_spread, port=6379, db=db_no_sprad, decode_responses=True)


for db_name in db_list:

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp = [], [], []

    for cnt, line in enumerate(result_data):
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        target_score = score if open_flg == False else get_decimal_add(score, term)
        tmp_hl = redis_db_spread.zrangebyscore(db_name_sprad, target_score , target_score )
        spread_tmp = default_spread  #spread情報が取れなければdefault_spread

        if len(tmp_hl) != 0:
            tmp_body = tmp_hl[0]
            tmp_val = json.loads(tmp_body)
            tmp_spr = tmp_val.get("spread")
            if tmp_spr != None:
                if float(tmp_spr) < 1:
                    #1以下は正しくスプレッドが登録されていないので10倍にする
                    spread_tmp = int(get_decimal_multi(tmp_val.get("spread"), 10))
                else:
                    spread_tmp = int(tmp_spr)

        tmps["s"] = spread_tmp

        rm_cnt = redis_db.zremrangebyscore(db_name, score, score)  # 削除した件数取得
        if rm_cnt != 1:
            # 削除できなかったらおかしいのでエラーとする
            print("cannot remove!!!", score)
            exit()

        redis_db.zadd(db_name, json.dumps(tmps), score)

        if cnt % 1000000 == 0:
            dt_now = datetime.datetime.now()
            print(dt_now, " ", cnt)

for db_name in db_tick_list:

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp = [], [], []

    for cnt, line in enumerate(result_data):
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        target_score = score if open_flg == False else get_decimal_add(score, term)
        tmp_hl = redis_db_spread.zrangebyscore(db_name_sprad, target_score , target_score )
        spread_tmp = default_spread  #spread情報が取れなければdefault_spread

        if len(tmp_hl) != 0:
            tmp_body = tmp_hl[0]
            tmp_val = json.loads(tmp_body)
            tmp_spr = tmp_val.get("spread")
            if tmp_spr != None:
                if float(tmp_spr) < 1:
                    #1以下は正しくスプレッドが登録されていないので10倍にする
                    spread_tmp = int(get_decimal_multi(tmp_val.get("spread"), 10))
                else:
                    spread_tmp = int(tmp_spr)

        tmps["s"] = spread_tmp

        tk_list = tmps.get("tk").split(",")

        new_tk_str = ""
        for tmp_tk in tk_list:
            tmp_close, tmp_spread = tmp_tk.split(":")
            if new_tk_str == "":
                new_tk_str = tmp_close + ":" + str(spread_tmp)
            else:
                new_tk_str = new_tk_str + "," + tmp_close + ":" + str(spread_tmp)

        tmps["tk"] = new_tk_str

        rm_cnt = redis_db.zremrangebyscore(db_name, score, score)  # 削除した件数取得
        if rm_cnt != 1:
            # 削除できなかったらおかしいのでエラーとする
            print("cannot remove!!!", score)
            exit()

        redis_db.zadd(db_name, json.dumps(tmps), score)

        if cnt % 1000000 == 0:
            dt_now = datetime.datetime.now()
            print(dt_now, " ", cnt)

print("FINISH")
# 終わったらメールで知らせる
mail.send_message(host, ": make_spread_db finished!!!")