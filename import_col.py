import json
import socket

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
他のデータから指定した値を取り込む
"""
output_log_name = "/home/reicou/import_col.txt"
output = output_log(output_log_name,print_flg=True)

start = datetime.datetime(2016, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2024, 12, 1)
end_stp = int(time.mktime(end.timetuple()))

db_no_export = 3
host_export = "127.0.0.1"

db_no_import = 3
host_import = "127.0.0.1"

symbol_export = "EURJPY"
col_export_list = ["d1"]

symbol_import = "USDJPY"

delete_cols = ["EURJPY_d1"]

bet_term = 1
terms = [3]

redis_db_export = redis.Redis(host=host_export, port=6379, db=db_no_export, decode_responses=True)
redis_db_import = redis.Redis(host=host_import, port=6379, db=db_no_import, decode_responses=True)

for term in terms:

    db_list_export = make_db_list(symbol_export,term, bet_term)
    db_list_import = make_db_list(symbol_import,term, bet_term)

    for exp_db_name, imp_db_name in zip(db_list_export,db_list_import):

        print("db_name:", imp_db_name)

        result_data = redis_db_import.zrangebyscore(imp_db_name, start_stp, end_stp, withscores=True)
        print("result_data length:" + str(len(result_data)))

        for cnt, line in enumerate(result_data):
            body = line[0]
            score = line[1]
            tmps = json.loads(body)

            result_data_exp = redis_db_export.zrangebyscore(exp_db_name, score, score, withscores=True)
            if len(result_data_exp) == 0:
                output("target_col data zero:", score, exp_db_name)
                continue

            tmps_exp = json.loads(result_data_exp[0][0])

            for col in col_export_list:
                target_col = tmps_exp.get(col)
                if target_col == None:
                    output("target_col is None:",score,col,exp_db_name)
                    continue

                col_name = symbol_export + "~" + col

                tmps[col_name] = float(target_col)

            # 削除したい値があれば削除する
            for col in delete_cols:
                if col in tmps.keys():
                    del tmps[col]

            rm_cnt = redis_db_import.zremrangebyscore(imp_db_name, score, score)  # 削除した件数取得
            if rm_cnt != 1:
                # 削除できなかったらおかしいのでエラーとする
                print("cannot remove!!!", score)
                exit()

            redis_db_import.zadd(imp_db_name, json.dumps(tmps), score)

            if cnt % 1000000 == 0:
                dt_now = datetime.datetime.now()
                print(dt_now, " ", cnt)

print("FINISH")
# 終わったらメールで知らせる
mail.send_message(socket.gethostname(), ": import_col finished!!!")