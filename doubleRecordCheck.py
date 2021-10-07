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


"""
重複するレコードを確認

"""

start_day = "2021/01/01 00:00:00" #この時間含む(以上)
end_day = "2021/09/30 22:00:00"  # この時間含む(以下)

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple()))

#変化率のlogをとる
math_log = False

db_no = 8

#取得元DB
db_name = "GBPJPY_30_SPR"

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)


def check():

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    close_tmp, time_tmp, score_tmp, spread_tmp = [], [], [], []

    for line in result_data:
        body = line[0]
        score = int(line[1])

        tmp_val = redis_db.zrangebyscore(db_name, score, score)
        if len(tmp_val) >= 2:
            #レコードが重複している場合
            print(body)
            print(score)

        #redis_db_new.zremrangebyscore()

if __name__ == "__main__":
    check()


