import json

import numpy as np
import os
import redis
from datetime import datetime
import time
import sys
from decimal import Decimal

"""
既存のDBのカラムを削除する
"""
bet_term = 1
terms = [5,30,180]
symbol = "USDJPY"

#削除したいカラム
delete_cols = ["eh", "el",]

start = datetime(2023, 4, 1,  )
start_stp = int(time.mktime(start.timetuple()))
end = datetime(2024, 2, 3, )
end_stp = int(time.mktime(end.timetuple()))

db_no = 2

host = "127.0.0.1"

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)


for term in terms:
    db_list = []
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))

    else:
        db_list.append(symbol + "_" + str(term) + "_0")

    for db in db_list:
        print(db)
        #redis_db.zremrangebyscore(db,1609459200, 1654041600)
        result_data = redis_db.zrangebyscore(db, start_stp, end_stp, withscores=True)
        print("result_data length:", len(result_data))
        cnt = 0
        for line in result_data:
            cnt += 1
            body = line[0]
            score = line[1]
            tmp = json.loads(body)

            # 削除したい値があれば削除する
            for col in delete_cols:
                if col in tmp.keys():
                    del tmp[col]

            rm_cnt = redis_db.zremrangebyscore(db, score, score)  # 削除した件数取得
            if rm_cnt != 1:
                # 削除できなかったらおかしいのでエラーとする
                print("cannot remove!!!", score)
                exit()

            redis_db.zadd(db, json.dumps(tmp), score)

            if cnt % 100000 == 0:
                print(datetime.now(), cnt)