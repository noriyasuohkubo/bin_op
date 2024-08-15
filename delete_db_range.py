import numpy as np
import os
import redis
from datetime import datetime
import time
from decimal import Decimal
from util import  *
"""
db削除
"""
#Rename
#redis_db.rename("GBPJPY_2_0_OLD","GBPJPY_2_0")

db_no = 2
host = "127.0.0.1"

start_day = "2023/04/01 00:00:00"  # この時間含む(以上)
end_day = "2024/06/12 00:00:00"  # この時間含めない(未満)

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) - 1  # 含めないので1秒マイナス

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

bet_term = 2
terms = [2,10,60,300]
symbol = "USDJPY"

db_list = ["USDJPY_2_0_TICK"]
#db_list = []

for term in terms:
    db_list.extend(make_db_list(symbol, term, bet_term))
    """
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))

    else:
        db_list.append(symbol + "_" + str(term) + "_0")
    """
print(db_list)

for db in db_list:
    print(db)
    redis_db.zremrangebyscore(db,start_stp, end_stp)

