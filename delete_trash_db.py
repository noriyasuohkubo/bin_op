import numpy as np
import os
import redis
import datetime
import time
from decimal import Decimal

"""
db削除
不要な短いtermのDBを削除する
"""
# Rename
# redis_db.rename("GBPJPY_2_0_OLD","GBPJPY_2_0")

db_no = 3
host = "127.0.0.1"

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

bet_term = 15
terms = [15, 60, 300]
symbol = "USDJPY"

delete_term = 5

for term in terms:
    db_list = []
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))

    else:
        db_list.append(symbol + "_" + str(term) + "_0")

    delete_db_list = []
    if term >= delete_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(delete_term)))):
            delete_db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * delete_term)))

    else:
        delete_db_list.append(symbol + "_" + str(term) + "_0")

    for delete_db in delete_db_list:
        if (delete_db in db_list) == False:
            print(delete_db)
            redis_db.delete(delete_db)

