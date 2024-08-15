import numpy as np
import os
import redis
from datetime import datetime
import time
import sys
from decimal import Decimal

bet_term = 5
terms = [300]
symbol = "USDJPY"

in_db_no = 4
out_db_no = 3
host = "127.0.0.1"

redis_db_in = redis.Redis(host=host, port=6379, db=in_db_no, decode_responses=True)
redis_db_out = redis.Redis(host=host, port=6379, db=out_db_no, decode_responses=True)

for term in terms:
    db_list = []
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))

    else:
        db_list.append(symbol + "_" + str(term) + "_0")

    for db in db_list:
        print(db)
        #redis_db_out.flushdb()
        redis_db_in.move(db, out_db_no)
