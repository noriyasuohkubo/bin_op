import numpy as np
import os
import redis
import datetime
import time
from decimal import Decimal
from util import *

"""
db削除
"""
#Rename
#redis_db.rename("GBPJPY_2_0_OLD","GBPJPY_2_0")

db_no = 2
host = "127.0.0.1"

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

bet_term = 1
terms = [5,30]
symbol = "USDJPY"

for term in terms:

    db_list = make_db_list(symbol,term, bet_term)
    """
    db_list = []
    #db_list = ["USDJPY_5_0","USDJPY_5_IND300","USDJPY_5_0_TICK",]
    #db_list = ["USDJPY_5_0","USDJPY_5_IND300",]

    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))

    else:
        db_list.append(symbol + "_" + str(term) + "_0")
    """

    for db in db_list:
        print(db)
        #redis_db.zremrangebyscore(db,1709424000, 1714521600)
        redis_db.delete(db)
