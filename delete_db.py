import numpy as np
import os
import redis
import datetime
import time

"""
dukaで作成したDBデータにもとづいて
close,open,timeのみ抽出する(DB軽量化のため)
"""



db_no = 1
host = "127.0.0.1"


redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
redis_db.flushdb() #"Delete all keys in the current database"