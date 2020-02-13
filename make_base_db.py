import json
import numpy as np
import os
import redis
import datetime
import time

"""
dukaで作成したDBデータにもとづいて
close,timeのみ抽出する(DB軽量化のため)
"""

# 処理時間計測
t1 = time.time()

#抽出元のDB名
symbol_org = "GBPJPY"

#新規に作成するDB名
symbol = "GBPJPY_BASE"

db_no = 3
host = "127.0.0.1"

start = datetime.datetime(2009, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2020, 1, 1)
end_stp = int(time.mktime(end.timetuple()))

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
# print("result_data length:" + str(len(result_data)))
for i, v in enumerate(result_data):
    body = v[0]
    score = v[1]
    tmps = json.loads(body)

    child = {'close': tmps.get("close"),
             'time': tmps.get("time")}
    redis_db.zadd(symbol, json.dumps(child), score)

    if i % 10000000 == 0:
        dt_now = datetime.datetime.now()
        print(dt_now, " ", i)


"""
#pipeline処理はむしろ何故か遅い
while start_stp < end_stp:
    tmp_end = start + datetime.timedelta(days=1)
    tmp_end_stp = int(time.mktime(tmp_end.timetuple()))
    result_data = redis_db.zrangebyscore(symbol_org, start_stp, tmp_end_stp, withscores=True)
    #print("result_data length:" + str(len(result_data)))
    for line in result_data:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        child = {'close': tmps.get("close"),
                 'time': tmps.get("time")}
        redis_db.zadd(symbol, child, score)

    with redis_db.pipeline() as pipe:
        try:
            for line in result_data:
                body = line[0]
                score = line[1]
                tmps = json.loads(body)

                child = {'close': tmps.get("close"),
                         'time': tmps.get("time")}
                redis_db.zadd(symbol, child, score)
            pipe.execute()
        except:
            print("db regist  error occured:");
        else:
            print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), start.strftime("%Y/%m/%d "), "OK")
        finally:
            pipe.reset()

    start = start + datetime.timedelta(days=1)
    start_stp = int(time.mktime(start.timetuple()))
"""
t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))

#手動にて抽出元のDBを削除して永続化すること！
#print("now db saving")
#redis_db.save()
