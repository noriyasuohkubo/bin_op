import json
import numpy as np
import os
import redis
from datetime import datetime
import time

"""
dukaで作成したDBデータにもとづいて
移動平均などの各インジケータデータを作成する

"""

#作成する移動平均の長さリスト
#avg_dict = {"10":"","20":"","50":"","100":"","200":""}
avg_dict = {5:"",}
symbol = "GBPJPY"

import_db_no = 3
export_db_no = 3

export_host = "127.0.0.1"
import_host = "127.0.0.1"

start = datetime(2018, 5, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 5, 27)
end_stp = int(time.mktime(end.timetuple()))

export_r = redis.Redis(host=export_host, port=6379, db=export_db_no)
import_r = redis.Redis(host=import_host, port=6379, db=import_db_no)

"""
result_data = export_r.zrangebyscore(symbol, start_stp, end_stp, withscores=True)

close_tmp, time_tmp, score_tmp = [], [], []

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    score_tmp.append(score)
    # ask,bidの仲値
    close_tmp.append(tmps.get("close"))
    time_tmp.append(tmps.get("time"))

#numpyにする
close = np.array(close_tmp)
"""
close = np.full(10,5)
print("close.shape")
#print(close[:-10])

for key in avg_dict:
    avg_tmp = np.convolve(close, np.ones(key) / float(key), 'valid')
    print(avg_tmp.shape)
    print(avg_tmp)
    avg_dict[key] = avg_tmp


"""
for i, val in enumerate(close):
    for avg in avg_list:
        # データがたりなく移動平均作成できないばあいcontinue
        if i < avg -1:
            continue



    child = {'open': cand.open_price,
             'close': cand.close_price,
             'ask': cand.ask_price,
             'high': cand.high,
             'low': cand.low,
             'ask_volume': cand.ask_volume,
             'bid_volume': cand.bid_volume,
             'time': stringify(cand.timestamp)}

    imp = import_r.zrangebyscore(symbol + "_TRADE", score, score)
    if len(imp) == 0:
        import_r.save()
        import_r.zadd(symbol + "_TRADE", body, score)
"""