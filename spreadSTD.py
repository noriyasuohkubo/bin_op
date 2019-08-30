import numpy as np
import redis
import json
from scipy.ndimage.interpolation import shift
import logging.config
from datetime import datetime
from datetime import timedelta
import time
from indices import index
from decimal import Decimal
from readConf import *
from matplotlib import pyplot as plt
import pandas as pd

start = datetime(2013, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2019, 1, 1)
end_stp = int(time.mktime(end.timetuple()))

except_index = False
except_highlow = True

np.random.seed(0)

in_num=1

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

def get_redis_data(sym):
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(sym, start_stp, end_stp, withscores=False)
    #result = r.zrevrange(symbol, 0  , rec_num  , withscores=False)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    spread_tmp = []

    not_except_index_cnts = []
    print(result[0:5])

    for line in result:
        tmps = json.loads(line)

        mid = float((Decimal(str(tmps.get("close"))) + Decimal(str(tmps.get("ask")))) / Decimal("2"))
        close_tmp.append(mid)
        spread_tmp.append(float(Decimal(str(tmps.get("ask"))) - Decimal(str(mid))))

        time_tmp.append(tmps.get("time"))

    #close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]

    close_data, high_data, low_data, label_data, time_data, price_data, end_price_data, close_abs_data = [], [], [], [], [], [], [], []
    spread_data = []
    not_except_data = []
    up =0
    same =0
    data_length = len(spread_tmp) -1 - maxlen - pred_term -1

    for i in range(data_length):
        #maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
        tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
        delta =tmp_time_aft - tmp_time_bef

        if delta.total_seconds() > ((maxlen-1) * int(s)):
            #print(tmp_time_aft)
            continue

        #ハイローオーストラリアの取引時間外を学習対象からはずす
        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue

        spread_data.append(spread_tmp[1 + i + maxlen -1])

    spread_np = np.array(spread_data)

    return spread_np


if __name__ == "__main__":
    for sym in symbols:
        spread_np = get_redis_data(sym)

    df = pd.DataFrame(pd.Series(spread_np.ravel()).describe()).transpose()
    print(df)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(spread_np, 'g')
    plt.show()
