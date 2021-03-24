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
import matplotlib.pyplot as plt
from scipy import stats

"""
変化率を取得して、平均、標準偏差、中央値を求めヒストグラムを描写

"""

start_day = "2009/01/01 00:00:00" #この時間含む(以上)
end_day = "2020/01/01 00:00:00"  # この時間含めない(未満)

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

math_log = False

db_no = 3
#取得元DB
db_name = "GBPJPY_2_0"

#予測間隔
pred_term = 1

bet_term = 2

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

def get():
    # 処理時間計測
    t1 = time.time()

    result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
    print("result_data length:" + str(len(result_data)))

    d_list =[]
    c_list = []
    s_list = []

    for line in result_data:
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)

        c_list.append(float(tmps.get("c")))
        s_list.append(score)

    for i, c in enumerate(c_list):
        if i < pred_term:
            continue

        try:
            start_score = s_list[i - pred_term]
            end_score = s_list[i]
            if end_score != start_score + (pred_term * bet_term):
                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                continue

        except IndexError:
            # start_scoreのデータなしなのでスキップ
            continue

        divide = float(c_list[i]) / float(c_list[i - pred_term])
        if c_list[i] == c_list[i - pred_term]:
            divide = 1
        if math_log:
            divide = 10000 * math.log(divide)
        else:
            divide = 10000 * (divide - 1)

        d_list.append(divide)

    d_np =np.array(d_list)
    d_np = np.sort(d_np)

    print("data length:", len(d_np))
    print("avg:", np.average(d_np))
    print("std:", np.std(d_np))
    print("mid:", d_np[int(len(d_np)/2) -1]) #中央値
    print("max:", np.max(d_np))
    print("min:", np.min(d_np))

    mean = np.average(d_np)
    se = np.std(d_np)
    Degree_of_freedom = len(d_np) - 1

    #信頼区間
    alphas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999)

    for alpha in alphas:
        bottom, up = stats.t.interval(alpha=alpha, loc=mean, scale=se, df=Degree_of_freedom)
        print(str(alpha), "bottom:", bottom, "up:", up)
        #print(np.where((d_np >= bottom) & (d_np <= up)))
        print("num:", len(np.where((d_np >= bottom) & (d_np <= up))[0]))

    t2 = time.time()
    elapsed_time = t2-t1
    #print("経過時間：" + str(elapsed_time))

    #plt.hist(d_np, range=(-10, 10), bins=20)
    #plt.show()

if __name__ == "__main__":
    get()


