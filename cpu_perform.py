import numpy as np
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
import logging.config
import time

from decimal import Decimal
from readConf2 import *
from datetime import datetime
import time

##DBに残した各種処理時間の統計を計算する

#CPUのスレッド数を制限してロードアベレージの上昇によるハングアップを防ぐ
os.environ["OMP_NUM_THREADS"] = "3"

current_dir = os.path.dirname(__file__)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

host = "itl7"
db_no = 8
db_key = "GBPJPY_30_SPR"

start = datetime(2022, 1, 24, 0)
end = datetime(2022, 1, 25, 0)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))

if __name__ == "__main__":

    #start day of measure the cpu perform
    first_time = 0
    last_time = 0
    r = redis.Redis(host= host, port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(db_key, start_score, end_score, withscores=True)

    betTake_tmp, spreadTake_tmp, closeTake_tmp, predictTake_tmp = [], [], [], []
    betTake_num,spreadTake_num,closeTake_num,predictTake_num = 0, 0 , 0, 0

    for line in result:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)

        bet_tmp = 0
        if tmps.get("betTake") != None:
            bet_tmp = tmps.get("betTake")
            if first_time == 0:
                first_time = tmps.get("time")
            if bet_tmp !=0:
                betTake_tmp.append(bet_tmp)
                betTake_num = betTake_num +1

        spread_tmp = 0
        if tmps.get("spreadTake") != None:
            spread_tmp = tmps.get("spreadTake")
            if spread_tmp !=0:
                spreadTake_tmp.append(spread_tmp)
                spreadTake_num = spreadTake_num +1

        close_tmp = 0
        if tmps.get("closeTake") != None:
            close_tmp = tmps.get("closeTake")
            if close_tmp != 0:
                closeTake_tmp.append(close_tmp)
                closeTake_num = closeTake_num + 1

        predict_tmp = 0
        if tmps.get("predictTake") != None:
            predict_tmp = tmps.get("predictTake")
            if predict_tmp != 0:
                predictTake_num = predictTake_num + 1
                predictTake_tmp.append(predict_tmp)

        last_time = tmps.get("time")

    betTake_np = np.array(betTake_tmp)
    spreadTake_np = np.array(spreadTake_tmp)
    closeTake_np = np.array(closeTake_tmp)
    predictTake_np = np.array(predictTake_tmp)

    print("measure start day:", first_time)
    print("measure end day:", last_time)

    print("betTake Avg:", betTake_np.sum() / betTake_num if betTake_num !=0 else 0," Num:", betTake_num,)
    print("spreadTake Avg:", spreadTake_np.sum() / spreadTake_num if spreadTake_num !=0 else 0, " Num:", spreadTake_num )
    print("closeTake Avg:", closeTake_np.sum() / closeTake_num if closeTake_num !=0 else 0, " Num:", closeTake_num )
    print("predictTame Avg:", predictTake_np.sum() / predictTake_num if predictTake_num !=0 else 0, " Num:", predictTake_num, )

    print("betTake MAX:", np.max(betTake_np))
    print("spreadTake MAX:", np.max(spreadTake_np))
    print("closeTake MAX:", np.max(closeTake_np))
    print("predictTame MAX:", np.max(predictTake_np))

    # ヒストグラムを出力
    # グラフ作成の参考:
    # https://pythondatascience.plavox.info/matplotlib/%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0

    """
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("betTakes")
    ax1.hist(betTake_np,range=(10,100)) #rangeの下限を0にすると0の数が多すぎて平均の値付近の数がグラフで分からないため10に設定

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("spreadTakes")
    ax2.hist(spreadTake_np,range=(1,50))

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("closeTakes")
    ax3.hist(closeTake_np,range=(1,50))

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("predictTakes")
    ax4.hist(predictTake_np,range=(1,50))

    plt.show()
    """