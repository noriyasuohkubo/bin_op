import random
import numpy as np
from decimal import Decimal
import subprocess
import redis
from datetime import datetime
import time
import json
import pandas as pd
import psutil
import gc
from util import *
import send_mail as mail
import socket

"""
make_fx_answerで作成したDBデータをpandasのpickleに保存する
"""

db_no = 2
db_names = ["USDJPY_5_0_ANSWER300_IFOAA:0.5-0.5-0.5",
            "USDJPY_5_0_ANSWER300_IFOAA:0.75-0.75-0.75",
            "USDJPY_5_0_ANSWER300_IFOAA:1-1-1",]

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

start_day = "2016/01/01 00:00:00" #この時間含む(以上)
end_day = "2022/05/01 00:00:00"  # この時間含めない(未満) 終了日は月から金としなけらばならない

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')
start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス
print(datetime.now())
print("start end:", str(start_stp) + " " + str(end_stp))


def make_pandas():
    for db_name in db_names:
        print(db_name)
        pickle_file_name = "/db2/answer/" + db_name + ".pickle"
        result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
        df_org = {}
        for data in result_data:
            body = json.loads(data[0])
            for k,v in body.items():
                if k in df_org.keys():
                    df_org[k].append(v)
                else:
                    df_org[k] = [v]

        df = pd.DataFrame(data=df_org)
        df = df.set_index("sc", drop=False)
        df = df.sort_index(ascending=True)# scoreで昇順 古い順にする
        df.to_pickle(pickle_file_name)

        #df2 = pd.read_pickle(pickle_file_name)
        #print(df2[:50].loc[:, ["sc","300-bp"]])


if __name__ == "__main__":
    make_pandas()