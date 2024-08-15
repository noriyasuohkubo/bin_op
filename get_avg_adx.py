import json
import numpy as np
import redis
from datetime import datetime
import time
from decimal import Decimal
from util import *

db_no = 2
host = "127.0.0.1"

start_day = "2022/11/01 00:00:00" #この時間含む(以上)
end_day =   "2023/11/04 00:00:00"  # この時間含めない(未満)

start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) -1 #含めないので1秒マイナス

bet_term = 2
terms = [6]
except_hour_list = [20, 21, 22, ]
symbol = "EURUSD"

target_colum = "60-satr-5"
need_len = calc_need_len([target_colum], bet_term)

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

adx_list = []

for term in terms:
    """
    db_list = []
   
    if term >= bet_term:
        for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
            db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))
    else:
        db_list.append(symbol + "_" + str(term) + "_0")
    """
    db_list = ["EURUSD_2_IND"]

    for db in db_list:
        print(db)
        result_data = redis_db.zrangebyscore(db, start_stp, end_stp, withscores=True)

        for i, line in enumerate(result_data):
            body = line[0]
            score = int(line[1])
            # 取引時間外を対象からはずす
            if len(except_hour_list) != 0:
                if datetime.fromtimestamp(score).hour in except_hour_list:
                    continue

            try:
                start_score = result_data[i - need_len][1]
                end_score = result_data[i][1]

                if end_score != start_score + (need_len * bet_term):
                    # print(start_score, end_score, start_score + ((need_len + pred_term_length) * bet_term))
                    # 時刻がつながっていないものは除外 たとえば日付またぎなど
                    continue
            except Exception:
                # start_score end_scoreのデータなしなのでスキップ
                continue

            tmps = json.loads(body)
            if np.isnan(tmps.get(target_colum)) == False:
                adx_list.append(tmps.get(target_colum))


adx_np = np.array(adx_list)
adx_np_sort = np.sort(adx_np) #昇順にならべかえ
adx_len = len(adx_np_sort)
print("data length:", adx_len)
#print("中央値:",np.median(adx_np_sort))
num = 199
for i in range(num):
    print(str(i+1) + "/" + str(num+1) +"値:",adx_np_sort[int((adx_len / (num+1)) * (i + 1)) - 1])

print("MAX:",np.max(adx_np_sort))
print("MIN:",np.min(adx_np_sort))
print("AVG:", np.average(adx_np_sort))
print("STD:", np.std(adx_np_sort))