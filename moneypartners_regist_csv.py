import csv
import json

import redis
from datetime import datetime, timedelta
from util import *
import time

"""
マネパのデータダウンロードの項目から約定履歴をCSVダウンロードし、
ファイルの最初の3行を削除して、注文番号の降順に並んであること！！！
"""

def regist_csv(regist_pair, redis_db, db_name, filename):
    format = '%Y/%m/%d %H:%M:%S'
    regist_cnt = 0

    prev_kessai = None

    try:
        with open(filename, encoding='Shift-jis', newline='') as f:
            dict_reader = csv.DictReader(f)

            #for i in dict_reader.fieldnames:
            #    print(i)

            for row in dict_reader:
                kubun = row["注文区分"]

                if kubun == "新規":
                    pair = row["通貨ペア"]
                    type = row["売買"]
                    lot = float(str(row["約定数量"]).replace(",", ""))
                    order_time = row["約定日時"]
                    order_dt = datetime.strptime(order_time, format) + timedelta(hours=-9)
                    order_score = int(time.mktime(order_dt.timetuple()))
                    start_rate = float(row["約定レート"])

                    if type == "買":
                        sign = 0
                    elif type == "売":
                        sign = 2

                    if prev_kessai != None and prev_kessai["start_rate"] == start_rate :
                        print(prev_kessai)
                        if (sign == 0 and prev_kessai["sign"] == 2) or (sign == 2 and prev_kessai["sign"] == 0):

                            end_rate = prev_kessai["end_rate"]
                            deal_score = prev_kessai["deal_score"]

                            if type == "買":
                                profit = get_decimal_sub(start_rate, end_rate)
                            elif type == "売":
                                profit = get_decimal_sub(end_rate, start_rate)

                            child ={
                                'order_score':order_score,
                                'deal_score': deal_score,
                                'start_rate': start_rate,
                                'end_rate': end_rate,
                                'profit':profit,
                                'sign':sign,
                                'lot':lot,

                            }

                            if pair == regist_pair:
                                if deal_score != None:
                                    # 既存レコードがなければ追加
                                    tmp_val = redis_db.zrangebyscore(db_name, order_score, order_score)
                                    if len(tmp_val) == 0:
                                        redis_db.zadd(db_name, json.dumps(child), order_score)
                                        regist_cnt += 1

                    prev_kessai = None

                elif kubun == "決済":
                    type = row["売買"]
                    start_rate = float(row["建玉レート"])
                    end_rate = float(row["約定レート"])
                    deal_time = row["約定日時"]
                    deal_dt = datetime.strptime(deal_time, format) + timedelta(hours=-9)
                    deal_score = int(time.mktime(deal_dt.timetuple()))

                    if type == "買":
                        sign = 0
                    elif type == "売":
                        sign = 2

                    prev_kessai ={
                        'deal_score': deal_score,
                        'start_rate': start_rate,
                        'end_rate': end_rate,
                        'sign':sign,
                    }




    except Exception as e:
        print("Error Occured!!:", tracebackPrint(e))

    print("regist_cnt:", regist_cnt)

if __name__ == "__main__":
    host = 'win7'
    db_no = 8
    #登録するペア
    regist_pair = "USD/JPY"
    db_name = "USDJPY" + "_4_MONEYPARTNERS_HISTORY"

    redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
    dir = "/Users/reico/Downloads/"
    filenames = ["PFX_EXECUTION_20250307082807.csv",

                 ]

    print("start moneypartners regist csv")
    for filename in filenames:
        regist_csv(regist_pair, redis_db, db_name, dir + filename)
    print("end moneypartners regist csv")