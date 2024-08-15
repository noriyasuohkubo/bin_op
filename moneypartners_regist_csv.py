import csv
import json

import redis
from datetime import datetime, timedelta
from util import *
import time

"""
マネパのデータダウンロードの項目から約定履歴をCSVダウンロードし、
ファイルの最初の3行を削除して、注文番号の昇順に並べなおしておくこと！！！
"""

def regist_csv(regist_pair, redis_db, db_name, filename):
    format = '%Y/%m/%d %H:%M:%S'
    regist_cnt = 0
    start_rate = None
    order_score = None

    try:
        with open(filename, encoding='Shift-jis', newline='') as f:
            dict_reader = csv.DictReader(f)

            #for i in dict_reader.fieldnames:
            #    print(i)

            for row in dict_reader:
                kubun = row["注文区分"]

                if kubun == "新規":
                    pair = row["通貨ペア"]
                    lot = float(row["約定数量"])
                    order_time = row["約定日時"]
                    order_dt = datetime.strptime(order_time, format) + timedelta(hours=-9)
                    order_score = int(time.mktime(order_dt.timetuple()))
                    start_rate = float(row["約定レート"])

                elif kubun == "決済":
                    type = row["売買"]
                    end_rate = float(row["約定レート"])
                    deal_time = row["約定日時"]
                    deal_dt = datetime.strptime(deal_time, format) + timedelta(hours=-9)
                    deal_score = int(time.mktime(deal_dt.timetuple()))

                    if type == "買":
                        sign = 2
                        profit = get_decimal_sub(start_rate, end_rate)
                    elif type == "売":
                        sign = 0
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


    except Exception as e:
        print("Error Occured!!:", tracebackPrint(e))

    print("regist_cnt:", regist_cnt)

if __name__ == "__main__":
    host = 'win5'
    db_no = 8
    #登録するペア
    regist_pair = "USD/JPY"
    db_name = "USDJPY" + "_60_MONEYPARTNERS_HISTORY"

    redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
    dir = "/db2/win5/"
    filenames = ["PFX_NANO_EXECUTION_20240815054604.csv",

                 ]

    print("start moneypartners regist csv")
    for filename in filenames:
        regist_csv(regist_pair, redis_db, db_name, dir + filename)
    print("end moneypartners regist csv")