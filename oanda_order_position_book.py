
import copy
import time
from datetime import datetime
from bokeh.plotting import figure, show
from oandapyV20 import API
from bokeh.layouts import gridplot
from datetime import timedelta

import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import pytz
import numpy as np
import redis
import json
from util import *

"""
オアンダAPIからオープンオーダー、オープンポジションの取得
5分おきのデータとなる
各priceは幅の下限を示している

"""
oanda_symbol = "USD_JPY"
symbol = oanda_symbol.replace("_","")

db_no = 2
db_name_order = symbol + "_OANDA_ORD"
db_name_position = symbol + "_OANDA_POS"

up_count = 20 #DB登録する現在のレートより高いレンジのデータ数
dw_count = up_count + 1 #DB登録する現在のレートより高いレンジのデータ数


account_number = "001-009-1776566-002"
access_token = "d8857ea62b46560c67a3f1795d821cfc-871c87d6775ae067960de7a0bfe7f9d0"

api = API(access_token=access_token, environment="live")

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)


def getInstrumentsOrderBook(params, regist_score):

    # APIへ過去データをリクエスト
    ic = instruments.InstrumentsOrderBook(instrument=oanda_symbol,
                                          params=params)
    api.request(ic)
    #print(ic.response)
    order_book = ic.response["orderBook"]
    time = order_book["time"]
    now_price = order_book["price"]
    bucketWidth = order_book["bucketWidth"]

    price_list = []
    vol_list = [] #longCountPercent - shortCountPercentのリスト

    for raw in order_book["buckets"]:
        price_list.append(raw["price"])
        vol_list.append(float(Decimal(raw["longCountPercent"]) - Decimal(raw["shortCountPercent"])))

    price_list = np.array(price_list)
    vol_list = np.array(vol_list)

    up_idx = np.where(now_price < price_list)
    dw_idx = np.where(now_price >= price_list)

    up_dict = dict(zip(price_list[up_idx],vol_list[up_idx]))
    dw_dict = dict(zip(price_list[dw_idx],vol_list[dw_idx]))

    up_sort = sorted(up_dict.items(), key=lambda x: x[0], ) #priceを昇順にする
    dw_sort = sorted(dw_dict.items(), key=lambda x: x[0], reverse=True) #priceを降順にする

    child = {
        "time":time,
        "wid":bucketWidth,
    }

    data_list = []
    for i, d in enumerate(dw_sort):
        if i >= dw_count:
            break

        k = d[0]
        v = d[1]
        data_list.append(str(k) + ":" + str(v))

    data_list.reverse() #priceの昇順にする.DBにはpriceの昇順で登録するため

    for i, d in enumerate(up_sort):
        if i >= up_count:
            break

        k = d[0]
        v = d[1]
        data_list.append(str(k) + ":" + str(v))


    data_list_str = list_to_str(data_list, ",")
    child["data"] = data_list_str

    tmp_val = redis_db.zrangebyscore(db_name_order, regist_score, regist_score)
    if len(tmp_val) == 0:
        redis_db.zadd(db_name_order, json.dumps(child), regist_score)

def getInstrumentsPositionBook(params, regist_score):

    # APIへ過去データをリクエスト
    ic = instruments.InstrumentsPositionBook(instrument=oanda_symbol,
                                          params=params)
    api.request(ic)
    #print(ic.response)
    order_book = ic.response["positionBook"]
    time = order_book["time"]
    now_price = order_book["price"]
    bucketWidth = order_book["bucketWidth"]

    price_list = []
    vol_list = [] #longCountPercent - shortCountPercentのリスト

    for raw in order_book["buckets"]:
        price_list.append(raw["price"])
        vol_list.append(float(Decimal(raw["longCountPercent"]) - Decimal(raw["shortCountPercent"])))

    price_list = np.array(price_list)
    vol_list = np.array(vol_list)

    up_idx = np.where(now_price < price_list)
    dw_idx = np.where(now_price >= price_list)

    up_dict = dict(zip(price_list[up_idx],vol_list[up_idx]))
    dw_dict = dict(zip(price_list[dw_idx],vol_list[dw_idx]))

    up_sort = sorted(up_dict.items(), key=lambda x: x[0], ) #priceを昇順にする
    dw_sort = sorted(dw_dict.items(), key=lambda x: x[0], reverse=True) #priceを降順にする

    child = {
        "time":time,
        "wid":bucketWidth,
    }

    data_list = []
    for i, d in enumerate(dw_sort):
        if i >= dw_count:
            break

        k = d[0]
        v = d[1]
        data_list.append(str(k) + ":" + str(v))

    data_list.reverse() #priceの昇順にする.DBにはpriceの昇順で登録するため

    for i, d in enumerate(up_sort):
        if i >= up_count:
            break

        k = d[0]
        v = d[1]
        data_list.append(str(k) + ":" + str(v))


    data_list_str = list_to_str(data_list, ",")
    child["data"] = data_list_str

    tmp_val = redis_db.zrangebyscore(db_name_position, regist_score, regist_score)
    if len(tmp_val) == 0:
        redis_db.zadd(db_name_position, json.dumps(child), regist_score)

def convert_to_utc(dt):
    return dt.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":

    start_time = time.perf_counter()

    start_dt = datetime(2023, 11, 5, 0, 0) #unix time
    #start_dt = datetime(2023, 5, 21, 20, 0) #unix time
    end_dt = datetime(2023, 12, 3, 0, 0) #unix time

    while True:
        if end_dt <= start_dt:
            break

        params = {
            "time": convert_to_utc(start_dt)
        }
        regist_score = int(time.mktime(start_dt.timetuple()))
        print(start_dt)

        try:
            getInstrumentsOrderBook(params, regist_score)
            #time.sleep(1)
            getInstrumentsPositionBook(params, regist_score)
        except Exception as e:
            print(e)

        start_dt = start_dt + timedelta(minutes=5)

        #time.sleep(1)


    print("END! Processing Time(Sec)", time.perf_counter() - start_time)