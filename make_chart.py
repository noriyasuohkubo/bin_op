import mplfinance as mpf
import pandas as pd
import redis
import numpy as np
from datetime import  datetime, timedelta
import time
import json
import matplotlib.pyplot as plt
import tkinter as tk
from util import *


#history_list(例:score,stime,etime,sprice, eprice, type(BUY or SELL), profit_pips,)
def make_chart(history_list, save_dir=None):
    if save_dir == None:
        save_dir = "/app/fx/chart/tmp"

    dt_str = history_list[0]["stime"]

    #print("make chart:", dt_str)
    #print(history_list)

    if save_dir != None:
        tmp_dt_str = dt_str.replace("/", "-").replace(" ","_")
        filename = save_dir + "/" + tmp_dt_str +  ".png"

    tmp_dt = datetime.strptime(dt_str, '%Y/%m/%d %H:%M:%S')

    mark_up_start_x_win = []
    mark_up_start_y_win = []
    mark_dw_start_x_win = []
    mark_dw_start_y_win = []

    mark_up_end_x_win = []
    mark_up_end_y_win = []
    mark_dw_end_x_win = []
    mark_dw_end_y_win = []

    mark_up_start_x_lose = []
    mark_up_start_y_lose = []
    mark_dw_start_x_lose = []
    mark_dw_start_y_lose = []

    mark_up_end_x_lose = []
    mark_up_end_y_lose = []
    mark_dw_end_x_lose = []
    mark_dw_end_y_lose = []

    start_dt = tmp_dt + timedelta(hours=-2,minutes=-6)
    end_dt = tmp_dt + timedelta(hours=1)

    start_stp = int(time.mktime(start_dt.timetuple()))
    end_stp = int(time.mktime(end_dt.timetuple())) - 1

    redis_db_new = redis.Redis(host='127.0.0.1', port=6379, db=1, decode_responses=True)
    new_db_name = 'USDJPY_M1'

    result_data = redis_db_new.zrangebyscore(new_db_name, start_stp, end_stp, withscores=True)

    d = {
        'time':[],
        'Open':[],
        "Close": [],
        'High': [],
        "Low": [],
        "EMA13": [],
        "EMA50": [],
        "EMA100": [],
        "EMA500": [],
    }

    for i, line in enumerate(result_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)
        d['time'].append(datetime.fromtimestamp(score))
        d['Open'].append(float(tmps.get("o")))
        d['Close'].append(float(tmps.get("c")))
        d['High'].append(float(tmps.get("h")))
        d['Low'].append(float(tmps.get("l")))

        d['EMA13'].append(float(tmps.get("ema-13")))
        d['EMA50'].append(float(tmps.get("ema-50")))
        d['EMA100'].append(float(tmps.get("ema-100")))
        d['EMA500'].append(float(tmps.get("ema-500")))

        for h in history_list:
            sc = datetime.strptime(h["stime"], '%Y/%m/%d %H:%M:%S').timestamp()
            sc = get_decimal_sub(sc, get_decimal_mod(sc, 60))
            type = h["type"]
            pips = h["profit_pips"]

            if sc == score:
                if pips >= 0:
                    if type == 'BUY':
                        mark_up_start_x_win.append(i)
                        mark_up_start_y_win.append(float(h["sprice"]))
                    elif type == 'SELL':
                        mark_dw_start_x_win.append(i)
                        mark_dw_start_y_win.append(float(h["sprice"]))
                else:
                    if type == 'BUY':
                        mark_up_start_x_lose.append(i)
                        mark_up_start_y_lose.append(float(h["sprice"]))
                    elif type == 'SELL':
                        mark_dw_start_x_lose.append(i)
                        mark_dw_start_y_lose.append(float(h["sprice"]))

            sc = datetime.strptime(h["etime"], '%Y/%m/%d %H:%M:%S').timestamp()
            sc = get_decimal_sub(sc, get_decimal_mod(sc, 60))
            if sc == score:
                if pips >= 0:
                    if type == 'BUY':
                        mark_up_end_x_win.append(i)
                        mark_up_end_y_win.append(float(h["eprice"]))
                    elif type == 'SELL':
                        mark_dw_end_x_win.append(i)
                        mark_dw_end_y_win.append(float(h["eprice"]))
                else:
                    if type == 'BUY':
                        mark_up_end_x_lose.append(i)
                        mark_up_end_y_lose.append(float(h["eprice"]))
                    elif type == 'SELL':
                        mark_dw_end_x_lose.append(i)
                        mark_dw_end_y_lose.append(float(h["eprice"]))

    df =pd.DataFrame(d)
    df.set_index('time', inplace=True)
    # 移動平均線の計算
    #df['EMA13'] = df['Close'].rolling(window=13).mean()
    #df['EMA50'] = df['Close'].rolling(window=50).mean()
    #df['EMA100'] = df['Close'].rolling(window=100).mean()
    #df['EMA500'] = df['Close'].rolling(window=500).mean()
    #for index, row in df.iterrows():
    #    print(row)

    # 4. チャートの描画
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        title='USD/JPY M1 Chart',
        #mav=(13,50,100),  # sma移動平均線を表示
        #ema=(13,50, 100,500),  # ema移動平均線を表示
        #ema=(10,200),  # ema移動平均線を表示
        volume=False,
        returnfig =True,
    )
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # 2. メインチャートのAxes（通常は axes[0]）を取得して直線を引く
    ax = axes[0]

    # Matplotlibのplotやaxhline/axvline、axlineを使用
    ax.plot(d['EMA13'],)
    ax.plot(d['EMA50'],)
    ax.plot(d['EMA100'],)
    ax.plot(d['EMA500'],)

    ax.scatter(mark_up_start_x_win, mark_up_start_y_win,marker='^', c = 'magenta')
    ax.scatter(mark_up_end_x_win, mark_up_end_y_win,marker='v', c = 'magenta')

    ax.scatter(mark_dw_start_x_win, mark_dw_start_y_win,marker='>', c = 'magenta')
    ax.scatter(mark_dw_end_x_win, mark_dw_end_y_win,marker='<', c = 'magenta')

    ax.scatter(mark_up_start_x_lose, mark_up_start_y_lose, marker='^', c='red')
    ax.scatter(mark_up_end_x_lose, mark_up_end_y_lose, marker='v', c='red')

    ax.scatter(mark_dw_start_x_lose, mark_dw_start_y_lose, marker='>', c='red')
    ax.scatter(mark_dw_end_x_lose, mark_dw_end_y_lose, marker='<', c='red')

    #plt.show()

    if save_dir != None:
        fig.savefig(filename)

if __name__ == '__main__':

    history_list_tmp = [
        [1734615412.0, '2024/12/19 13:36:52', '2024/12/19 13:37:22', 156.851, 156.863, 'BUY', 0.012, -3.390238280205349,
         -0.6913913755490775, -0.608262276805549, 23.903894518142454],
    ]

    history_list = []
    for h in history_list_tmp:
        history_list.append(
            {
                "score":float(h[0]),
                "stime": h[1],
                "etime": h[2],
                "sprice": float(h[3]),
                "eprice": float(h[4]),
                "type": h[5],
                "profit_pips": float(h[6]),
            }
        )


    make_chart(history_list=history_list, )