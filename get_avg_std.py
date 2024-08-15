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
from util import *
import subprocess
import psutil

"""
変化率を取得して、平均、標準偏差、中央値を求めヒストグラムを描写

"""

start_day = "2022/11/01 00:00:00"  # この時間含む(以上)
end_day = "2023/12/03 00:00:00"  # この時間含めない(未満)


start_day_dt = datetime.strptime(start_day, '%Y/%m/%d %H:%M:%S')
end_day_dt = datetime.strptime(end_day, '%Y/%m/%d %H:%M:%S')

start_stp = int(time.mktime(start_day_dt.timetuple()))
end_stp = int(time.mktime(end_day_dt.timetuple())) - 1  # 含めないので1秒マイナス

math_log = False

db_no = 2
# 取得元DB
db_name = "JPNIDXJPY_2_0"

# 予測間隔
pred_term = 30
bet_term = 2
print("pred_sec",pred_term * bet_term)


ans_term = str(int(Decimal(str(pred_term)) * Decimal(str(bet_term))))

except_hour_list = [20,21,22]
target_spr_list = []

div_list = {"divide0.1": (0.1, 0.5),
            "divide0.5": (0.5, 1.0), "divide1.0": (1.0, 2.0), "divide2.0": (2.0, 3.0), "divide3.0": (3.0, 4.0),
            "divide4.0": (4.0, 5.0),
            "divide5.0": (5.0, 6.0), "divide6.0": (6.0, 7.0),
            "divide7.0over": (7.0, 10000),
            }

div_close_list = {"divide-0.5": (-1.0, -0.5), "divide-1.0": (-2.0, -1.0), "divide-2.0": (-3.0, -2.0),
                  "divide-3.0": (-4.0, -3.0), "divide-4.0": (-5.0, -4.0),
                  "divide-5.0": (-6.0, -5.0), "divide-6.0": (-7.0, -6.0),
                  "divide-7.0over": (-10000, -7.0),
                  "divide0.5": (0.5, 1.0), "divide1.0": (1.0, 2.0), "divide2.0": (2.0, 3.0), "divide3.0": (3.0, 4.0),
                  "divide4.0": (4.0, 5.0),
                  "divide5.0": (5.0, 6.0), "divide6.0": (6.0, 7.0),
                  "divide7.0over": (7.0, 10000),
                  }

div_abs_per_div_m_up = {}
div_abs_per_div_m_dw = {}

for k, v in div_list.items():
    div_abs_per_div_m_up[k] = []
    div_abs_per_div_m_dw[k] = []

div_close = {}
for k, v in div_close_list.items():
    div_close[k] = []

redis_db = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

result_data = redis_db.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
# print("result_data length:" + str(len(result_data)))

"""
# メモリ節約のためredis停止
#r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
sudo_password = 'Reikou0129'
command = 'systemctl stop redis'.split()
p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
sudo_prompt = p.communicate(sudo_password + '\n')[1]
# メモリ空き容量を取得
print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
"""

data_list = []
close_divide_list = []
close_sub_list = []
mean_divide_list = []
std_divide_list = []

print("data length", len(result_data))
for line in result_data:
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)
    tmps["score"] = score
    data_list.append(tmps)

del result_data
gc.collect()

for i, data in enumerate(data_list):

    if i < pred_term:
        continue

    try:
        start_score = data_list[i - 1]["score"]
        end_score = data_list[i - 1 + pred_term]["score"]
        if end_score != start_score + (pred_term * bet_term):
            # 時刻がつながっていないものは除外 たとえば日付またぎなど
            continue
    except IndexError:
        # start_scoreのデータなしなのでスキップ
        continue

    # 取引時間外を対象からはずす
    if len(except_hour_list) != 0:
        if datetime.fromtimestamp(data_list[i]["score"]).hour in except_hour_list:
            continue

    # 対象スプレッド以外をはずす
    if len(target_spr_list) != 0:
        if not (data_list[i - 1]["s"] in target_spr_list):
            continue

    bef = data_list[i - 1]["c"]
    aft = data_list[i - 1 + pred_term]["c"]
    divide = get_divide(bef, aft)
    sub = get_sub(bef, aft)
    close_divide_list.append(abs(divide))
    close_sub_list.append(abs(sub))

    for k, v in div_close_list.items():
        if divide > v[0] and divide <= v[1]:
            div_close[k].append(divide)
            break

    """
    bef_m = float(data_list[i][ans_term + "bm"])
    aft_m = float(data_list[i][ans_term + "am"])
    divide_m = get_divide(bef_m, aft_m)

    mean_divide_list.append(abs(divide_m))

    aft_std = float(data_list[i][ans_term + "as"])
    std_divide_list.append(abs(get_divide(aft_m, aft_m + aft_std)))

    if divide_m > 0:
    for k, v in div_list.items():
        if divide_m > v[0] and divide_m <= v[1]:
            div_abs_per_div_m_up[k].append(divide)
            break
    elif divide_m < 0:
        divide_m = divide_m * -1
        for k, v in div_list.items():
            if divide_m > v[0] and divide_m <= v[1]:
                div_abs_per_div_m_dw[k].append(divide * -1)
                break
    """
"""
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(close_list, 'g')
plt.show()
"""


print("終値のDIV分布")
d_np = np.array(close_divide_list)
print("data length:", len(d_np))
print("avg:", np.average(d_np))
print("std:", np.std(d_np))
print("mid:", d_np[int(len(d_np) / 2) - 1])  # 中央値
print("max:", np.max(d_np))
print("min:", np.min(d_np))
#print("max / avg:", np.max(d_np) / np.average(d_np))
"""
d_np_abs = abs(d_np)
d_np_abs_sort = np.sort(d_np_abs) #昇順にならべかえ
d_np_abs_sort_len = len(d_np_abs_sort)
#print("中央値:",np.median(adx_np_sort))
#DIVが小さい方から順に表示してハズレ値を確認する
num = 10000
tmp_list = []
for i in range(num):
    # print(str(i+1) + "/" + str(num) +"値:",d_np_abs_sort[int((d_np_abs_sort_len / (num)) * (i + 1)) - 1])
    tmp_list.append((d_np_abs_sort[int((d_np_abs_sort_len / (num)) * (i + 1)) - 1]))

plt.plot(tmp_list)
plt.show()
"""


print("終値のSUB分布")
d_np = np.array(close_sub_list)
print("data length:", len(d_np))
print("avg:", np.average(d_np))
print("std:", np.std(d_np))
print("mid:", d_np[int(len(d_np) / 2) - 1])  # 中央値
print("max:", np.max(d_np))
print("min:", np.min(d_np))
#print("max / avg:", np.max(d_np) / np.average(d_np))

"""
d_np_abs = abs(d_np)
d_np_abs_sort = np.sort(d_np_abs) #昇順にならべかえ
d_np_abs_sort_len = len(d_np_abs_sort)
#print("中央値:",np.median(adx_np_sort))
#DIVが小さい方から順に表示してハズレ値を確認する
num = 10000
tmp_list = []
for i in range(num):
    #print(str(i+1) + "/" + str(num) +"値:",d_np_abs_sort[int((d_np_abs_sort_len / (num)) * (i + 1)) - 1])
    tmp_list.append((d_np_abs_sort[int((d_np_abs_sort_len / (num)) * (i + 1)) - 1]))

plt.plot(tmp_list)
plt.show()

targets = [0.05,0.075, 0.1,0.2,0.3,0.4]
for target in targets:
    print("targetover:",target, len(np.where(d_np_abs >= target)[0])/d_np_abs_sort_len * 100, "%")

"""

"""
for i in range(200):
    target = float(Decimal(str(i + 1)) * Decimal("0.1"))
    num = len(np.where(d_np >= target)[0])
    print("div over ", target, " :", len(np.where(d_np >= target)[0]), " percent:", num / len(d_np) * 100)

print("")

print("終値の変化率ごとの件数")
for k, v in div_close_list.items():
    tmp_np = np.array(div_close[k])
    print("DIV ", k, " total:", len(tmp_np), " percent:", len(tmp_np) / len(d_np) * 100, " avg:",
          np.average(tmp_np))
print("")
"""

"""
print("平均値の分布")
d_np = np.array(mean_divide_list)
print("data length:", len(d_np))
print("avg:", np.average(d_np))
print("std:", np.std(d_np))
print("mid:", d_np[int(len(d_np)/2) -1]) #中央値
print("max:", np.max(d_np))
print("min:", np.min(d_np))
print("max / avg:", np.max(d_np)/np.average(d_np))

for i in range(20):
    target = i + 1
    num = len(np.where(d_np >= target )[0])
    print("div over ", target, " :", len(np.where(d_np >= target )[0]), " percent:", num/len(d_np)*100 )

print("")

print("偏差の分布")
d_np = np.array(std_divide_list)
print("data length:", len(d_np))
print("avg:", np.average(d_np))
print("std:", np.std(d_np))
print("mid:", d_np[int(len(d_np)/2) -1]) #中央値
print("max:", np.max(d_np))
print("min:", np.min(d_np))
print("max / avg:", np.max(d_np)/np.average(d_np))

for i in range(20):
    target = i + 1
    num = len(np.where(d_np >= target )[0])
    print("div over ", target, " :", len(np.where(d_np >= target )[0]), " percent:", num/len(d_np)*100 )

print("")

print("平均レートがあがった時に終値もあがった場合")
for k, v in div_list.items():
    up_np = np.array(div_abs_per_div_m_up[k])
    up_correct_np = np.where(up_np > 0)[0] #平均レートがあがった時に終値も上がった場合

    print("UP ", k, " total:", len(up_np), " correct:", len(up_correct_np)/len(up_np), " avg:", np.average(up_np))

print("")
print("平均レートがさがった時に終値もさがった場合")
for k, v in div_list.items():
    dw_np = np.array(div_abs_per_div_m_dw[k])
    dw_correct_np = np.where(dw_np > 0)[0]  # 平均レートがあがった時に終値も上がった場合

    print("DW ", k, " total:", len(dw_np), " correct:", len(dw_correct_np) / len(dw_np), " avg:", np.average(dw_np))
"""

"""
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
print("")
"""





