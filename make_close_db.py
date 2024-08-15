import json
import numpy as np
import os
import redis
import datetime
import time
import gc
import math
from decimal import Decimal

"""
make_base_dbで作成したDBデータにもとづいて
closeの変化率とclose,timeの値を保存していく

※dukasのデータは日本時間の土曜朝7時から月曜朝7時までデータがない
"""


# 処理時間計測
t1 = time.time()

#抽出元のDB名
symbol_org = "USDJPY"

close_shift = 1

#新規に作成するDB名
symbol = "USDJPY_5_0_TICK"

#直前の1分足のスコアをデータに含めるか
#例えば12：03：44なら12:02
include_min = False

close_flg = False

#変化率のlogをとる
math_log = False

volume_flg = False

#FXのテストデータをtickからつくる場合True
spread_flg = False

#FXのテストデータをtickからつくる場合True
tk_flg = True

#highとlowの変化率とclose,high,lowのレートを保持する
highlow_flg = False

#元のclose値を100倍するか
close_100_flg = False

in_db_no = 2
out_db_no = 2
host = "127.0.0.1"

start = datetime.datetime(2021, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime.datetime(2022, 5, 1)
end_stp = int(time.mktime(end.timetuple()))

redis_db_in = redis.Redis(host="ub3", port=6379, db=in_db_no, decode_responses=True)
redis_db_out = redis.Redis(host=host, port=6379, db=out_db_no, decode_responses=True)

result_data = redis_db_in.zrangebyscore(symbol_org, start_stp, end_stp, withscores=True)
print("result_data length:" + str(len(result_data)))

close_tmp, time_tmp, score_tmp = [], [], []
high_tmp, low_tmp = [] ,[]

volume_tmp, spread_tmp, tk_tmp = [], [], []

def get_divide(bef, aft):
    divide = aft / bef
    if aft == bef:
        divide = 1

    if math_log:
        divide = 10000 * math.log(divide, math.e * 0.1)
    else:
        divide = 10000 * (divide - 1)

    return divide

for line in result_data:
    body = line[0]
    score = line[1]
    tmps = json.loads(body)

    score_tmp.append(score)
    if close_100_flg == False:
        close_tmp.append(float(tmps.get("close")))
    else:
        close_tmp.append( float( Decimal(str(tmps.get("close"))) * Decimal(str(100)))  ) #dukaのレートは本来のレートの100分の1なのでもとにもどす

    time_tmp.append(tmps.get("time"))

    if volume_flg:
        volume_tmp.append(int(tmps.get("tick_volume")))
    if spread_flg:
        spread_tmp.append(int(tmps.get("spr")))
    if tk_flg:
        tk_tmp.append(tmps.get("tk"))

    if highlow_flg:
        if tk_flg:
            high_tmp.append( float(tmps.get("high")))
            low_tmp.append( float(tmps.get("low")))
        else:
            high_tmp.append( float( Decimal(str(tmps.get("high"))) * Decimal(str(100)))  )
            low_tmp.append( float( Decimal(str(tmps.get("low"))) * Decimal(str(100)))  )

close_np = np.array(close_tmp)
high_np = np.array(high_tmp)
low_np = np.array(low_tmp)

print("gc end")
print("close_tmp len:", len(close_tmp))
print(close_tmp[:10])

print("start:", datetime.datetime.fromtimestamp(score_tmp[0]))
print("end:", datetime.datetime.fromtimestamp(score_tmp[-1]))

# メモリ解放
del result_data, close_tmp, high_tmp, low_tmp
#gc.collect()

#変化率を作成
for i, v in enumerate(close_np):
    #変化元(close_shift前のデータ)がないのでとばす
    if i < close_shift:
        continue

    child = {}

    child['t'] = time_tmp[i]
    child['d'] = get_divide(close_np[i - close_shift], close_np[i])

    if highlow_flg:
        high_divide = get_divide(high_np[i - close_shift], high_np[i])
        low_divide = get_divide(low_np[i - close_shift], low_np[i])
        hl_divide = get_divide(high_np[i], low_np[i])

        child['h'] = high_np[i]
        child['hd'] = high_divide
        child['l'] = low_np[i]
        child['ld'] = low_divide
        child['hld'] = hl_divide

    if close_flg:
        child['c'] = close_np[i]
    if volume_flg:
        child['v'] = volume_tmp[i]
    if spread_flg:
        child['s'] = spread_tmp[i]
    if tk_flg:
        child['tk'] = tk_tmp[i]
    """
    if include_min:
        tmp_score = int(score_tmp[i]) - (int(time_tmp[i][-2:]) % 60) - 60
        child['min_score'] = tmp_score
    """

    #tmp_val = redis_db_out.zrangebyscore(symbol, score_tmp[i], score_tmp[i])
    #if len(tmp_val) == 0:
    #    redis_db_out.zadd(symbol, json.dumps(child), score_tmp[i])

    redis_db_out.zadd(symbol, json.dumps(child), score_tmp[i])

    if i % 10000000 == 0:
        dt_now = datetime.datetime.now()
        print(dt_now, " ", i)

t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))

#手動にて抽出元のDBを削除して永続化すること！
#print("now db saving")
#redis_db.save()
