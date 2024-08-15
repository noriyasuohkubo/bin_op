import os
import math
import numpy as np
from decimal import Decimal
import traceback
from datetime import timedelta ,date

def get_decimal_add(a,b):
    return float(Decimal(str(a)) + Decimal(str(b)))

def get_decimal_sub(a,b):
    return float(Decimal(str(a)) - Decimal(str(b)))

def get_decimal_multi(a,b):
    return float(Decimal(str(a)) * Decimal(str(b)))

def get_decimal_divide(a,b):
    return float(Decimal(str(a)) / Decimal(str(b)))

def get_decimal_mod(a,b):
    return float(Decimal(str(a)) % Decimal(str(b)))

#インデックスを求めるために必要なデータの最大長
#rsi:15
#cmo:15
#ppo:26
#slowd:9
#willr:14
#d1:2 d5:6 d15:16...
#sma5-1:6 sma5-5:10 sma5-15:20... timeperiod+roll
#dip7:8 dip14:15
#dim7:8 dip14:15
#adx7:14 adx14:28 period * 2
def get_need_len():
    need_length_dict = {
        "rsi": 0,
        "d": 0,
        "d20000": 0,
        "d5000": 0,
        "d2500": 0,
        "d1000": 0,
        "md20000": 0,
        "md10000": 0,
        "md": 0,
        "md1": 1,
        "wmd1": 1,
        "md5000": 0,
        "md2500": 0,
        "md1000": 0,
        "ad": 0,
        "bd": 0,
        "ehd1": 2,
        "eld1": 2,
        "d1": 0,
        "std_d":0,
        "sub": 0,
        "sub1": 0,
        "sub10": 0,
        "sub100": 0,
        "sub1000": 0,
        "subd": 0,
        "dm": 0,
        "sma": 0,
        "smac": -1,
        "smam": 0,
        "mm~smam1":0,
        "smams15":0,
        "smams30": 0,
        "smams60": 0,
        "wmams15": 0,
        "wmams30": 0,
        "wmams60": 0,
        "smam1": 0,
        "smam2": 1,
        "smam3": 2,
        "smam4": 3,
        "smam5": 4,
        "smam10": 9,
        "smam12": 11,
        "smam20": 19,
        "smam40": 39,

        "wmam1": 0,
        "ssmam": 0,
        "dip": 0,
        "dim": 0,
        "dx":0,
        "adx": -1,
        "atr": 1,
        "satr": 1,

        "bbu3": -1,
        "bbu2": -1,
        "bbu1": -1,
        "bbl1": -1,
        "bbl2": -1,
        "bbl3": -1,

        "bbcu3": -1,
        "bbcu2": -1,
        "bbcu1": -1,
        "bbcl1": -1,
        "bbcl2": -1,
        "bbcl3": -1,

        "sbbu3": -1,
        "sbbm": -1,
        "sbbl3": -1,

        "bbmu3": -1,
        "bbml3": -1,
        "sbbmu3": -1,
        "sbbml3": -1,
    }
    return need_length_dict

# 最大のidx計算に必要なデータの長さを算出
def calc_need_len(cols, bet_term):
    if len(cols) != 0:
        need_lens = get_need_len()
        tmp_need_length_list = []
        for col in cols:
            t, idx, t_len = col.split("-")
            if idx[0] == "f":
                #頭文字がfの場合は分足のデータなのでidx計算に必要な過去秒数は求めない(dukasデータは間に空きはない想定)
                tmp_need_length_list.append(bet_term) #最小秒数をいれとく
            else:
                need_sec = int(Decimal(str(need_lens[idx] + int(t_len))) * Decimal(str(t)))  # idx計算に必要な過去秒数を求める
                tmp_need_length_list.append(need_sec)

        # 最大のidx計算に必要な過去term
        #print(tmp_need_length_list)
        #print(max(tmp_need_length_list))
        need_len = Decimal(str(max(tmp_need_length_list))) / Decimal(str(bet_term))
        #print("need_len:", need_len)

        return need_len
    else:
        return 0

# 最大のidx計算に必要なデータの長さを算出
def get_max_need_len(cols):
    if len(cols) != 0:
        need_lens = get_need_len()
        tmp_need_length_list = []
        for col in cols:
            if col == "d1":
                tmp_need_length_list.append(1)
            else:
                idx, t_len = col.split("-")
                tmp_need_length_list.append(need_lens[idx] + int(t_len))
        return max(tmp_need_length_list)
    else:
        return 0

def get_divide_dec(bef, aft, multi=10000, math_log=False, e=0.1, r=False):

    divide = aft / bef
    if aft == bef:
        divide = 1

    if math_log:
        divide = multi * math.log(divide, math.e * e)
    else:
        divide = float(Decimal(str(multi)) * (Decimal(str(aft))/ Decimal(str(bef)) - Decimal(str(1))))

    if r:
        return round(divide, 4)
    else:
        return divide

def get_divide_dec_arr(bef, aft, multi=10000, math_log=False, e=0.1, r=False):
    return_arr = []
    for b, a in zip(bef, aft):
        return_arr.append(get_divide_dec(b, a, multi=multi, math_log=math_log, e=e, r=r))

    return np.array(return_arr)

def get_divide(bef, aft, multi=10000, math_log=False, e=0.1, r=False):

    divide = aft / bef
    if aft == bef:
        divide = 1

    if math_log:
        divide = multi * math.log(divide , math.e * e)
    else:
        divide = multi * (divide - 1)

    if r:
        return round(divide, 4)
    else:
        return divide

#ndarray用
get_divide_univ = np.frompyfunc(get_divide, 2, 1)

def get_divide_arr(bef, aft, multi=10000, math_log=False, e=0.1, r=False):
    return_arr = []
    for b, a in zip(bef, aft):
        return_arr.append(get_divide(b, a, multi=multi, math_log=math_log, e=e, r=r))

    return np.array(return_arr)

def get_sub(bef, aft, multi = 1):
    sub = float((Decimal(str(aft)) - Decimal(str(bef))) * Decimal(str(multi)))

    return sub

#ndarray用
get_sub_univ = np.frompyfunc(get_sub, 2, 1)

def get_sub_arr(bef, aft, multi = 1):
    return_arr = []
    for b, a in zip(bef, aft):
        return_arr.append(get_sub(b, a, multi=multi))

    return np.array(return_arr)

def get_rate(pred_list, bef_list, multi=10000):
    aft_list = bef_list * ((pred_list / multi) + 1)

    return aft_list


def get_rate_severe(pred_list, bef_list, multi=10000):
    tmp1 = Decimal(str(bef_list))
    tmp2 = Decimal(str(pred_list))
    aft_list = float(tmp1 * ((tmp2 / Decimal(str(multi))) + Decimal("1")))

    return aft_list


def get_rate_list(pred_list, bef_list):
    aft_list = []
    for pred, bef in zip(pred_list, bef_list):
        aft_list.append(get_rate_severe(pred, bef))

    return np.array(aft_list)

def get_satr(atr, close, multi=10000):
    tmp1 = Decimal(str(close))
    tmp2 = Decimal(str(atr))
    aft_list = float(tmp1 * ((tmp2 / Decimal(str(multi))) + Decimal("1")))

    return aft_list - close

def get_ask_bid(close, spr, pips):
    # sprはask,bidの差として入っているので、ask,bidを求めるために一旦半分にする
    tmp_spr = float(Decimal(str(spr)) / Decimal("2"))
    now_ask = float(Decimal(str(close)) + (Decimal(str(pips)) * (Decimal(str(tmp_spr)))))
    now_bid = float(Decimal(str(close)) - (Decimal(str(pips)) * (Decimal(str(tmp_spr)))))

    return [now_ask, now_bid]


def get_ask_bid_np(close, spr, pips):
    ask_list = []
    bid_list = []

    for cl, sp in zip(close, spr):
        # sprはask,bidの差として入っているので、ask,bidを求めるために一旦半分にする
        tmp_spr = float(Decimal(str(sp)) / Decimal("2"))
        ask_list.append(Decimal(str(cl)) + (Decimal(str(pips)) * Decimal(str(tmp_spr))))
        bid_list.append(Decimal(str(cl)) - (Decimal(str(pips)) * Decimal(str(tmp_spr))))

    return np.array(ask_list), np.array(bid_list)

def get_dir_cnt(dir_name):
    # ファイル数を出力
    return sum(os.path.isdir(os.path.join(dir_name, name)) for name in os.listdir(dir_name))

def rotate(input, n):
    return input[n:] + input[:n]


def makedirs(path):  # dirなければつくる
    if not os.path.isdir(path):
        os.makedirs(path)

def date_to_str(dt,format=None):
    if format == None:
        return dt.strftime('%Y%m%d')
    else:
        return dt.strftime(format)

def list_to_str(list, spl="-"):
    return_str = ""
    for i, v in enumerate(list):
        if i != 0:
            return_str = return_str + spl + str(v)
        else:
            return_str = str(v)
    return return_str


def make_db_list(symbol, db_term, bet_term):
    return_list = []
    if db_term >= bet_term:
        for i in range(int(get_decimal_divide(db_term, bet_term))):
            shift = get_decimal_sub(db_term, get_decimal_multi((i + 1), bet_term))
            return_list.append(symbol + "_" + str(db_term) + "_" + omit_zero_point_str(shift))
    else:
        return_list.append(symbol + "_" + str(db_term) + "_0")

    return return_list

# ロガー関数を返す(標準出力と/app/bin_op/log/app.logに出力 )
def printLog(logger):
    def f(*args):
        #print(*args)
        tmp_str = ""
        for i, a in enumerate(args):
            tmp_str = tmp_str + " " + str(a) if i != 0 else str(a)
        logger.info(tmp_str)
        print(tmp_str)

    return f

def output_log(file_name):
    def func(*args):
        with open(file_name, mode='a') as f:
            tmp_str = ""
            for i, a in enumerate(args):
                tmp_str = tmp_str + " " + str(a) if i !=0 else str(a)
            f.write(tmp_str)
            f.write('\n')
        print(tmp_str)

    return func

def tracebackPrint(e):
    return list(traceback.TracebackException.from_exception(e).format())

def standardization(x, ):
    x_mean = x.mean()
    x_std = x.std()
    return (x - x_mean) / x_std

def find_sunday(year, month, position):
    start = date(year, month, 1)
    day_delta = timedelta(days=1)
    counter = 0

    while True:
        if start.isoweekday() == 7: #7:SUNDAY
            counter += 1
        if counter == position:
            return start
        start += day_delta


def find_dst_begin(year):
    """
    DST starts the second sunday of March
    """
    return find_sunday(year, 3, 2)


def find_dst_end(year):
    """
    DST ends the first sunday of November
    """
    return find_sunday(year, 11, 1)

#サマータイムならTrue
#引数:datetime.date()
def is_dst(day):
    return day >= find_dst_begin(day.year) and day < find_dst_end(day.year)

#第何何曜日か返す
def get_weeknum(weekday, day):

    #1週間前の日付が同月かどうか調べる -> 1日より前か後かで判別
    #dayが1日以降(同月)なら出現回数+1してdayに1週間前の日付を代入(-7する)、1日より前(前の月の日付)なら処理終了
    weeks = 0
    while day > 0:
        weeks += 1
        day -= 7

    """
    例：今日が2017/11/13の場合、day = 13
    1ループ目・・・ day(=13) > 0 なので出現回数+1(weeks += 1)、1週間前の日付代入(day -= 7)
    2ループ目・・・ day(=6) > 0 なので出現回数+1(weeks += 1)、1週間前の日付代入(day -= 7)
    3ループ目・・・ day(=-1)は day > 0 を満たさないのでループを抜ける
    """

    #第一月曜なら0、第二火曜なら8。第五日曜である34まである想定
    #第一金曜日なら4
    return weekday + ((weeks - 1) * 7)

def get_lgbm_file_type(d):
    if d[:2] == "IF":
        return "/input_file/"
    elif d[:2] == "CF":
        return "/concat_file/"
    elif d[:2] == "MF":
        return "/merge_file/"
    elif d[:2] == "PF":
        return "/predict_file/"

#小数点一位が0の場合は0を省略した文字を返す
def omit_zero_point_str(d):
    if str(d)[-1] == "0":
        d = int(d)
    return str(d)
