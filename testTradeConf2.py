import os
from datetime import datetime
from datetime import timedelta
#from keras.utils.training_utils import multi_gpu_model
import time
os.environ["OMP_NUM_THREADS"] = "3"
#定数ファイル
host = "localhost"
#host = "amd3"

symbol = "GBPJPY"
symbols = [symbol]
#symbols = [symbol + "5", symbol + "10",symbol]

export_host = "local"

db_nos = {"opt":1,"snc":1,"noriyasu":11,"yoshiko":12,"local":8,"fil":14}

db_no = db_nos[export_host]

#取引制限ありと想定して予想させる
#この場合、必ずしも予想時に実際のトレードがされると限らないので、トレード実績を見たい場合はFalseにする
restrict_flg = False
restrict_term = 30

#THE OPTIONである
the_option_flg = False

#SONICである
the_sonic_flg = False

#TRB or SPR
type = "SPR"

#デモである
demo = False

#oandaのレートでトレードする場合
oanda_flg = False

#spreadPrevの値で計算する場合
spread_prev_flg = False
spread_key = "spread"
if spread_prev_flg:
    spread_key = "spreadPrev"

#2秒ごとの成績を計算する場合
per_sec_flg = True
#2秒ごとの成績を秒をキーとしてトレード回数と勝利数を保持
#{key:[trade_num,win_num]}
per_sec_dict = {}
per_sec_dict_real = {}
if per_sec_flg:
    for i in range(60):
        if i % 2 == 0:
            per_sec_dict[i] = [0,0]
            per_sec_dict_real[i] = [0, 0]

#除外するトレード秒を指定
#except_sec_list = [24,26,28,32]
except_sec_list = []

border = 0.565
border_up = 0.99

except_list = [20,21,22]
if(the_option_flg):
    except_list = [21,22]
if(the_sonic_flg):
    except_list = [19,20,21,22]

#スプレッドを除外するか
border_spread = [0.002,0.003,0.004,0.005]
#border_spread = [0.003,]
limit_border_flg = False

border_payout = 2.2
limit_payout_flg = False

spread = 1

#for usdjpy
#except_list = [1,8,10,11,12,17,18,19,20,21,22]

#for gbpjpy aus
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22]
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy opt
#except_list = [4,6,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy snc
#except_list = [3,4,6,7,8,9,10,11,13,14,15,16,17,20, 21, 22]

start = datetime(2020, 1, 1, 23)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2020, 12, 31, 22 )
end_stp = int(time.mktime(end.timetuple()))

maxlen = 600
pred_term = 15
s = "2"

trb_payout = {"30":950,"60":900,"180":900,"500":850}
spr_payout = {"30":1300,"60":1200,"180":1050,"500":1000}

if(the_option_flg):
    trb_payout = {"30": 830,}
    spr_payout = {"30": 1000, }

if(the_sonic_flg):
    spr_payout = {"30": 1020, }

db_no = db_nos[export_host]

if type == "TRB":
    payout = trb_payout[str(pred_term*int(s))]
elif type == "SPR":
    payout = spr_payout[str(pred_term * int(s))]
print("payout:",payout)

payoff = 1000

merg = ""
merg_file = ""
if merg != "":
    merg_file = "_merg_" + str(merg)

n_hidden = 60
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

drop = 0.0
in_num = 1
spread = 1

spread_list = {"spread0":(-1,0.000),"spread1":(0.000,0.001),"spread2":(0.001,0.002),"spread3":(0.002,0.003), "spread4":(0.003,0.004)
    ,"spread5":(0.004,0.005),"spread6":(0.005,0.006),"spread7":(0.006,0.007),"spread8":(0.007,0.008)
    , "spread10": (0.008, 0.010), "spread12": (0.010, 0.012), "spread14": (0.012, 0.014), "spread16": (0.014, 0.016),"spread16Over":(0.016,1),}

drawdown_list = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),"drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),"drawdown9over": (-90000, -1000000),}

#db_suffix_trade_list = {"ubuntu1":"","ubuntu2":"","ubuntu3":"_OPT","ubuntu4":"","ubuntu4-2":"","ubuntu5":"","ubuntu18":""}

#db_suffixs = (1,2,3,4,5)
#db_suffixs = ("",)
#db_suffix = db_suffix_trade_list[export_host]

if demo:
    db_suffix = "_DEMO"
else:
    db_suffix = ""

#db_key = symbol + "_" + str(int(s) * pred_term)  + "_" + type
db_key = symbol + "_" + str(int(s) * pred_term)  + "_SPR"

if(the_option_flg):
    db_key = db_key + "_OPT"
if(the_sonic_flg):
    db_key = db_key + "_SNC"

db_key_trade = db_key + "_TRADE"
print("db_key: " + db_key)


model_dir = "/app/bin_op/model"
gpu_count = 2
batch_size = 2048* gpu_count

except_index = False
except_highlow = True

#process_count = multiprocessing.cpu_count() - 1
process_count = 1
askbid = "_bid"

default_money = 0

current_dir = os.path.dirname(__file__)

file_prefix = "GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5.90*17"
#file_prefix ="GBPJPY_bydrop_in1_2_m400_term_30_hid1_40_hid2_0_hid3_0_hid4_0_drop_0.0_bid.hdf5.70*8"

history_file = os.path.join(current_dir, "history", file_prefix + "_history.csv")
model_file = os.path.join(model_dir, file_prefix)
print("Model is ", model_file)

if os.path.isfile(model_file) == False:
    print("the Model not exists!")

#トレード失敗した日を除外するか
except_fail_flg = False

#amd1でトレードエラーとなった日のリスト
#after 2020/04/17
amd1_fail_list = [
"2020/04/26",
"2020/04/29",
"2020/05/10",
"2020/05/24",
"2020/06/22",
"2020/07/02",
"2020/08/12",
]

#amd3でトレードエラーとなった日のリスト
amd3_fail_list = [
"2020/05/07",
"2020/05/10",
"2020/05/11",
"2020/05/24",
"2020/06/22",
"2020/07/02",
"2020/08/12",
"2020/08/13",
"2020/08/16",
]

#amd1,3でトレードエラーとなった日のリスト merge
fail_list = [
"2020/04/26",
"2020/04/29",
"2020/05/07",
"2020/05/10",
"2020/05/11",
"2020/05/24",
"2020/06/22",
"2020/07/02",
"2020/08/12",
"2020/08/13",
"2020/08/16",
]

fail_list_score=[]
#除外する日の開始スコアと終了スコアを追加していく
if len(fail_list) !=0 :
    for j in fail_list:
        tmp_str = j.split("/")
        tmp_date_start = datetime(int(tmp_str[0]), int(tmp_str[1]), int(tmp_str[2]), 23 )
        tmp_date_end = tmp_date_start + timedelta(hours=21)
        fail_list_score.append([int(time.mktime(tmp_date_start.timetuple())), int(time.mktime(tmp_date_end.timetuple())) ])
"""
for k in amd_fail_list_score:
    print(k[0], k[1])
"""