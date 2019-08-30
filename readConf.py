import os
import logging.config
#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

symbol = "EURJPY"
#symbol = "GBPUSD"
#symbols = [symbol, symbol + "1"]
symbols = [symbol]
"""
symbols = [symbol,
           symbol + "10", symbol + "20",symbol + "30", symbol + "40",symbol + "50", symbol + "60",
           symbol + "70", symbol + "80",symbol + "90", symbol + "100",
           symbol + "110"]
"""

maxlen = 400
pred_term = 15
s = "2"

merg = ""
merg_file = ""
if merg != "":
    merg_file = "_merg_" + str(merg)

n_hidden ={
    1: 40,
    2: 0,
    3: 0,
    4: 0,

}

hidden = ""
for k, v in sorted(n_hidden.items()):
    hidden = hidden + "_hid" + str(k) + "_" + str(v)

drop = 0.0

in_num = 1

spread = 1
#spread = 3

suffix = ""
db_suffix = ""

payout = 950
payoff = 1000

fx = False
fx_position = 10000
db_no = 3
#学習対象外スプレッドを設けるか
except_low_spread = False
limit_spread = 0.00008

except_index = False

#学習対象外時間を設けるか
except_highlow = True

#学習対象外時間(ハイローがやっていない時間)
except_list = [20,21,22]

#学習対象外時間(the optionがやっていない時間)
#except_list = [21,22,23]

#for usdjpy FX
#except_list = [4,5,9,12,16,18,19,20,21,22]
#except_list = [0,4,5,9,10,11,12,14,15,16,17,18,19,20,21,22]
#except_list = [3,7,9,10,11,12,13,17,19,20,21,22]

#for usdjpy
#except_list = [1,8,10,11,12,17,18,19,20,21,22]
#except_list = [0,1,4,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22]

#for gbpjpy aus
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22]

#for gbpjpy opt
#except_list = [4,6,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy snc
#except_list = [3,4,6,7,8,9,10,11,13,14,15,16,17,20, 21, 22]

spread_list = {"spread0":(-1,0.00000),"spread2":(0.00000,0.00002), "spread4":(0.00002,0.00004),"spread6":(0.00004,0.00006),"spread8":(0.00006,0.00008)
    , "spread10": (0.00008, 0.00010), "spread12": (0.00010, 0.00012), "spread14": (0.00012, 0.00014), "spread16": (0.00014, 0.00016),"spread16Over":(0.00016,1),}

drawdown_list = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),"drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),"drawdown9over": (-90000, -1000000),}

model_dir = "/app/bin_op/model"
gpu_count = 2
batch_size = 1024 * 8 * gpu_count
#process_count = multiprocessing.cpu_count() - 1
process_count = 1
askbid = "_bid"
type = "category"
"""

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop)  + askbid + merg_file
"""
file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + hidden + "_drop_" + str(drop)  + askbid + merg_file

history_file = os.path.join(current_dir, "history", file_prefix + "_history.csv")
model_file = os.path.join(model_dir, file_prefix + ".hdf5" + suffix)

def printLog(logger):
    def f(*args):
        print(*args)
        fmt =""
        for i, j in enumerate(args):
            fmt = fmt + "{" + str(i) + "} "
        logger.info(fmt.format(*args))

    return f

myLogger = printLog(loggerConf)

myLogger("Model is " , model_file)

#print("Model is ", model_file)
