import os
import logging.config
from decimal import Decimal

#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

symbol = "GBPJPY"

symbols = [symbol]
"""
symbols = [symbol,
           symbol + "2", symbol + "4",symbol + "6", symbol + "8",
           ]
"""

# 学習方法 Bidirectionalならby
# LSTMならlstm
#method = "lstm"
method = "lstm"

# Functional API
functional_flg = True

functional_str = ""
if functional_flg:
    functional_str = "_func"

maxlen = 400
maxlen_min = 200

pred_term = 15
# 学習データの間隔(秒)
s = "2"

# 何秒単位でトレードするか
merg = "2"

merg_file = ""
if merg != "":
    merg_file = "_merg_" + merg

# DB内データの秒間隔
db = "2"

# db内データの秒間隔と、学習データの間隔が異る場合のcloseデータをずらす間隔
# 例えば,30秒予想でDBが2秒間隔で学習データの間隔を10秒とするなら
# s=10,merg=2でDBデータを5個ずらしてその変化率を学習データとする。
# だが、DBが1秒間隔で学習データの間隔も1秒だが、トレードタイミングであるmerg=2で,mergの方が大きい数字の場合、
# ずらす必要はないため、close_shiftは1とし、検証時(testLstm.py)は秒をmergで割った余りが0のデータだけを使って結果をみる
if s!=db:
    if int(s) >= int(merg):
        close_shift = int(Decimal(s) / Decimal(merg))
    else:
        close_shift = 1
else:
    close_shift = 1
print("close_shift:" + str(close_shift))

# 学習時、close_shiftが1より大きいならデータ作成時間の秒を学習データの間隔sで割った余りがdata_setの値のものだけつかうようにする
# またclose_shiftがiならデータ作成時間の秒をトレード間隔mergで割った余りがdata_setの値のものだけつかうようにする
# 何も設定されていなければ全てのセットを使用する
#
data_set = []
data_set_str = "_set_ALL"
if len(data_set) != 0:
    data_set_str = "_set"
    for set in data_set:
        data_set_str = data_set_str + "_" + str(set)

n_hidden ={
    1: 40,
    2: 0,
    3: 0,
    4: 0,
}
dense_hidden ={
    1: 0,
    2: 0,
}

min_hidden = 20

min_hidden_str = ""
if min_hidden != "":
    min_hidden_str = "_mhid_" + str(min_hidden)

hidden = ""
d_hidden = ""

for k, v in sorted(n_hidden.items()):
    hidden = hidden + "_hid" + str(k) + "_" + str(v)

for k, v in sorted(dense_hidden.items()):
    if v !=0:
        d_hidden = d_hidden + "_hid" + str(k) + "_" + str(v)

drop = 0.0
#特徴量の種類 close_divide:closeの変化率
in_features = ["close_divide",]

in_features_str = ""
for feature in in_features:
    if in_features_str == "":
        in_features_str = feature
    else:
        in_features_str = in_features_str + "-" + feature

#使用するより大きい足のデータ
#例:2秒足データに加えて1分足のデータも使用する
in_longers = ["min_score",]
#in_longers = []

in_longers_str = ""
for longer in in_longers:
    if in_longers_str == "":
        in_longers_str = "_" + longer
    else:
        in_longers_str = in_longers_str + "-" + longer

in_longers_db = {"min_score":"GBPJPY_M1",}

#インプットの特徴量の種類数
x_length = len(in_features) + len(in_longers)

if functional_flg:
    x_length = len(in_features)

spread = 1
#spread = 3

suffix = ""
db_suffix = ""

payout = 1000
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
gpu_count = 3
batch_size = 1024 * 8 * gpu_count
#process_count = multiprocessing.cpu_count() - 1
process_count = 1
askbid = "_bid"
type = "category"

#file_prefix = symbol + "_" + method + "drop_in" + str(len(in_features)) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + hidden + "_drop_" + str(drop)  + askbid + merg_file + data_set_str
file_prefix = symbol+ "_" + method + functional_str + "_" + in_features_str + in_longers_str + "_" + s + "_m" + str(maxlen) + "(" + str(maxlen_min) + ")" + "_term_" + str(pred_term * int(s)) + hidden + d_hidden + min_hidden_str + "_drop_" + str(drop)  + askbid + merg_file + data_set_str


history_file = os.path.join(current_dir, "history", file_prefix + "_history.csv")
model_file = os.path.join(model_dir, file_prefix + ".hdf5" + suffix)

#ロガー関数を返す(標準出力と/app/bin_op/log/app.logに出力 )
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
