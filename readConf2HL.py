import os
import logging.config
from decimal import Decimal

def makedirs(path):#dirなければつくる
    if not os.path.isdir(path):
        os.makedirs(path)

#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

SYMBOL = "USDJPY"

# betする間隔(秒)
BET_TERM = 2

# 予測する間隔:DB1_TERMが2でPRED_TERMが15なら30秒後の予測をする
PRED_TERM = 15

DB_NO = 3 #dukascopy data
#DB_EVAL_NO = 0 #highlow data(GBPJPY)/oanda data(USDJPY)
DB_EVAL_NO = 1 #dukascopy data(GBPJPY)/dukascopy data(USDJPY)
#DB_EVAL_NO = 2 #oanda data(GBPJPY)


# Fake用DB 長い足を使わない場合とのテスト結果比較のため
# 長い足を特徴量とする場合、テストデータが少なくなって、長い足を使わない場合とのテスト結果が正確ではなくなるため
# 使用しない場合 0にする
#DB_FAKE_TERM = 300
DB_FAKE_TERM = 0
DB_FAKE_LIST = []
DB_FAKE_INPUT_LEN = 24

if DB_FAKE_TERM != 0:
    for i in range(int(Decimal(str(DB_FAKE_TERM)) / Decimal(str(BET_TERM)))):
        DB_FAKE_LIST.append(SYMBOL + "_" + str(DB_FAKE_TERM) + "_" +  str(DB_FAKE_TERM - ((i + 1) * BET_TERM)))

print(DB_FAKE_LIST)

# inputするデータの秒間隔
DB1_TERM = 2
DB1_LIST = []

# 読み込み対象のDB名を入れていく
# 例:db_termが60でsが2なら60秒間隔でデータ作成していることは変わらないが
# 位相が"01:02,02:02,03:02..." と"01:00,02:00,03:00..."で異なっている
if DB1_TERM >= BET_TERM:
    for i in range(int(Decimal(str(DB1_TERM)) / Decimal(str(BET_TERM)))):
        DB1_LIST.append(SYMBOL + "_" + str(DB1_TERM) + "_" +  str(DB1_TERM - ((i + 1) * BET_TERM)))

else:
    DB1_LIST.append(SYMBOL + "_" + str(DB1_TERM) + "_0")

print(DB1_LIST)

# inputする長い足のデータ秒間隔
## 使用しない場合 0にする
DB2_TERM = 10
#DB2_TERM = 0
DB2_LIST = []

if DB2_TERM != 0:
    for i in range(int(Decimal(str(DB2_TERM)) / Decimal(str(BET_TERM)))):
        DB2_LIST.append(SYMBOL + "_" + str(DB2_TERM) + "_" +  str(DB2_TERM - ((i + 1) * BET_TERM)))

print(DB2_LIST)

# inputする長い足のデータ秒間隔
# 使用しない場合 0にする
DB3_TERM = 30
#DB3_TERM = 0
DB3_LIST = []

if DB3_TERM != 0:
    for i in range(int(Decimal(str(DB3_TERM)) / Decimal(str(BET_TERM)))):
        DB3_LIST.append(SYMBOL + "_" + str(DB3_TERM) + "_" +  str(DB3_TERM - ((i + 1) * BET_TERM)))

print(DB3_LIST)

DB4_TERM = 90
#DB4_TERM = 0
DB4_LIST = []

if DB4_TERM != 0:
    for i in range(int(Decimal(str(DB4_TERM)) / Decimal(str(BET_TERM)))):
        DB4_LIST.append(SYMBOL + "_" + str(DB4_TERM) + "_" +  str(DB4_TERM - ((i + 1) * BET_TERM)))

print(DB4_LIST)

DB5_TERM = 300
#DB5_TERM = 0
DB5_LIST = []

if DB5_TERM != 0:
    for i in range(int(Decimal(str(DB5_TERM)) / Decimal(str(BET_TERM)))):
        DB5_LIST.append(SYMBOL + "_" + str(DB5_TERM) + "_" +  str(DB5_TERM - ((i + 1) * BET_TERM)))

print(DB5_LIST)

#DB6_TERM = 4
DB6_TERM = 0
DB6_LIST = []

if DB6_TERM != 0:
    for i in range(int(Decimal(str(DB6_TERM)) / Decimal(str(BET_TERM)))):
        DB6_LIST.append(SYMBOL + "_" + str(DB6_TERM) + "_" +  str(DB6_TERM - ((i + 1) * BET_TERM)))

print(DB6_LIST)

#DB7_TERM = 200
DB7_TERM = 0
DB7_LIST = []

if DB7_TERM != 0:
    for i in range(int(Decimal(str(DB7_TERM)) / Decimal(str(BET_TERM)))):
        DB7_LIST.append(SYMBOL + "_" + str(DB7_TERM) + "_" +  str(DB7_TERM - ((i + 1) * BET_TERM)))

print(DB7_LIST)

DB_TERMS = [
    DB1_TERM,
    DB2_TERM,
    DB3_TERM,
    DB4_TERM,
    DB5_TERM,
    DB6_TERM,
    DB7_TERM,
]

DB_TERM_STR = ""

for i, v in enumerate(DB_TERMS):
    print(v)
    if i !=0 and v != 0:

        DB_TERM_STR = DB_TERM_STR + "-" + str(v)
    elif i == 0 and v != 0:
        DB_TERM_STR = str(v)

INPUT_LEN = [
    300,
    300,
    240,
    80,
    24,

    ]

INPUT_LEN_STR = ""

for i, v in enumerate(INPUT_LEN):
    if i !=0:
        INPUT_LEN_STR = INPUT_LEN_STR + "-" + str(v)
    else:
        INPUT_LEN_STR = str(v)

LSTM_UNIT =[
    30,
    30,
    24,
    8,
    4,



]
LSTM_UNIT_STR = ""

for i, v in enumerate(LSTM_UNIT):
    if i !=0:
        LSTM_UNIT_STR = LSTM_UNIT_STR + "-" + str(v)
    else:
        LSTM_UNIT_STR = str(v)

#設定確認
if len(INPUT_LEN) == 2:
    if DB2_TERM == 0:
        print("SHOULD CONFIG DB2_TERM !!!")
        exit(1)
    if len(LSTM_UNIT) < 2:
        print("SHOULD CONFIG LSTM_UNIT 2 !!!")
        exit(1)
elif len(INPUT_LEN) == 3:
    if DB2_TERM == 0:
        print("SHOULD CONFIG DB2_TERM !!!")
        exit(1)
    if DB3_TERM == 0:
        print("SHOULD CONFIG DB3_TERM !!!")
        exit(1)
    if len(LSTM_UNIT) < 3:
        print("SHOULD CONFIG LSTM_UNIT 3 !!!")
        exit(1)

elif len(INPUT_LEN) == 4:
    if DB2_TERM == 0:
        print("SHOULD CONFIG DB2_TERM !!!")
        exit(1)
    if DB3_TERM == 0:
        print("SHOULD CONFIG DB3_TERM !!!")
        exit(1)
    if DB4_TERM == 0:
        print("SHOULD CONFIG DB4_TERM !!!")
        exit(1)
    if len(LSTM_UNIT) < 4:
        print("SHOULD CONFIG LSTM_UNIT 4 !!!")
        exit(1)
elif len(INPUT_LEN) == 5:
    if DB2_TERM == 0:
        print("SHOULD CONFIG DB2_TERM !!!")
        exit(1)
    if DB3_TERM == 0:
        print("SHOULD CONFIG DB3_TERM !!!")
        exit(1)
    if DB4_TERM == 0:
        print("SHOULD CONFIG DB4_TERM !!!")
        exit(1)
    if DB5_TERM == 0:
        print("SHOULD CONFIG DB5_TERM !!!")
        exit(1)
    if len(LSTM_UNIT) < 5:
        print("SHOULD CONFIG LSTM_UNIT 5 !!!")
        exit(1)

DENSE_UNIT =[
    16,
    8,
    4,

]
DENSE_UNIT_STR = ""

for i, v in enumerate(DENSE_UNIT):
    if i !=0:
        DENSE_UNIT_STR = DENSE_UNIT_STR + "-" + str(v)
    else:
        DENSE_UNIT_STR = str(v)

DROP = 0.0
#ls正則化

L_K_RATE = 0 #lstm kernel_regularizer
L_R_RATE = 0 #lstm recurrent_regularizer
#L_K_RATE = 0.00001
#L_R_RATE = 0.00001
L_D_RATE = 0 #dense kernel_regularizer


#DIVIDE_MAX = 5 #0.1%を外れ値として除外
#DIVIDE_MAX = 4.1 #0なら制限なし Categoryの場合、外れ値はあまり関係ないので0として全て学習対象とする 100なら1%変化したということ
DIVIDE_MAX = 0
DIVIDE_MIN = 0

FX = False
FX_STR = ""
if FX :
    FX_STR = "_FX"

FX_REAL_SPREAD_FLG = False #Oandaのスプレッドデータを使用する場合

FX_POSITION = 15000 #20秒なら15000, 30秒なら10000, 60秒なら5000
FX_STOP_LOSS = -0.013 #FX_REAL_SPREAD_FLG==TrueならFalseの時のFX_STOP_LOSSよりFX_SPREAD分マイナスする
FX_MORE_BORDER_RGR = 0.0
FX_MORE_BORDER_CAT = 0
START_MONEY = 1000000
FX_TARGET_SPREAD_LIST = [0,1,2,3,4,5,6,7,8,9,10]

#本番データのスプレッドを加味したテストをする場合
#正解を出すときに使うSPREADに実データのスプレッドを使用する
REAL_SPREAD_FLG = False

#eval時に本番データを使う場合
REAL_SPREAD_EVAL_FLG = False

#実トレードした結果を加味する場合
TRADE_FLG = False

#スプレッドにより除外する場合
EXCEPT_SPREAD_FLG = False
#除外しないスプレッド
#TARGET_SPREAD_LIST = [0,1,2,3,4,5,6,7,8,9,]
TARGET_SPREAD_LIST = [1]

#除外するトレード秒を指定
EXCEPT_SEC_LIST = []
#EXCEPT_SEC_LIST = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,]
#EXCEPT_SEC_LIST = [32,34,36,38,40,42,44,46,48,50,52,54,56,58,0,]

FORMER_LIST = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,]
LATTER_LIST = [32,34,36,38,40,42,44,46,48,50,52,54,56,58,0,]

#除外するトレード分を指定
EXCEPT_MIN_LIST = []
#EXCEPT_MIN_LIST = [57]

#スプレッドごとの対象外の時間
EXCEPT_LIST_BY_SPERAD = {}
#EXCEPT_LIST_BY_SPERAD = {1:[0,7,8,9,10,11,12,13,14,15,16,17],}

EXCEPT_DIV_LIST = []
#EXCEPT_DIV_LIST = ["-0.0", "0.0"]

#予想divideの上限値、これより大きい場合は取引しない
EXCEPT_DIVIDE_MAX = 0

DB_TRADE_NO = 4
DB_TRADE_NAME = "GBPJPY_30_SPR_TRADE"

#PAYOUT = 1300 #30秒 spread
#PAYOUT = 1200 #60秒
PAYOUT = 1000 #30秒 trubo

PAYOFF = 1000

#学習対象外時間(ハイローがやっていない時間)

#FXの場合 スプレッドを見る限り
#開始時間については
#0時0分(マーケット開始から2:00経過)から取引するのがよさそう(サマータイムでない時期)
#23時0分(マーケット開始から2:00経過)から取引するのがよさそう(サマータイム時期)

#終了時間については
#21時までがよさそう(サマータイムでない時期)
#20時までがよさそう(サマータイム時期)

EXCEPT_LIST = [20,21,22] #highlow
#EXCEPT_LIST = [20,21,22,23,0,1] #3時間データ蓄積

if FX:
    EXCEPT_LIST = [20, 21, 22, 23]  # FX

DRAWDOWN_LIST = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),
                 "drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),
                 "drawdown9over": (-90000, -1000000),}

SPREAD_LIST = {"spread0":(-1,0),"spread1":(0,1),"spread2":(1,2),"spread3":(2,3), "spread4":(3,4)
    ,"spread5":(4,5),"spread6":(5,6),"spread7":(6,7),"spread8":(7,8),"spread9":(8,9)
    , "spread10": (9, 10), "spread11": (10, 11),"spread12": (11, 12),"spread13": (12, 13), "spread14": (13, 14),"spread15": (14, 15), "spread16": (15, 16),"spread16Over":(16,1000),}


DIVIDE_BEF_LIST = {"divide0":(-1,0),"divide0.1":(0,0.1),"divide0.2":(0.1,0.2),"divide0.3":(0.2,0.3),"divide0.4":(0.3,0.4),"divide0.5":(0.4,0.5),
                    "divide0.5over": (0.5, 10000),
               }

DIVIDE_PREV_LENGTH = 15

DIVIDE_LIST = {"divide0.01":(-1,0.01),"divide0.02":(0.01,0.02),"divide0.03":(0.02,0.03),"divide0.04":(0.03,0.04), "divide0.05":(0.04,0.05)
    ,"divide0.06":(0.05,0.06),"divide0.07":(0.06,0.07),"divide0.08":(0.07,0.08),"divide0.09":(0.08,0.09),
"divide0.1":(0.09,0.1),"divide0.2":(0.1,0.2),"divide0.3":(0.2,0.3),"divide0.4":(0.3,0.4), "divide0.5":(0.4,0.5)
    ,"divide0.6":(0.5,0.6),"divide0.7":(0.6,0.7),"divide0.8":(0.7,0.8),"divide0.9":(0.8,0.9)
    , "divide1.0": (0.9, 1.0), "divide1.1": (1.0, 1.1), "divide1.2": (1.1, 1.2), "divide1.3": (1.2, 1.3),"divide1.4":(1.3,1.4)
    , "divide1.5": (1.4, 1.5), "divide1.6": (1.5, 1.6),"divide1.7":(1.6,1.7),"divide1.8":(1.7,1.8),"divide1.9":(1.8,1.9),"divide2.0":(1.9,2.0),"divide2.0over": (2.0, 10000),
               }

DIVIDE_AFT_LIST = {"divide0.5":(-1,0.5),"divide1":(0.5,1.0),"divide2":(1,2),"divide3":(2,3), "divide4":(3,4)
    ,"divide5":(4,5),"divide6":(5,6),"divide7":(6,7),"divide8":(7,8),"divide9":(8,9),"divide10":(9,10)
    , "divide11": (10, 11),"divide12":(11,12),"divide13":(12,13),"divide13over": (13, 10000),
               }

# 学習方法 Bidirectionalならby
# LSTMならlstm
METHOD = "LSTM"
#METHOD = "SimpleRNN" #1エポックの学習時間がLSTMの約7倍も掛かり、精度も劣るので使わない
#METHOD = "GRU" #1エポックの学習時間がかかりすぎる

#METHOD = "LSTM2" #LSTM予想を参考にさらに予想させる

#METHOD = "BY"
#METHOD = "NORMAL"


#直前のレートから最古のレートまでのdivideを特徴量とする
DIVIDE_ALL_FLG = False

#divideではなく、直接レートを予想する
#そのため、特徴量はclose値とする
DIRECT_FLG = False

#入力に予想時のレートをくわえる
NOW_RATE_FLG = False

GPU_COUNT = 2
BATCH_SIZE = 1024 * 1 * GPU_COUNT
#process_count = multiprocessing.cpu_count() - 1
PROCESS_COUNT = 2

#LEARNING_TYPE = "CATEGORY" #多クラス分類
#LEARNING_TYPE = "CATEGORY_BIN_UP" #2クラス分類
#LEARNING_TYPE = "CATEGORY_BIN_DW" #2クラス分類
LEARNING_TYPE = "CATEGORY_BIN_BOTH" #2クラス分類 UP,DOWN両方を使ってテストする用　学習するためではない
#LEARNING_TYPE = "CATEGORY_BIN_FOUR" #2クラス分類 UP,DOWNそれぞれを1分の前半、後半の秒を使って学習させたモデルを使ってテストする用　学習するためではない
#LEARNING_TYPE = "REGRESSION_SIGMA" #回帰 sigma
#LEARNING_TYPE = "REGRESSION" #回帰

#Category 予測において　SPREADではなく変化率で分類する場合
#レートが153.846153846154(=0.001/0.0000065)のとき、0.001円上がるとDivideは0.065
#※Divie＝((X_after/X_before) -1) * 10000

BORDER_DIV = 0
#BORDER_DIV = 0.065 #(spread=1)
#BORDER_DIV = 0.130 #(spread=2)
#BORDER_DIV = 0.195 #(spread=3)
#BORDER_DIV = 0.260 #(spread=4)
#BORDER_DIV = 0.325 #(spread=5)
#BORDER_DIV = 0.390 #(spread=6)

SPREAD = 1

FX_SPREAD = 10

BORDER_STR = ""
if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_UP" or LEARNING_TYPE == "CATEGORY_BIN_DW":
    BORDER_STR = "_SPREAD" + str(SPREAD)
    if BORDER_DIV != 0:
        BORDER_STR = "_BDIV" + str(BORDER_DIV)
else:
    #regressionの場合
    SPREAD = 1

OUTPUT = 0
if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
    OUTPUT = 3
elif (LEARNING_TYPE == "CATEGORY_BIN_UP" or LEARNING_TYPE == "CATEGORY_BIN_DW" ):
    OUTPUT = 2
elif LEARNING_TYPE == "REGRESSION_SIGMA":
    OUTPUT = 2 # 平均, β(=logα α=標準偏差) #1つ目がmuで2つ目がbeta(精度パラメーターとする)
elif LEARNING_TYPE == "REGRESSION":
    OUTPUT = 1


LOSS_TYPE = "C-ENTROPY"
#LOSS_TYPE = "B-ENTROPY"

#LOSS_TYPE = "MSE"
#LOSS_TYPE = "HUBER"

#LOSS_TYPE = "LOG_COSH"
#LOSS_TYPE = "MAE"
#LOSS_TYPE = "HINGE"
#LOSS_TYPE = "SQUARED_HINGE"
#LOSS_TYPE = "POISSON"

R_I = 'glorot_uniform' #RNN系の初期値
D_I = 'glorot_uniform' #DENSE系の初期値
O_I = 'glorot_uniform' #OUTPUT系の初期値
INIT_STR = "_GU-GU-GU" #初期値の設定(R_I,D_I,O_I)によってモデル名のsuffixを変える

SUFFIX = "_UB1_202101_90"
#SUFFIX = "_UB1_202101_90_EXCSEC-2-30"
#SUFFIX = "_UB1_202101_90_EXCSEC-32-0"

#変化率の対数をとる
DIVIDE_LOG_FLG = False
if DIVIDE_LOG_FLG:
    SUFFIX += "_DLOG"

BATCH_NORMAL = False
BATCH_NORMAL_STR = ""
if BATCH_NORMAL:
    BATCH_NORMAL_STR = "_BNORMAL"

if DIVIDE_ALL_FLG:
    SUFFIX += "DIVIDE_ALL"


if METHOD == "LSTM2":
    SUFFIX += "_GEN1" #gen1はLSTM2の1世代目(2ならLSTM2の結果をさらに受けて予想する)

if DIRECT_FLG:
    SUFFIX += "_DIRECT"

if NOW_RATE_FLG:
    SUFFIX += "_NRATE"

if DB_FAKE_TERM != 0:
    SUFFIX += "_DBFAKE-" + str(DB_FAKE_TERM) + "-" + str(DB_FAKE_INPUT_LEN)



FILE_PREFIX = ""

L_D_STR = ""
if L_D_RATE != 0:
    L_D_STR = "_L-R" + str(L_D_RATE)

DIVIDE_MIN_STR = ""
if DIVIDE_MIN != 0:
    DIVIDE_MIN_STR = "_DIVIDEMIN" + str(DIVIDE_MIN)

EPOCH = 40

#LEARNING_RATE = 0.001
LEARNING_RATE = 0.001

#試験的にモデルつくる際につける目印
SUFFIX += "_L-RATE" + str(LEARNING_RATE)

TERM = PRED_TERM * DB1_TERM

if METHOD == "LSTM" or METHOD == "BY" or METHOD == "LSTM2" or METHOD == "SimpleRNN"  or METHOD == "RNN"  or METHOD == "GRU" :
    FILE_PREFIX = SYMBOL + "_" + LEARNING_TYPE + "_" + METHOD + "_BET" + str(BET_TERM) + "_TERM" + str(TERM) + \
                  "_INPUT" + DB_TERM_STR + \
                  "_INPUT_LEN" + INPUT_LEN_STR + \
                  "_L-UNIT" + LSTM_UNIT_STR + "_D-UNIT" + DENSE_UNIT_STR + "_DROP" + str(DROP) + \
                  "_L-K" + str(L_K_RATE) + "_L-R" + str(L_R_RATE) + L_D_STR + BATCH_NORMAL_STR + "_DIVIDEMAX" + str(DIVIDE_MAX) + DIVIDE_MIN_STR + \
                  BORDER_STR + SUFFIX + "_LOSS-" + LOSS_TYPE + INIT_STR + FX_STR
elif METHOD == "NORMAL":
    FILE_PREFIX = SYMBOL + "_" + LEARNING_TYPE + "_" + METHOD + "_BET" + str(BET_TERM) + "_TERM" + str(TERM) + \
                  "_INPUT" + DB_TERM_STR + \
                  "_INPUT_LEN" + INPUT_LEN_STR + \
                  "_D-UNIT" + DENSE_UNIT_STR + "_DROP" + str(DROP) + \
                  "_L-K" + str(L_K_RATE) + "_L-R" + str(L_R_RATE) + L_D_STR + BATCH_NORMAL_STR + "_DIVIDEMAX" + str(DIVIDE_MAX) + DIVIDE_MIN_STR + \
                  BORDER_STR + SUFFIX + "_LOSS-" + LOSS_TYPE + INIT_STR + FX_STR

#CPUのみ、またはGPU1つしか搭載していない場合
SINGLE_FLG = False
#SINGLE_FLG = Trueのとき、学習するデバイスを指定
#0 :RTX3090
#1 :RTX3080
DEVICE = "1"

# 0:新規作成
# 1:modelからロード
# 2:chekpointからロード
LOAD_TYPE = 0
LOADING_NUM = "40"

# 1つのモデルに対して実行した学習回数
LEARNING_NUM = "40"

# 保存用のディレクトリ
MODEL_DIR = "/app/model/bin_op/" + FILE_PREFIX + "-" + LEARNING_NUM
HISTORY_DIR = "/app/history/bin_op/" + FILE_PREFIX + "-" + LEARNING_NUM
CHK_DIR = "/app/chk/bin_op/" + FILE_PREFIX + "-" + LEARNING_NUM

# Load用のディレクトリ
MODEL_DIR_LOAD = "/app/model/bin_op/" + FILE_PREFIX + "-" + LOADING_NUM
CHK_DIR_LOAD = "/app/chk/bin_op/" + FILE_PREFIX + "-" + LOADING_NUM

HISTORY_PATH = os.path.join(HISTORY_DIR, FILE_PREFIX)

LOAD_CHK_NUM = "0009"
LOAD_CHK_PATH = os.path.join(CHK_DIR_LOAD, LOAD_CHK_NUM)

def print_load_info():
    if LOAD_TYPE == 0:
        print("新規作成")
    elif LOAD_TYPE == 1:
        print("modelからロード")
        print("LOADING_NUM:", LOADING_NUM)
    elif LOAD_TYPE == 2:
        print("chekpointからロード")
        print("LOAD_CHK_NUM:", LOAD_CHK_NUM)

    print("Model is " , FILE_PREFIX)


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



