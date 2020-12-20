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

SYMBOL = "GBPJPY"

# betする間隔(秒)
BET_TERM = 2

# 予測する間隔:Sが2でPRED_TERMが15なら30秒後の予測をする
PRED_TERM = 15

DB_NO = 3

# inputするデータの秒間隔
DB1_TERM = 2
DB1_LIST = []

# 読み込み対象のDB名を入れていく
# 例:db_termが60でsが2なら60秒間隔でデータ作成していることは変わらないが
# 位相が"01:02,02:02,03:02..." と"01:00,02:00,03:00..."で異なっている
for i in range(int(Decimal(str(DB1_TERM)) / Decimal(str(BET_TERM)))):
    DB1_LIST.append(SYMBOL + "_" + str(DB1_TERM) + "_" +  str(DB1_TERM - ((i + 1) * BET_TERM)))

print(DB1_LIST)

# inputする長い足のデータ秒間隔
# 使用しない場合 0にする
DB2_TERM = 10
#DB2_TERM = 0
DB2_LIST = []

if DB2_TERM != 0:
    for i in range(int(Decimal(str(DB2_TERM)) / Decimal(str(BET_TERM)))):
        DB2_LIST.append(SYMBOL + "_" + str(DB2_TERM) + "_" +  str(DB2_TERM - ((i + 1) * BET_TERM)))

print(DB2_LIST)

# inputする長い足のデータ秒間隔
# 使用しない場合 0にする
#DB3_TERM = 30
DB3_TERM = 0
DB3_LIST = []

if DB3_TERM != 0:
    for i in range(int(Decimal(str(DB3_TERM)) / Decimal(str(BET_TERM)))):
        DB3_LIST.append(SYMBOL + "_" + str(DB3_TERM) + "_" +  str(DB3_TERM - ((i + 1) * BET_TERM)))

print(DB3_LIST)

# 学習方法 Bidirectionalならby
# LSTMならlstm
METHOD = "LSTM"

INPUT_LEN = [
    300,
    300,
    #100,
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
    #20,
]
LSTM_UNIT_STR = ""

for i, v in enumerate(LSTM_UNIT):
    if i !=0:
        LSTM_UNIT_STR = LSTM_UNIT_STR + "-" + str(v)
    else:
        LSTM_UNIT_STR = str(v)

DENSE_UNIT =[
    #20,
    #10,
    #5,
]
DENSE_UNIT_STR = ""

for i, v in enumerate(DENSE_UNIT):
    if i !=0:
        DENSE_UNIT_STR = DENSE_UNIT_STR + "-" + str(v)
    else:
        DENSE_UNIT_STR = str(v)

DROP = 0.0
#ls正則化
L2_RATE = 0.01

DIVIDE_MAX = 5.3 #0.1%を外れ値として除外

SPREAD = 1

SUFFIX = ""
DB_SUFFIX = ""

PAYOUT = 1000
PAYOFF = 1000

FX = False
FX_POSITION = 10000

#学習対象外時間(ハイローがやっていない時間)
EXCEPT_LIST = [20,21,22]

DRAWDOWN_LIST = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),
                 "drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),
                 "drawdown9over": (-90000, -1000000),}

GPU_COUNT = 2
BATCH_SIZE = 1024 * 5 * GPU_COUNT
#process_count = multiprocessing.cpu_count() - 1
PROCESS_COUNT = 1

LEARNING_TYPE = "CATEGORY" #分類
#LEARNING_TYPE = "REGRESSION" #回帰

OUTPUT = 0
if LEARNING_TYPE == "CATEGORY":
    OUTPUT = 3
elif LEARNING_TYPE == "REGRESSION":
    OUTPUT = 2 # 平均, β(=1/α^2 α=標準偏差)

FILE_PREFIX = SYMBOL + "_" + LEARNING_TYPE + "_" + METHOD + "_BET" + str(BET_TERM) + "_TERM" + str(PRED_TERM * BET_TERM) + \
              "_INPUT" + str(DB1_TERM) + "-" + str(DB2_TERM) + "-" + str(DB3_TERM) + \
              "_INPUT_LEN" + INPUT_LEN_STR + \
              "_L-UNIT" + LSTM_UNIT_STR + "_D-UNIT" + DENSE_UNIT_STR + "_DROP" + str(DROP) + \
              "_L" + str(L2_RATE) + "_DIVIDEMAX" + str(DIVIDE_MAX)
EPOCH = 10

LEARNING_RATE = 0.0001

# 0:新規作成
# 1:modelからロード
# 2:chekpointからロード
LOAD_TYPE = 1
LOADING_NUM = "90*15"

# 1つのモデルに対して実行した学習回数
# モデルを引き続きロードして学習する場合1を足す
LEARNING_NUM = "90*25"

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

if LOAD_TYPE == 0:
    print("新規作成")
elif LOAD_TYPE == 1:
    print("modelからロード")
    print("LOADING_NUM:", LOADING_NUM)
elif LOAD_TYPE == 2:
    print("chekpointからロード")
    print("LOAD_CHK_NUM:", LOAD_CHK_NUM)


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

myLogger("Model is " , FILE_PREFIX)



