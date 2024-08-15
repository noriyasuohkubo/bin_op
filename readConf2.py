import math
import os
import logging.config
from decimal import Decimal
import numpy as np
from tensorflow.keras import initializers


def makedirs(path):#dirなければつくる
    if not os.path.isdir(path):
        os.makedirs(path)

#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

SEED = 0

SYMBOL = "USDJPY"

# betする間隔(秒)
BET_TERM = 5

# 予測する間隔:DB1_TERMが2でPRED_TERMが15なら30秒後の予測をする
PRED_TERM = 6

DB_NO = 3 #dukascopy data
#DB_EVAL_NO = 0 #highlow data(GBPJPY)/oanda data(USDJPY)
#DB_EVAL_NO = 1 #dukascopy data(GBPJPY)/dukascopy tick data(USDJPY)
DB_EVAL_NO = 2 #oanda data(GBPJPY)/oanda tick data(USDJPY)

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
DB1_TERM = 300
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
#DB2_TERM = 300
DB2_TERM = 0
DB2_LIST = []

if DB2_TERM != 0:
    for i in range(int(Decimal(str(DB2_TERM)) / Decimal(str(BET_TERM)))):
        DB2_LIST.append(SYMBOL + "_" + str(DB2_TERM) + "_" +  str(DB2_TERM - ((i + 1) * BET_TERM)))

print(DB2_LIST)

# inputする長い足のデータ秒間隔
# 使用しない場合 0にする
#DB3_TERM = 900
DB3_TERM = 0
DB3_LIST = []

if DB3_TERM != 0:
    for i in range(int(Decimal(str(DB3_TERM)) / Decimal(str(BET_TERM)))):
        DB3_LIST.append(SYMBOL + "_" + str(DB3_TERM) + "_" +  str(DB3_TERM - ((i + 1) * BET_TERM)))

print(DB3_LIST)

DB4_TERM = 0
#DB4_TERM = 900
DB4_LIST = []

if DB4_TERM != 0:
    for i in range(int(Decimal(str(DB4_TERM)) / Decimal(str(BET_TERM)))):
        DB4_LIST.append(SYMBOL + "_" + str(DB4_TERM) + "_" +  str(DB4_TERM - ((i + 1) * BET_TERM)))

print(DB4_LIST)

#DB5_TERM = 300
DB5_TERM = 0
DB5_LIST = []

if DB5_TERM != 0:
    for i in range(int(Decimal(str(DB5_TERM)) / Decimal(str(BET_TERM)))):
        DB5_LIST.append(SYMBOL + "_" + str(DB5_TERM) + "_" +  str(DB5_TERM - ((i + 1) * BET_TERM)))

print(DB5_LIST)

#LSTM8用
#DB_VOLUME_TERM = 2
DB_VOLUME_TERM = 0
DB_VOLUME_LIST = []
DB_VOLUME_INPUT_LEN = 1

if DB_VOLUME_TERM != 0:
    for i in range(int(Decimal(str(DB_VOLUME_TERM)) / Decimal(str(BET_TERM)))):
        DB_VOLUME_LIST.append(SYMBOL + "_" + str(DB_VOLUME_TERM) + "_" +  str(DB_VOLUME_TERM - ((i + 1) * BET_TERM)))
print(DB_VOLUME_LIST)

TMP_DB_TERMS = [
    DB1_TERM,
    DB2_TERM,
    DB3_TERM,
    DB4_TERM,
    DB5_TERM,
]

DB_TERMS = []
for dt in TMP_DB_TERMS:
    if dt != 0:
        DB_TERMS.append(dt)

DB_TERM_STR = ""

for i, v in enumerate(DB_TERMS):
    print(v)
    if i !=0 and v != 0:

        DB_TERM_STR = DB_TERM_STR + "-" + str(v)
    elif i == 0 and v != 0:
        DB_TERM_STR = str(v)

INPUT_LEN = [
    120,
    #120,
    #20,


    ]

INPUT_LEN_STR = ""

for i, v in enumerate(INPUT_LEN):
    if i !=0:
        INPUT_LEN_STR = INPUT_LEN_STR + "-" + str(v)
    else:
        INPUT_LEN_STR = str(v)

LSTM_UNIT =[
    30,
    #60,
    #6,
    #2

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
    #4,
    #2,

]
DENSE_UNIT_STR = ""

for i, v in enumerate(DENSE_UNIT):
    if i !=0:
        DENSE_UNIT_STR = DENSE_UNIT_STR + "-" + str(v)
    else:
        DENSE_UNIT_STR = str(v)

DROP = 0.0
#ls正則化

L_K_RATE = "" #lstm kernel_regularizer # 正則化タイプ - 値
L_R_RATE = "" #lstm recurrent_regularizer
L_D_RATE = "" #dense kernel_regularizer

L_K_STR = ""
if L_K_RATE != "":
    L_K_STR = "_LK" + str(L_K_RATE)
L_R_STR = ""
if L_R_RATE != "":
    L_R_STR = "_LR" + str(L_R_RATE)
L_D_STR = ""
if L_D_RATE != "":
    L_D_STR = "_LD" + str(L_D_RATE)

DIVIDE_MAX = 0
#DIVIDE_MAX = 7 # DIV 7:2004-2020データで0.04%のものを除外( md, msd)
DIVIDE_MIN = 0

#1秒データを使用する場合DB名を入れる　使用しなければ空文字
#DB_EXTRA_1 = "USDJPY_1_0"
DB_EXTRA_1 = ""
DB_EXTRA_1_LEN = 360
DB_EXTRA_1_UNIT = 36

ADX_LEN = 14
ADX = ""
ADX_STR = ""
if ADX != "":
    ADX_STR = "_ADX" + ADX

FX = True
FX_TICK_DB = SYMBOL + "_" + str(BET_TERM) + "_0"  + "_TICK"

FX_REAL_SPREAD_FLG = True #Oandaのスプレッドデータを使用する場合True
FX_FUND = 600000
FX_LEVERAGE = 25
FX_POSITION = 833 #Singleなら150000
#FX_POSITION = 150000
#FX_STOP_LOSS = -0.010 #FX_REAL_SPREAD_FLG==TrueならFalseの時のFX_STOP_LOSSよりFX_SPREAD分マイナスする 0なら損切りなし
FX_STOP_LOSS = 0.03
FX_MORE_BORDER_RGR = 0.0
FX_MORE_BORDER_CAT = 0.3
START_MONEY = 1000000
#FX_TARGET_SPREAD_LIST = [0,1,2,3,4]
FX_TARGET_SPREAD_LIST = []
FX_SINGLE_FLG = False #ポジションは一度に1つしか持たない
FX_TARGET_SHIFT = [] #betするシフト
FX_NOT_EXT_FLG = False #延長しない場合True
FX_TP_SIG = 3 #takeprofit, stoplossするシグマ
FX_SL_SIG = 3
FX_SL_D = 1
FX_LIMIT_SIG = 1 #指値注文の場合に使用するシグマ
FX_TAKE_PROFIT_FLG = False #takeprofitするか
FX_STOP_LOSS_FLG = False #stoplossするか
FX_SPREAD = 3

#FX_SINGLE_FLGがFalseの場合、保持できる最大ポジション数
FX_MAX_POSITION_CNT = int(Decimal(str(DB1_TERM)) * Decimal(str(PRED_TERM)) / Decimal(str(BET_TERM))) #特に指定しなければ最大を設定

#ハイローで取引制限がかかった状態にする
RESTRICT_FLG = False
RESTRICT_SEC = 32 #取引制限がかかる秒数

REFER_FLG = True

#　トレード回数の調整　1なら調整なし、2なら1/2にする、3なら1/3にトレード回数を減らす　
# トレード回数が多くてハイローの制限にひっかからないようにするため
TRADE_DIVIDE = 1

#本番データのスプレッドを加味したテストをする場合
#正解を出すときに使うSPREADに実データのスプレッドを使用する
REAL_SPREAD_FLG = False

#eval時に本番データを使う場合
REAL_SPREAD_EVAL_FLG =False

#実トレードした結果を加味する場合
TRADE_FLG = False

#テスト時に除外しないスプレッド
TARGET_SPREAD_LISTS = []
#TARGET_SPREAD_LISTS = [[1,2,3,4,5,6,7,8],]

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
#EXCEPT_LIST_BY_SPERAD = {1:[23,0,6,7,8,9,10,11,12,13,14,15],}

EXCEPT_DIV_LIST = []
#EXCEPT_DIV_LIST = ["-0.0", "0.0"]

EXCEPT_DIVIDE_MIN = 0
#予想divideの上限値、これより大きい場合は取引しない
EXCEPT_DIVIDE_MAX = 0

DB_TRADE_NO = 8
DB_TRADE_NAME = "GBPJPY_30_SPR_TRADE"

PAYOUT = 1300 #30秒 spread
#PAYOUT = 1200 #60秒
#PAYOUT = 1000 #30秒 trubo
#PAYOUT = 950 #30秒 trubo

PAYOFF = 1000

#PAYOUT = 2600
#PAYOFF = 2000

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


DIVIDE_DIV_LIST = { "divide-0.5": (-1.0, -0.5),"divide-1.0": (-2.0, -1.0),"divide-2.0": (-3.0, -2.0),"divide-3.0": (-4.0, -3.0),"divide-4.0": (-5.0, -4.0),
                "divide-5.0": (-6.0, -5.0),"divide-6.0": (-7.0, -6.0),
                "divide-7.0over": (-10000, -7.0),
                "divide0.5": (0.5, 1.0),"divide1.0": (1.0, 2.0),"divide2.0": (2.0, 3.0),"divide3.0": (3.0, 4.0),"divide4.0": (4.0, 5.0),
                "divide5.0": (5.0, 6.0),"divide6.0": (6.0, 7.0),
                "divide7.0over": (7.0, 10000),
                }
"""
DIVIDE_DIV_LIST = {
                "divide0.1over": (0.1, 10000),"divide0.5over": (0.5, 10000),"divide1over": (1, 10000),"divide2over": (2, 10000),
                "divide3over": (3, 10000),"divide4over": (4, 10000),"divide5over": (5, 10000),"divide6over": (6, 10000),"divide7over": (7, 10000),
                }
"""

# 学習方法 Bidirectionalならby
# LSTMならlstm
#METHOD = "BY"
#METHOD = "NORMAL"

METHOD = "LSTM"
#METHOD = "SimpleRNN" #1エポックの学習時間がLSTMの約7倍も掛かり、精度も劣るので使わない
#METHOD = "GRU" #1エポックの学習時間がかかりすぎる

#METHOD = "LSTM2" #LSTM予想を参考にさらに予想させる RGR用
#METHOD = "LSTM3"  #秒データを参考にする
# 秒データをone-hotで入力するためのone-hotのながさ
SEC_OH_LEN = int(Decimal("60") / Decimal(str(BET_TERM)))
#print("SEC_OH_LEN:", SEC_OH_LEN)

#METHOD = "LSTM4"  #分データを参考にする
# 秒データをone-hotで入力するためのone-hotのながさ
MIN_OH_LEN = 60

#METHOD = "LSTM5"  #時間データを参考にする
HOUR_OH_LEN = 24

#METHOD = "LSTM6"  #秒、分データを参考にする
#METHOD = "LSTM7"  #秒、分、時データを参考にする

#METHOD = "LSTM8"  #秒、分、時データに加えてボリューム(tick数)も参考にする

#METHOD = "LSTM9" #LSTM予想を参考にさらに予想させる Category用
LSTM9_INPUTS = [5,10] #参考にする予想を、予想時からどれぐらい以前にするか 5なら2×5で10秒前の予想
LSTM9_PRED_DBS = ["2D",] #参考にする予想のDB内でのキー test時は複数記述
LSTM9_PRED_DB_DEF = "2D" #参考にする予想のDB内でのキー test時はtestLstm2.pyなどでDataSequence2.set_db9_nameメソッドを呼んで適宜変更する
LSTM9_USE_CLOSE = False

LSTM9_INPUTS_STR = ""
if METHOD == "LSTM9" :
    for idx, ipt in enumerate(LSTM9_INPUTS):
        if idx == 0:
            LSTM9_INPUTS_STR = "_INPUT9" + "-" + str(ipt)
        else:
            LSTM9_INPUTS_STR = LSTM9_INPUTS_STR + "-" + str(ipt)
    if LSTM9_USE_CLOSE:
        LSTM9_INPUTS_STR = LSTM9_INPUTS_STR + "-C"

#直前のレートから最古のレートまでのdivideを特徴量とする
DIVIDE_ALL_FLG = False

#divideではなく、直接レートを予想する
#そのため、特徴量はclose値とする
DIRECT_FLG = False

#入力に予想時のレートをくわえる
NOW_RATE_FLG = False

GPU_COUNT = 2

#バッチサイズについて なるべく小さい方が精度がでる 最大で2048
#see:https://wandb.ai/wandb_fc/japanese/reports/---Vmlldzo1NTkzOTg
BATCH_SIZE = 1024 * 1 * 2
#BATCH_SIZE = 1024 * 1 * 16
#BATCH_SIZE = 1024 * 1 * 24

#BATCH_SIZE = 1024 * 1 * 4 #テスト時にエラーでた場合はバッチサイズさげる

#process_count = multiprocessing.cpu_count() - 1
PROCESS_COUNT = 2

#LEARNING_TYPE = "CATEGORY" #多クラス分類
#LEARNING_TYPE = "CATEGORY_BIN_UP" #2クラス分類
#LEARNING_TYPE = "CATEGORY_BIN_DW" #2クラス分類
#LEARNING_TYPE = "CATEGORY_BIN_BOTH" #2クラス分類 UP,DOWN両方を使ってテストする用　学習するためではない
#LEARNING_TYPE = "CATEGORY_BIN_FOUR" #2クラス分類 UP,DOWNそれぞれを1分の前半、後半の秒を使って学習させたモデルを使ってテストする用　学習するためではない
#LEARNING_TYPE = "REGRESSION_SIGMA" #回帰 sigma
LEARNING_TYPE = "REGRESSION" #回帰

#INPUT_DATA = "" #closeのdivideのみ特徴量とする
#INPUT_DATA = "d"
INPUT_DATA = "d"
#INPUT_DATA = "mcd"
#INPUT_DATA = "md-msd-mcd"
#INPUT_DATA = "md-msd-mcd-mmd"
#INPUT_DATA = "md-msd-mcd-mmd-mhd-mld"
#INPUT_DATA = "md-msd-mcd-d"
#INPUT_DATA = "d"
#INPUT_DATA = "md-msd-mcd-mmd-d"

INPUT_SEPARATE_FLG = False #特徴量を別々のLSTMで学習させる

INPUT_DATA_STR = ""
if INPUT_DATA != "":
    INPUT_DATA_STR = "_" + INPUT_DATA
    if INPUT_SEPARATE_FLG:
        ipt_lists = INPUT_DATA.split("-")
        if len(ipt_lists) > 1:
            INPUT_DATA_STR = INPUT_DATA_STR + "-SEP"
        else:
            INPUT_SEPARATE_FLG = False

#OUTPUT_DATA = "md"
OUTPUT_DATA = "d"
#OUTPUT_DATA = "hd-ld"

OUTPUT_DATA_STR = ""
if OUTPUT_DATA != "":
    OUTPUT_DATA_STR = "_" + OUTPUT_DATA

#Category 予測において　SPREADではなく変化率で分類する場合
#レートが153.846153846154(=0.001/0.0000065)のとき、0.001円上がるとDivideは0.065
#※Divie＝((X_after/X_before) -1) * 10000

BORDER_DIV = 0

SPREAD = 1

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
    if OUTPUT_DATA != "":
        opt_lists = OUTPUT_DATA.split("-")
        OUTPUT = len(opt_lists)
    else:
        OUTPUT = 1


#LOSS_TYPE = "C-ENTROPY"
#LOSS_TYPE = "B-ENTROPY"

#LOSS_TYPE = "MAE"
LOSS_TYPE = "MSE"
#LOSS_TYPE = "HUBER"

#LOSS_TYPE = "LOG_COSH"
#LOSS_TYPE = "HINGE"
#LOSS_TYPE = "SQUARED_HINGE"
#LOSS_TYPE = "POISSON"

#glorot_uniform
#glorot_normal
#he_normal
#he_uniform
#lecun_normal
#lecun_uniform

#ADAM, ADABOUND, AMSBOUND
OPT = "ADAM"

INIT_STR = "_GU-GU-GU" #初期値の設定(R_I,D_I,O_I)によってモデル名のsuffixを変える

#SUFFIX = "_200401_202012"
#SUFFIX = "_201201_202012"
SUFFIX = "_201601_202012"

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

if METHOD == "LSTM8":
    SUFFIX += "_VOL-" + str(int(Decimal(str(DB_VOLUME_TERM)) * Decimal(str(DB_VOLUME_INPUT_LEN))))

if DB_EXTRA_1 != "":
    SUFFIX += "_EXTRA1"

FILE_PREFIX = ""

DIVIDE_MIN_STR = ""
if DIVIDE_MIN != 0:
    DIVIDE_MIN_STR = "_DIVIDEMIN" + str(DIVIDE_MIN)

EPOCH = 40

LEARNING_RATE = 0.0001

TAG = ""

#試験的にモデルつくる際につける目印
SUFFIX += "_L-RATE" + str(LEARNING_RATE)
if TAG != "":
    SUFFIX += "-" + TAG

TERM = PRED_TERM * DB1_TERM

if METHOD == "LSTM" or METHOD == "BY" or METHOD == "LSTM2" or METHOD == "LSTM3" or METHOD == "LSTM4" or METHOD == "LSTM5" or METHOD == "LSTM6" or METHOD == "LSTM7" or METHOD == "LSTM8" or METHOD == "LSTM9" or METHOD == "SimpleRNN"  or METHOD == "RNN"  or METHOD == "GRU" :
    FILE_PREFIX = SYMBOL + "_" + LEARNING_TYPE + "_" + METHOD + "_BET" + str(BET_TERM) + "_TERM" + str(TERM) + \
                  "_INPUT" + DB_TERM_STR + \
                  "_INPUT_LEN" + INPUT_LEN_STR + \
                  "_L-UNIT" + LSTM_UNIT_STR + "_D-UNIT" + DENSE_UNIT_STR + "_DROP" + str(DROP) + \
                  L_K_STR + L_R_STR + L_D_STR + ADX_STR+ BATCH_NORMAL_STR + "_DIVIDEMAX" + str(DIVIDE_MAX) + DIVIDE_MIN_STR + \
                  BORDER_STR + SUFFIX + "_LOSS-" + LOSS_TYPE + "_" + OPT + LSTM9_INPUTS_STR + INPUT_DATA_STR + OUTPUT_DATA_STR + "_BS" + str(BATCH_SIZE) + "_SEED" + str(SEED) + "_noshuffle2-1"
elif METHOD == "NORMAL":
    FILE_PREFIX = SYMBOL + "_" + LEARNING_TYPE + "_" + METHOD + "_BET" + str(BET_TERM) + "_TERM" + str(TERM) + \
                  "_INPUT" + DB_TERM_STR + \
                  "_INPUT_LEN" + INPUT_LEN_STR + \
                  "_D-UNIT" + DENSE_UNIT_STR + "_DROP" + str(DROP) + \
                  L_K_STR + L_R_STR + L_D_STR + ADX_STR + BATCH_NORMAL_STR + "_DIVIDEMAX" + str(DIVIDE_MAX) + DIVIDE_MIN_STR + \
                  BORDER_STR + SUFFIX + "_LOSS-" + LOSS_TYPE + "_" + OPT + INPUT_DATA_STR + OUTPUT_DATA_STR + "_BS" + str(BATCH_SIZE) + "_SEED" + str(SEED)

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

def get_divide(bef, aft, math_log=False):
    divide = aft / bef
    if aft == bef:
        divide = 1

    if math_log:
        divide = 10000 * math.log(divide, math.e * 0.1)
    else:
        divide = 10000 * (divide - 1)

    return divide

def get_divide_np(bef, aft, math_log=False):
    divide = aft / bef

    if math_log:
        divide = 10000 * math.log(divide, math.e * 0.1)
    else:
        divide = 10000 * (divide - 1)

    return divide

def get_divide_arr(bef, aft, math_log=False):
    return_arr = []
    for b, a in zip(bef, aft):
        return_arr.append(get_divide(b,a,math_log))

    return np.array(return_arr)

def get_rate(pred_list, bef_list):
    aft_list = bef_list * ((pred_list / 10000) + 1)
    return aft_list

def get_rate_severe(pred_list, bef_list):
    tmp1 = Decimal(str(bef_list))
    tmp2 = Decimal(str(pred_list))
    aft_list= float(tmp1 * ((tmp2 / Decimal("10000")) + Decimal("1")))
    return aft_list

def get_rate_list(pred_list, bef_list):
    aft_list = []
    for pred, bef in zip(pred_list, bef_list):
        aft_list.append(get_rate_severe(pred,bef))

    return np.array(aft_list)

def get_ask_bid(close, spr):
    # sprはask,bidの差として入っているので、ask,bidを求めるために一旦半分にする
    tmp_spr = float(Decimal(str(spr)) / Decimal("2"))
    now_ask = close + (0.001 * tmp_spr)
    now_bid = close - (0.001 * tmp_spr)

    return [now_ask, now_bid]

def get_ask_bid_np(close, spr):
    ask_list = []
    bid_list = []

    for cl, sp in zip(close, spr):
        # sprはask,bidの差として入っているので、ask,bidを求めるために一旦半分にする
        tmp_spr = float(Decimal(str(sp)) / Decimal("2"))
        ask_list.append(cl + (0.001 * tmp_spr))
        bid_list.append(cl - (0.001 * tmp_spr))

    return np.array(ask_list), np.array(bid_list)

def get_fx_position(rate):
    if FX_SINGLE_FLG:
        return FX_LEVERAGE * FX_FUND / rate
    else:
        return FX_LEVERAGE * FX_FUND / rate / (TERM / BET_TERM)

def get_dir_cnt(dir_name):
    # ファイル数を出力
    return sum(os.path.isdir(os.path.join(dir_name, name)) for name in os.listdir(dir_name))

def rotate(input, n):
    return input[n:] + input[:n]

myLogger = printLog(loggerConf)



