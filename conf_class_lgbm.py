import os
from decimal import Decimal
from util import *
import copy
import socket
import redis
import json
import lightgbm as lgbm

class ConfClassLgbm():
    def __init__(self):
        self.CONF_TYPE = "LGBM"

        self.JPY_FIX = 130 #固定のドル円レート　BTC用

        self.SYMBOL = "USDJPY"

        self.JPY_FLG = True if "JPY" in self.SYMBOL else False

        if self.SYMBOL == "EURUSD":
            self.PIPS = 0.00001
        elif self.SYMBOL == "BTCUSD":
            self.PIPS = 0.1
        elif self.SYMBOL == "BTCJPY":
            self.PIPS = 1
        else:
            self.PIPS = 0.001

        self.BTC_FLG = True if "BTC" in self.SYMBOL else False

        self.INDEX_COL = "score" #pandasデータの並び替え基準カラム

        self.BET_TERM = 2 # betする間隔(秒)
        self.DATA_TERM = 2 #学習・テストデータの行間隔(sec)
        self.PRED_TERM = 20 # DATA_TERMの何ターム後の予測をするか

        self.BET_SHIFT = 2

        self.START_TERM = 0 #予測開始レートを何term後にするか thinkmarketsでは発注から約定まで数秒までかかるため、その対応としての設定
        self.END_TERM = 0  # 予測終了レートをPRED_TERMから何term後にするか thinkmarketsでは決済注文から約定まで数秒までかかるため、その対応としての設定

        self.START_TERM_STR = "" if self.START_TERM == 0 else "_ST" + str(self.START_TERM)
        self.END_TERM_STR = "" if self.END_TERM == 0 else "_ET" + str(self.END_TERM)

        self.TERM = self.PRED_TERM * self.DATA_TERM

        self.DB_HOST = 'localhost'
        self.DB_EVAL_NO = 2

        self.LEARNING_TYPES ={
            "CATEGORY":1, "CATEGORY_BIN":2, "CATEGORY_BIN_UP":3, "CATEGORY_BIN_DW":4, "CATEGORY_BIN_BOTH":5, "CATEGORY_BIN_FOUR":6,
            "REGRESSION_SIGMA":7, "REGRESSION":8, "REGRESSION_UP":9, "REGRESSION_DW":10, "REGRESSION_OCOPS":11,
            "CATEGORY_OCOPS":12, "CATEGORY_BIN_UP_IFD":13, "CATEGORY_BIN_DW_IFD":14, "CATEGORY_BIN_UP_IFO":15, "CATEGORY_BIN_DW_IFO":16,
            "CATEGORY_BIN_UP_IFDSF":17, "CATEGORY_BIN_DW_IFDSF":18, "CATEGORY_BIN_UP_TP":19, "CATEGORY_BIN_DW_TP":20,
            "CATEGORY_BIN_UP_OCO":21, "CATEGORY_BIN_DW_OCO":22, "CATEGORY_BIN_UP_OCOA":23, "CATEGORY_BIN_DW_OCOA":24,
        }

        self.LEARNING_TYPE = "CATEGORY"
        self.LEARNING_TYPE_STR = "_LT" + str(self.LEARNING_TYPES[self.LEARNING_TYPE])
        self.LEARNING_TYPE_NO = self.LEARNING_TYPES[self.LEARNING_TYPE]

        self.INPUT_DATA = [] #特徴量
        tmp_input = '2-d-1@2-d-10@2-d-100@2-d-110@2-d-120@2-d-130@2-d-140@2-d-150@2-d-160@2-d-170@2-d-180@2-d-190@2-d-2@2-d-20@2-d-200@2-d-210@2-d-220@2-d-230@2-d-240@2-d-250@2-d-260@2-d-270@2-d-280@2-d-290@2-d-3@2-d-30@2-d-300@2-d-310@2-d-320@2-d-330@2-d-340@2-d-350@2-d-360@2-d-370@2-d-380@2-d-390@2-d-4@2-d-40@2-d-400@2-d-410@2-d-420@2-d-430@2-d-440@2-d-450@2-d-460@2-d-470@2-d-480@2-d-490@2-d-5@2-d-50@2-d-500@2-d-6@2-d-60@2-d-7@2-d-70@2-d-8@2-d-80@2-d-9@2-d-90@744-9-DW@744-9-DW-4@744-9-SAME@744-9-SAME-4@744-9-UP@744-9-UP-4@771-33-DW@771-33-DW-4@771-33-SAME@771-33-SAME-4@771-33-UP@771-33-UP-4@773-9-DW@773-9-DW-4@773-9-SAME@773-9-SAME-4@773-9-UP@773-9-UP-4@774-32-DW@774-32-DW-4@774-32-SAME@774-32-SAME-4@774-32-UP@774-32-UP-4@798-15-DW@798-15-DW-4@798-15-SAME@798-15-SAME-4@798-15-UP@798-15-UP-4'.split("@")
        self.INPUT_DATA.extend(tmp_input)

        #他モデルの予想を使用する場合、そのモデルで使用するデータ長を考慮する必要があるので手動で設定する
        #self.INPUT_DATA_LENGTH = calc_need_len(self.INPUT_DATA, self.DATA_TERM)
        self.INPUT_DATA_LENGTH = int(Decimal(60*60*4) / Decimal(str(self.DATA_TERM))) + 1

        #カテゴリ特徴量がある場合に指定 なければ空のリストにする
        self.CATEGORY_INPUT = []
        #self.CATEGORY_INPUT = tmp_input

        self.USE_H = True
        self.USE_M = True
        self.USE_S = True
        self.USE_W = False
        self.USE_WN  = True

        if self.USE_H:
            self.INPUT_DATA.extend(["hour"])
        if self.USE_M:
            self.INPUT_DATA.extend(["min"])
        if self.USE_S:
            self.INPUT_DATA.extend(["sec"])
        if self.USE_W:
            self.INPUT_DATA.extend(["week"])
        if self.USE_WN:
            self.INPUT_DATA.extend(["weeknum"])

        self.INPUT_DATA_STR = list_to_str(self.INPUT_DATA, "@")

        #ANSWERファイルを使用する場合
        self.ANSWER_DB = ""
        self.ANSWER_DB_FILE = "/db2/answer/" + self.ANSWER_DB + ".pickle"
        self.ANSWER_DB_TYPE = self.ANSWER_DB.split("ANSWER" + str(self.TERM) + "_")[1] if self.ANSWER_DB != "" else ""
        self.ANSWER_STR = "_ASW-" + self.ANSWER_DB_TYPE if self.ANSWER_DB != "" else ""

        self.ATR_COL = ""

        # データを絞り込む指標の列名
        self.IND_COLS = []
        # 絞り込む指標の値をハイフン区切りにする
        self.IND_RANGES = [
            []
        ]
        self.IND_NEED_LENS = []
        for col in self.IND_COLS:
            #DATA_TERMとBET_TERMが異なっている場合があるので、さらにその分割って短くする
            tmp_need = get_decimal_divide(calc_need_len([col], self.BET_TERM), get_decimal_divide(self.DATA_TERM, self.BET_TERM))
            self.IND_NEED_LENS.append(tmp_need)

        self.IND_STR = ""
        for i, col in enumerate(self.IND_COLS):
            ranges = self.IND_RANGES[i]
            if len(ranges) != 0:
                self.IND_STR = self.IND_STR + "_" + col

            for r in ranges:
                self.IND_STR = self.IND_STR + "_" + str(r)
        if self.IND_STR != "":
            self.IND_STR = "_IND_" + self.IND_STR

        self.REAL_SPREAD_FLG = False
        self.REAL_SPREAD_STR = "" if self.REAL_SPREAD_FLG == False else "_RS"

        self.FX = True
        self.FX_TICK_DB =  self.SYMBOL + "_" + str(self.BET_TERM) + "_0_TICK"
        self.FX_REAL_SPREAD_FLG = True #Oandaのスプレッドデータを使用する場合True
        self.FX_REAL_SPREAD_FLG_STR = "_RS" if self.FX_REAL_SPREAD_FLG else ""

        ### 取引会社ごとの設定 START ###

        self.FX_FUND = 600000 #for oanda
        #self.FX_FUND = 900000 #for threetrader

        #self.FX_FUND = 600000
        self.FX_LEVERAGE = 25 #for 国内FX
        #self.FX_LEVERAGE = 2000 #for vantage
        #self.FX_LEVERAGE = 500 #for threetrader
        #self.FX_LEVERAGE = 200 #for mexc feature
        #self.FX_LEVERAGE = 1  # for mexc spot

        #取引量(lot):
        # 0の場合は　FX_FUND * FX_LEVERAGE / rate / (TERM / DATA_TERM)
        # 0でない場合は　FX_FIX_POSITION。 ただし、FX_FUND * FX_LEVERAGE - (既に保持しているポジション数 * FX_FIX_POSITION * rate)  の値が0以上であれば(資金余裕があれば)新規ポジションを持てる
        #self.FX_FIX_POSITION = 10000 #for oanda
        #self.FX_FIX_POSITION = 50000 #for threetrader
        #self.FX_FIX_POSITION = 100000 #for thinkmarkets
        self.FX_FIX_POSITION = 200000 # moneypartners(for GBPJPY)
        #self.FX_FIX_POSITION = 0

        #self.START_MONEY = 620000 #for oanda
        #self.START_MONEY = 3600000 #for oanda
        #self.START_MONEY = 3600000 #for threetrader
        self.START_MONEY = 1000000

        #self.ADJUST_PIPS = -0.002
        #self.ADJUST_PIPS = -0.005
        #self.ADJUST_PIPS = -0.004 #USDJPY testLstmFX2_answer testLstmFX2_rgr_limit用 海外FXなどの場合はoandaよりスプレッドが広いので、その分のスプレッドを加味する
        #self.ADJUST_PIPS = -0.00001 #EURUSD
        #self.ADJUST_PIPS = -0.001
        self.ADJUST_PIPS = 0.0

        self.IGNORE_MINUS_SPREAD = True #テスト時にマイナススプレッドを対象外とする testLstmFX2_answer用

        self.BTCUSD_SPREAD_PERCENT = 0.0  #BTCUSDのMEXCでの取引手数料パーセント

        ### 取引会社ごとの設定 END ###

        self.FX_MORE_BORDER_RGR = 0.0
        self.FX_MORE_BORDER_CAT = 0.3

        self.FX_SINGLE_FLG = True #ポジションは一度に1つしか持たない
        self.FX_TARGET_SHIFT = [] #betするシフト
        self.FX_NOT_EXT_FLG = False #延長しない場合True testLstmFX2_rgr_limit用
        self.FX_TP_SIG = 3 #takeprofit, stoplossするシグマ
        self.FX_SL_SIG = 3
        self.FX_SL_D = 1
        self.FX_LIMIT_SIG = 1 #指値注文の場合に使用するシグマ

        # regressionの場合
        #self.FX_TAKE_PROFIT = 0.06 #testLstmFX2_rgr_limit用
        #self.FX_STOP_LOSS = 0.4 #testLstmFX2_rgr_limit用

        # categoryの場合は辞書形式にする
        self.FX_TAKE_PROFIT= {'0-1':0.1}
        self.FX_STOP_LOSS= {'0-1': 0.08}

        self.FX_TP_ATR = 1 #atrをもとにtakeprofit, stoplossを決める場合に、atrに掛ける値
        self.FX_SL_ATR = 1

        self.FX_TP_PRED = 5 #予想レートをもとにtakeprofit, stoplossを決める場合に、予想レートに掛ける値
        self.FX_SL_PRED = 5

        self.FX_MIN_TP_SL = None #takeprofit, stoplossを決める場合にFX会社によって決められている最低限の値幅 fxtf:0.08

        self.FX_TAKE_PROFIT_FLG = False #takeprofitするか testLstmFX2_rgr_limit用
        self.FX_STOP_LOSS_FLG = True #stoplossするか testLstmFX2_rgr_limit用

        self.TP_SL_MODE = "auto" #auto:実際の取引でFX会社が自動でTPやSLを行う場合 or manual:手動でこちらのローカルマシンで行う場合 testLstmFX2_rgr_limit用
        #self.TP_SL_MODE = "manual" #auto:実際の取引でFX会社が自動でTPやSLを行う場合 or manual:手動でこちらのローカルマシンで行う場合 testLstmFX2_rgr_limit用
        self.TP_SL_MANUAL_TERM = 2 #TP_SL_MODEがmanualの場合にTPやSLを行う秒間隔 testLstmFX2_rgr_limit用

        self.CHANGE_STOPLOSS_FLG = False #途中で損切りラインを上げる testLstmFX2_rgr_limit用
        self.CHANGE_STOPLOGG_TERM = 4 #None:毎ループで損切りラインを上げるか判断 or 秒数:この秒数ごとに損切りラインを上げるか判断 testLstmFX2_rgr_limit用

        self.FX_BORDER_ATR = None # ATRが突然上がった場合決済するATR　Noneは設定なし

        self.FX_NOT_EXT_MINUS = None #Noneでない場合、初回延長判断時にこの値より利益が少ないなら延長しない。最初の予想が外れているなら、延長判断に使う予想も外れいている可能性が高い

        self.FX_MAX_TRADE_SEC = None #最大取引時間 設定しない場合はNone

        self.FX_STOPLOSS_PER_SEC_FLG = False #True:Betしてから一定秒数ごとにレート変化に応じて回復見込みが薄ければ損切りする
        self.FX_STOPLOSS_PER_SEC = 4 #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に損切り判定する秒間隔
        self.FX_STOPLOSS_PER_SEC_DB_NO = 10 #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に参照するDB_NO
        self.FX_STOPLOSS_PER_SEC_DB_LIST = [p for p in range(30) if p!= 0 and Decimal(p) % Decimal(self.FX_STOPLOSS_PER_SEC) == 0 ] #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に参照するDB(秒数)
        self.FX_STOPLOSS_PER_SEC_EXT_ONLY_FLG = False #True:延長したときのみ損切りする
        self.FX_STOPLOSS_PER_SEC_BORDER = 0.01 #レートが回復しそうな見込みがこの可能性以上あれば損切りしない

        self.BUY_FLG = True
        self.SELL_FLG = True

        self.TRADE_SHIFT = 1 #この秒数で割り切れるシフトでしか取引しない.  None:設定なし

        self.RESTRICT_FLG = True # 取引制限がかかった状態にする
        self.RESTRICT_SEC = 16 #取引制限がかかる秒数

        self.SAME_SHIFT_NG_FLG = True #True:既に同じシフトの建玉がある場合は新規注文しない testLstmFX2_rgr_limit.py用
        self.NG_SHIFT = 24

        #self.FX_MAX_POSITION_CNT = None #保持可能な最大ポジション数 Noneなら制限なし 0なら指定なし testLstmFX2_rgr_limit.py用
        self.FX_MAX_POSITION_CNT = 10

        if self.FX_SINGLE_FLG:
            self.FX_MAX_POSITION_CNT = 1
        else:
            if self.FX_MAX_POSITION_CNT == 0:
                #指定がない場合
                if self.RESTRICT_FLG:
                    self.FX_MAX_POSITION_CNT =  self.TERM / self.RESTRICT_SEC
                else:
                    self.FX_MAX_POSITION_CNT = self.TERM / self.TRADE_SHIFT

                if self.FX_NOT_EXT_FLG:
                    #延長しない場合はいつまでもポジションを持ち続けることがないので制限なしにする
                    self.FX_MAX_POSITION_CNT = None

        #self.PAYOUT = 1300 #30秒 spread
        self.PAYOUT = 1000
        #self.PAYOUT = 960
        self.PAYOFF = 1000

        # 学習対象外時間
        self.EXCEPT_LIST = [20,21,22]
        if self.BTC_FLG:
            self.EXCEPT_LIST = []
        self.EXCEPT_LIST_STR = "_EL" + list_to_str(self.EXCEPT_LIST, spl="-") if len(self.EXCEPT_LIST) != 0 else ""

        #テスト対象外時間 testLstmFX2_answer testLstmFX2_rgr_limit用
        self.EXCEPT_LIST_HOUR_TEST = [20,21,22]
        if self.FX and self.BTC_FLG == False:
            self.EXCEPT_LIST_HOUR_TEST = [20, 21, 22, 23]
        elif self.BTC_FLG:
            self.EXCEPT_LIST_HOUR_TEST = []

        #テスト対象外秒 testLstmFX2_answer testLstmFX2_rgr_limit用
        self.EXCEPT_LIST_SEC_TEST = []

        # d:d(変化率)を求める sub:sub(差)を求める
        self.OUTPUT_TYPE = "d"
        self.OUTPUT_TYPE_STR = "_OT-" + self.OUTPUT_TYPE
        self.OUTPUT_MULTI = 1

        #変化の基準の直近データをCloseにする場合にTrue　Falseの場合はOUTPUT_DATAを基準の直近データとする
        self.OUTPUT_DATA_BEF_C = False
        self.OUTPUT_DATA_BEF_C_STR = "_ODBC" if self.OUTPUT_DATA_BEF_C else ""

        # regression系の場合  c:変化の基準をcloseとする smam60:変化の基準を直近60秒のsmamと予想時間直前の60秒のsmamとする
        # category系の場合  c:up or dwの基準をcloseの値とする smam60:基準を直近60秒のsmamとする
        self.OUTPUT_DATA = "c"
        self.OUTPUT_DATA_STR = "_OD-" + self.OUTPUT_DATA + self.OUTPUT_DATA_BEF_C_STR
        self.OUTPUT_LIST = self.OUTPUT_DATA.split("_")

        # Category 予測において　SPREADではなく変化率で分類する場合
        # レートが153.846153846154(=0.001/0.0000065)のとき、0.001円上がるとDivideは0.065
        # ※Divie＝((X_after/X_before) -1) * 10000
        self.BORDER_DIV = 0.01

        # lgbm_make_data.pyで作成しておく正解リスト BORDER_DIV別にlgbm_make_data.pyでデータ作成するのを防ぐ為、まとめて作成する
        self.BORDER_DIV_LIST = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

        self.SPREAD = 1

        # lgbm_make_data.pyで作成しておく正解リスト BORDER_DIV別にlgbm_make_data.pyでデータ作成するのを防ぐ為、まとめて作成する
        self.SPREAD_LIST = [1,]

        self.BORDER_STR = ""
        self.BORDER_LIST_STR = ""

        if self.LEARNING_TYPE == "CATEGORY" or self.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or self.LEARNING_TYPE == "CATEGORY_BIN_UP" or self.LEARNING_TYPE == "CATEGORY_BIN_DW":
            if self.OUTPUT_TYPE == "sub":
                self.BORDER_STR = "_SPREAD" + str(self.SPREAD)
            elif self.OUTPUT_TYPE == "d":
                self.BORDER_STR = "_BDIV" + str(self.BORDER_DIV)

        if self.OUTPUT_TYPE == "d":
            self.BORDER_LIST_STR = "_BDIVL" + list_to_str(self.BORDER_DIV_LIST)
        elif self.OUTPUT_TYPE == "sub":
            self.BORDER_LIST_STR = "_SPREADL" + list_to_str(self.SPREAD_LIST)

        #METRIC
        #For Category
        #self.METRIC = "binary_logloss"
        #self.METRIC = "multi_logloss"

        #For Regression
        #self.METRIC = "MAE"
        #self.METRIC = "MSE"
        #self.METRIC = "HUBER"
        #self.METRIC = "LOG_COSH"
        #self.METRIC = "HINGE"
        #self.METRIC = "SQUARED_HINGE"
        #self.METRIC = "POISSON"

        if self.LEARNING_TYPE == "CATEGORY" or self.LEARNING_TYPE == "CATEGORY_BIN_BOTH":
            self.OBJECTIVE = "multiclass"
            self.METRIC = "multi_logloss"

        elif (self.LEARNING_TYPE == "CATEGORY_BIN_UP" or self.LEARNING_TYPE == "CATEGORY_BIN_DW"):
            self.OBJECTIVE = "binary"
            self.METRIC = "binary_logloss"

        elif "REGRESSION" in self.LEARNING_TYPE:
            self.OBJECTIVE = "regression"
            self.METRIC = "rmse"

        # パラメータ探索しない場合:NORMAL
        # optunaの場合:OPTUNA
        # optunaのLightGBMTunerを使う場合:TUNER
        self.TUNER_TYPE = 'NORMAL'
        self.TUNER_TYPE_STR = "_TUNERTYPE-" + self.TUNER_TYPE

        self.LGBM_PARAM_DICT={
            'boosting_type' : 'gbdt',
            'device_type' :'cpu', #cpu or gpu
            'learning_rate':0.1,  # default 0.1
            'max_bin':500,  # default 255 大きすぎると過学習。小さくすると精度は下がるが汎化性能が上がる。
            'max_depth':-1,  # default -1
            #'feature_pre_filter': False, #min_data_in_leaf=min_child_samplesをチューニングするときにはFalse
            'min_data_in_leaf':20,  # default 20
            'num_leaves':31,  # default 31 一つの木の最大葉数
            'num_threads':14,  # LightGBMに用いるスレッド数 実際のCPUコア数にすれば良い
            'seed': 42,
        }

        self.LGBM_PARAM_STR = "_LGBM-PARAM"
        for k,v in self.LGBM_PARAM_DICT.items():
            self.LGBM_PARAM_STR += "-" + k + "_" + str(v)

        #OPTUNA用パラメータ
        self.LGBM_OPTUNA_PARAM_DICT={
            #'learning_rate': [0.01,0.1],
            'num_leaves': [2, 256],
            'min_data_in_leaf': [5,100],
            'lambda_l1': [1e-8, 10.0],
            'lambda_l2': [1e-8, 10.0],
        }

        self.LGBM_OPTUNA_PARAM_STR = "_LGBM-OPTUNA-PARAM"
        for k,v in self.LGBM_OPTUNA_PARAM_DICT.items():
            self.LGBM_OPTUNA_PARAM_STR += "-" + k + "_" + list_to_str(v, spl=":")

        self.LGBM_OPTUNA_PARAM_STR = "" if self.TUNER_TYPE != 'OPTUNA' else self.LGBM_OPTUNA_PARAM_STR

        #その他のパラメータ　使用しないパラメータはコメントアウトしないとモデル名に含まれてしまう
        self.OTHER_PARAM_DICT={
            #'gpu_device_id':0,
            'early_stopping_rounds': 30,
            'num_boost_round': 10000,

            'n_trials' : 50, #optuna用のパラメータ:探索回数

        }

        self.OTHER_PARAM_DICT_STR = "_OTHER-PARAM"
        for k,v in self.OTHER_PARAM_DICT.items():
            self.OTHER_PARAM_DICT_STR += "-" + k + "_" + str(v)

        self.FLOAT = "32" #データがfloat64なら64を入力

        #self.SUFFIX = "_20040101_20230331"
        self.SUFFIX = "_20100101_20230331"
        #self.SUFFIX = "_20180101_20221030"
        #self.SUFFIX = "_20100101_20230331"

        self.EVAL = "_20230401_20240630"
        #self.EVAL = "_20230401_20240504"

        self.FILE_PREFIX = ""
        self.FILE_PREFIX_DB = ""
        self.MODEL_DIR = "/app/model_lgbm/bin_op/"

        self.DRAWDOWN_LIST = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),
                 "drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),
                 "drawdown9over": (-90000, -1000000),}

        #FILE_PREFIXを作成
        self.make_file_prefix()

    def change_fx_real_spread_flg(self, flg):
        self.FX_REAL_SPREAD_FLG = flg

    def change_learning_rate(self, rate):
        self.LEARNING_RATE = rate
        self.make_file_prefix()

    def make_file_prefix(self):

        self.FILE_PREFIX_DB = self.SYMBOL + self.LEARNING_TYPE_STR + "_B" + str(self.BET_TERM) + "_BS" + str(self.BET_SHIFT) + "_D" + str(self.DATA_TERM) + "_T" + str(self.TERM) + \
                              self.START_TERM_STR + self.END_TERM_STR + self.TUNER_TYPE_STR + self.LGBM_PARAM_STR + self.LGBM_OPTUNA_PARAM_STR + self.OTHER_PARAM_DICT_STR + \
                              "_FL" + self.FLOAT + self.REAL_SPREAD_STR + self.BORDER_STR  + self.OUTPUT_TYPE_STR + self.OUTPUT_DATA_STR + self.SUFFIX + self.EVAL + \
                              self.IND_STR + self.ANSWER_STR + self.EXCEPT_LIST_STR  + "_" + socket.gethostname()


    def numbering(self):
        # win2のDBを参照してモデルのナンバリングを行う
        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        result = r.zrevrange("MODEL_NO", 0, -1, withscores=True)  # 全件取得
        if len(result) == 0:
            print("CANNOT GET MODEL_NO")
            exit(1)
        else:
            newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)
            self.FILE_PREFIX_DB = self.FILE_PREFIX_DB + "_MN" + str(newest_no)  # モデルに番号をつける
            self.FILE_PREFIX = "MN" + str(newest_no)

            for line in result:
                body = line[0]
                tmps = json.loads(body)
                tmp_name = tmps.get("name")
                if tmp_name == self.FILE_PREFIX_DB:
                    # 同じモデルがないが確認
                    print("The Model Already Exists!!!")
                    exit(1)

            # DBにモデルを登録
            child = {
                'name': self.FILE_PREFIX_DB,
                'no': newest_no,
                'input': self.INPUT_DATA_STR,
            }
            r.zadd("MODEL_NO", json.dumps(child), newest_no)

    def get_fx_position(self, rate):
        if self.FX_FIX_POSITION == 0:
            if self.FX_SINGLE_FLG:
                    return self.FX_LEVERAGE * self.FX_FUND / rate
            else:
                if self.RESTRICT_FLG:
                    return self.FX_LEVERAGE * self.FX_FUND / rate / (self.TERM / self.RESTRICT_SEC)
                else:
                    return self.FX_LEVERAGE * self.FX_FUND / rate / (self.TERM / self.TRADE_SHIFT)
        else:
            if type(rate).__module__ == "numpy":
                return np.full(len(rate), self.FX_FIX_POSITION)
            elif type(rate) == list:
                return np.full(len(rate), self.FX_FIX_POSITION).tolist()
            else:
                return self.FX_FIX_POSITION

    def get_fx_position_jpy(self, rate, jpy):
        if self.FX_FIX_POSITION == 0:
            if self.FX_SINGLE_FLG:
                    return self.FX_LEVERAGE * self.FX_FUND / jpy / rate
            else:
                if self.RESTRICT_FLG:
                    return self.FX_LEVERAGE * self.FX_FUND / jpy / rate / (self.TERM / self.RESTRICT_SEC)
                else:
                    return self.FX_LEVERAGE * self.FX_FUND / jpy / rate / (self.TERM / self.TRADE_SHIFT)

        else:
            if type(rate).__module__ == "numpy":
                return np.full(len(rate), self.FX_FIX_POSITION)
            elif type(rate) == list:
                return np.full(len(rate), self.FX_FIX_POSITION).tolist()
            else:
                return self.FX_FIX_POSITION
