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

        #レート取得元FX会社
        self.FROM_DATA = "DUKAS" #OANDA

        self.JPY_FIX = 130 #固定のドル円レート　BTC用

        self.SYMBOL = "USDJPY"

        self.JPY_FLG = True if "JPY" in self.SYMBOL else False

        if self.SYMBOL == "EURUSD":
            self.PIPS = 0.00001
        elif self.SYMBOL == "BTCUSD":
            self.PIPS = 0.01
        elif self.SYMBOL == "BTCJPY":
            self.PIPS = 1
        else:
            self.PIPS = 0.001

        self.BTC_FLG = True if "BTC" in self.SYMBOL else False

        self.INDEX_COL = "score" #pandasデータの並び替え基準カラム

        self.BET_TERM = 1 # betする間隔(秒)
        self.DATA_TERM = 1 #学習・テストデータの行間隔(sec)
        self.PRED_TERM = 30 # DATA_TERMの何ターム後の予測をするか

        self.BET_SHIFT = 1

        self.START_TERM = 0 #予測開始レートを何term後にするか thinkmarketsでは発注から約定まで数秒までかかるため、その対応としての設定
        self.END_TERM = 0  # 予測終了レートをPRED_TERMから何term後にするか thinkmarketsでは決済注文から約定まで数秒までかかるため、その対応としての設定

        self.START_TERM_STR = "" if self.START_TERM == 0 else "_ST" + str(self.START_TERM)
        self.END_TERM_STR = "" if self.END_TERM == 0 else "_ET" + str(self.END_TERM)

        self.TERM = self.PRED_TERM * self.DATA_TERM

        self.DB_HOST = 'localhost'
        #self.DB_EVAL_NO = 2
        self.DB_EVAL_NO = 1 #vantage
        #self.DB_EVAL_NO = 4 #sbi
        #self.DB_EVAL_NO = 8 #asahi-markets
        #self.DB_EVAL_NO = 9 #gaitame

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

        self.LEARNING_TYPE_STOPLOSS = False #True: CATEGORY_BIN_UP or CATEGORY_BIN_DW時のDataSequence2.pyでのデータ作成時に、ストップロスに引っかかる場合は不正解とする
        self.LEARNING_TYPE_STOPLOSS_PRICE = -0.5
        if self.LEARNING_TYPE_STOPLOSS:
            self.LEARNING_TYPE_STR += "-SL" + str(self.LEARNING_TYPE_STOPLOSS_PRICE)
            if (self.LEARNING_TYPE in ["CATEGORY_BIN_UP", "CATEGORY_BIN_DW"]) == False:
                #CATEGORY_BIN_UP CATEGORY_BIN_DW以外でのLEARNING_TYPE_STOPLOSS=True はNG。CATEGORYではSTOPLOSSに掛かった場合の正解が定められないため。
                print("LEARNING_TYPE_STOPLOSS must be in CATEGORY_BIN_UP or CATEGORY_BIN_DW")
                exit(1)

        self.INPUT_DATA = [] #特徴量
        #tmp_input = '1-d-1@1-d-10@1-d-1018@1-d-10234@1-d-106@1-d-11258@1-d-1146@1-d-12@1-d-122@1-d-12282@1-d-1274@1-d-138@1-d-14@1-d-1402@1-d-1530@1-d-154@1-d-16@1-d-170@1-d-1786@1-d-18@1-d-186@1-d-2@1-d-2042@1-d-218@1-d-22@1-d-2298@1-d-250@1-d-2554@1-d-26@1-d-2810@1-d-282@1-d-3@1-d-30@1-d-3066@1-d-314@1-d-34@1-d-346@1-d-3578@1-d-378@1-d-38@1-d-4@1-d-4090@1-d-42@1-d-442@1-d-4602@1-d-5@1-d-50@1-d-506@1-d-5114@1-d-5626@1-d-570@1-d-58@1-d-6@1-d-6138@1-d-634@1-d-66@1-d-698@1-d-7162@1-d-74@1-d-762@1-d-8@1-d-8186@1-d-82@1-d-890@1-d-90@1-d-9210@1-ema-1080@1-ema-120@1-ema-1200@1-ema-13@1-ema-1320@1-ema-1440@1-ema-240@1-ema-300@1-ema-360@1-ema-420@1-ema-480@1-ema-540@1-ema-60@1-ema-600@1-ema-720@1-ema-840@1-ema-960'.split("@")
        tmp_input = '1500-196-DW@1500-196-DW-1@1500-196-DW-2@1500-196-SAME@1500-196-SAME-1@1500-196-SAME-2@1500-196-UP@1500-196-UP-1@1500-196-UP-2@1504-196-DW@1504-196-DW-1@1504-196-DW-2@1504-196-SAME@1504-196-SAME-1@1504-196-SAME-2@1504-196-UP@1504-196-UP-1@1504-196-UP-2@1633-79-DW@1633-79-DW-1@1633-79-DW-2@1633-79-SAME@1633-79-SAME-1@1633-79-SAME-2@1633-79-UP@1633-79-UP-1@1633-79-UP-2@1750-883-DW@1750-883-DW-1@1750-883-DW-2@1750-883-SAME@1750-883-SAME-1@1750-883-SAME-2@1750-883-UP@1750-883-UP-1@1750-883-UP-2@1835-67-DW@1835-67-DW-1@1835-67-DW-2@1835-67-SAME@1835-67-SAME-1@1835-67-SAME-2@1835-67-UP@1835-67-UP-1@1835-67-UP-2'.split("@")
        #tmp_input = 'diff_1@diff_10@diff_1018@diff_10234@diff_106@diff_11258@diff_1146@diff_12@diff_122@diff_12282@diff_1274@diff_138@diff_14@diff_1402@diff_1530@diff_154@diff_16@diff_170@diff_1786@diff_18@diff_186@diff_2@diff_2042@diff_218@diff_22@diff_2298@diff_250@diff_2554@diff_26@diff_2810@diff_282@diff_3@diff_30@diff_3066@diff_314@diff_34@diff_346@diff_3578@diff_378@diff_38@diff_4@diff_4090@diff_42@diff_442@diff_4602@diff_5@diff_50@diff_506@diff_5114@diff_5626@diff_570@diff_58@diff_6@diff_6138@diff_634@diff_66@diff_698@diff_7162@diff_74@diff_762@diff_8@diff_8186@diff_82@diff_890@diff_90@diff_9210'.split("@")
        self.INPUT_DATA.extend(tmp_input)

        #他モデルの予想を使用する場合、そのモデルで使用するデータ長を考慮する必要があるので手動で設定する
        #self.INPUT_DATA_LENGTH = calc_need_len(self.INPUT_DATA, self.DATA_TERM)
        self.INPUT_DATA_LENGTH = int(Decimal(60*60*4) / Decimal(str(self.DATA_TERM))) + 1

        #カテゴリ特徴量がある場合に指定 なければ空のリストにする
        self.CATEGORY_INPUT = []
        #self.CATEGORY_INPUT = tmp_input

        #lgbm_make_data.pyで時分秒データを作成する場合True
        self.MAKE_TIME_DATA = False
        self.MAKE_TIME_DATA_STR = "" if self.MAKE_TIME_DATA == False else "_TIME-DATA"

        self.USE_H = False
        self.USE_M = False
        self.USE_S = False
        self.USE_W = False
        self.USE_WN  = False

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

        #Datasequens2のデータに予想結果までの最安値と最高値を含める場合:True
        self.INCLUDE_HL_FLG = True

        #キリのいいレートからどれくらい離れているかを特徴量とする
        #キリのいいレート幅を指定する Noneなら指定なし
        self.RANGE_FROM_RATE = []
        #self.RANGE_FROM_RATE = [0.1,0.5,1.0]

        for rfr in self.RANGE_FROM_RATE:
            self.INPUT_DATA.extend(["range_from_rate" + str(rfr)])

        self.RANGE_FROM_RATE_STR = "" if len(self.RANGE_FROM_RATE) == 0 else "_RFR" + list_to_str(self.RANGE_FROM_RATE)

        self.RANGE_FROM_RATE_LIST = [] #lgbm_make_dataでまとめて作成する対象
        #self.RANGE_FROM_RATE_LIST = [0.1,0.5,1.0] #lgbm_make_dataでまとめて作成する対象
        self.RANGE_FROM_RATE_LIST_STR = "" if len(self.RANGE_FROM_RATE_LIST) == 0 else "_RFRL" + list_to_str(self.RANGE_FROM_RATE_LIST)

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

        # 以下、意識すべき水平線上の過去レートの数を特徴量に加える場合
        self.HOR_DB_CORE_LIST = [] #例:"60_1440_0.01"
        self.HOR_LINE_NUM = 1 #参照する水平線の数:1 or 3
        self.HOR_STR = ""
        if self.HOR_LINE_NUM != 1 and self.HOR_LINE_NUM != 3:
            print("HOR_LINE_NUM must be 1 or 3")
            exit(1)

        for i, h in enumerate(self.HOR_DB_CORE_LIST):
            if i == 0:
                self.HOR_STR = "_HOR"
            self.HOR_STR += "-" + self.HOR_DB_CORE

            for j, in range(self.HOR_LINE_NUM):
                if j == 0:
                    self.INPUT_DATA.extend([h])
                elif j == 1:
                    self.INPUT_DATA.extend([h + "-low"])
                elif j == 2:
                    self.INPUT_DATA.extend([h + "-high"])

        if self.HOR_STR != "":
            self.HOR_STR += "-" + str(self.HOR_LINE_NUM)

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
        #self.FX_LEVERAGE = 25 #for 国内FX
        #self.FX_LEVERAGE = 2000 #for vantage
        self.FX_LEVERAGE = 500 #for threetrader
        #self.FX_LEVERAGE = 200 #for mexc feature
        #self.FX_LEVERAGE = 1  # for mexc spot
        #self.FX_LEVERAGE = 50 #oanda

        #取引量(lot):
        # 0の場合は　FX_FUND * FX_LEVERAGE / rate / (TERM / DATA_TERM)
        # 0でない場合は　FX_FIX_POSITION。 ただし、FX_FUND * FX_LEVERAGE - (既に保持しているポジション数 * FX_FIX_POSITION * rate)  の値が0以上であれば(資金余裕があれば)新規ポジションを持てる
        #self.FX_FIX_POSITION = 10000 #for oanda
        #self.FX_FIX_POSITION = 50000 #for threetrader
        #self.FX_FIX_POSITION = 100000 #for thinkmarkets
        #self.FX_FIX_POSITION = 200000 # moneypartners(for GBPJPY)
        self.FX_FIX_POSITION = 0

        #self.START_MONEY = 620000 #for oanda
        #self.START_MONEY = 3600000 #for oanda
        #self.START_MONEY = 3600000 #for threetrader
        self.START_MONEY = 1000000

        #self.ADJUST_PIPS = -0.008
        #self.ADJUST_PIPS = -0.006
        #self.ADJUST_PIPS = -0.004 #threetrader
        #self.ADJUST_PIPS = -0.005 #tradeview
        #self.ADJUST_PIPS = -0.002
        self.ADJUST_PIPS = 0.0 #vantage
        #self.ADJUST_PIPS = 0.002

        #スプレッドにより獲得PIPSを調整する testLstmFX2_answer用
        self.ADJUST_PIPS_SPREAD_FLG = True

        self.TPSL_ADJUST_PIPS = -0.003 #スリップによる損失 testLstmFX2_rgr_limit用
        #self.TPSL_ADJUST_PIPS = 0.0

        self.IGNORE_MINUS_SPREAD = True #テスト時にマイナススプレッドを対象外とする testLstmFX2_answer用

        self.BTCUSD_SPREAD_PERCENT = 0.0  #BTCUSDのMEXCでの取引手数料パーセント

        ### 取引会社ごとの設定 END ###

        #self.FX_MORE_BORDER_RGR = 0.0
        #self.FX_MORE_BORDER_CAT = 0.3

        self.FX_SINGLE_FLG = False #ポジションは一度に1つしか持たない
        self.FX_TARGET_SHIFT = [] #betするシフト
        self.FX_NOT_EXT_FLG = False #延長しない場合True testLstmFX2_rgr_limit用
        self.FX_TP_SIG = 3 #takeprofit, stoplossするシグマ
        self.FX_SL_SIG = 3
        self.FX_SL_D = 1
        self.FX_LIMIT_SIG = 1 #指値注文の場合に使用するシグマ

        self.FX_TAKE_PROFIT_FLG = False #takeprofitするか testLstmFX2_rgr_limit用
        self.FX_STOP_LOSS_FLG = False #stoplossするか testLstmFX2_rgr_limit用

        self.TP_SL_MODE = "auto" #auto:実際の取引でFX会社が自動でTPやSLを行う場合 or manual:手動でこちらのローカルマシンで行う場合 testLstmFX2_rgr_limit用
        #self.TP_SL_MODE = "manual" #auto:実際の取引でFX会社が自動でTPやSLを行う場合 or manual:手動でこちらのローカルマシンで行う場合 testLstmFX2_rgr_limit用
        self.TP_SL_MANUAL_TERM = 4 #TP_SL_MODEがmanualの場合にTPやSLを行う秒間隔 testLstmFX2_rgr_limit用

        self.FX_BORDER_ATR = None # ATRが突然上がった場合決済するATR　Noneは設定なし

        self.FX_NOT_EXT_MINUS = None #Noneでない場合、初回延長判断時にこの値より利益が少ないなら延長しない。最初の予想が外れているなら、延長判断に使う予想も外れいている可能性が高い

        self.FX_MAX_TRADE_SEC = None #最大取引時間 設定しない場合はNone

        self.BUY_FLG = True
        self.SELL_FLG = True

        self.TRADE_SHIFT = 1 #この秒数で割り切れるシフトでしか取引しない.  None:設定なし

        self.RESTRICT_FLG = False  # 取引制限がかかった状態にする
        self.RESTRICT_SEC = 1  # 取引制限がかかる秒数

        self.SAME_SHIFT_NG_FLG = True #True:既に同じシフトの建玉がある場合は新規注文しない testLstmFX2_rgr_limit.py用
        self.NG_SHIFT = self.TERM

        #self.FX_MAX_POSITION_CNT = None #保持可能な最大ポジション数 Noneなら制限なし 0なら指定なし testLstmFX2_rgr_limit.py用
        self.FX_MAX_POSITION_CNT = 0

        # FX_MAX_POSITION_CNTの指定がない場合
        if self.FX_SINGLE_FLG:
            self.FX_MAX_POSITION_CNT = 1
        else:
            if self.FX_MAX_POSITION_CNT == 0:
                # 指定がない場合
                self.FX_MAX_POSITION_CNT = int(get_decimal_divide(self.TERM, self.BET_SHIFT))

        #EXCEPT_DIVIDE_MIN か EXCEPT_DIVIDE_MAXが0でない場合、
        #DATA_TERM * DIVIDE_PREV_LENGTH 秒前のレートと予想時レートのdivideが指定divideより大きかったり小さかったら対象外とする
        #これを設定するとlgbm_make_dataで学習データから該当データを対象外とする
        #テストデータからは除外されない
        self.DIVIDE_PREV_LENGTH = 300
        self.EXCEPT_DIVIDE_MIN = 0
        self.EXCEPT_DIVIDE_MAX = 0 #予想divideの上限値、これより大きい場合は取引しない

        self.EXCEPT_DIVIDE_STR = ""
        if self.EXCEPT_DIVIDE_MIN != 0 or self.EXCEPT_DIVIDE_MAX != 0:
            self.EXCEPT_DIVIDE_STR = "_DPL" + str(self.DIVIDE_PREV_LENGTH)

        if self.EXCEPT_DIVIDE_MIN != 0:
            self.EXCEPT_DIVIDE_STR = self.EXCEPT_DIVIDE_STR + "_EDMIN" + str(self.EXCEPT_DIVIDE_MIN)
        if self.EXCEPT_DIVIDE_MAX != 0:
            self.EXCEPT_DIVIDE_STR = self.EXCEPT_DIVIDE_STR + "_EDMAX" + str(self.EXCEPT_DIVIDE_MAX)

        #答えまでの最小divで学習データを絞る
        self.EXCEPT_DIVIDE_MIN_AFTER = 0 #0:絞らない
        self.EXCEPT_DIVIDE_MIN_AFTER_STR = "_EDMIN-A" + str(self.EXCEPT_DIVIDE_MIN_AFTER) if self.EXCEPT_DIVIDE_MIN_AFTER != 0 else ""

        #バイナリオプションの場合true
        self.BINARY = False #testLstmFX2_answer.py用

        #self.PAYOUT = 1300 #30秒 spread
        self.PAYOUT = 1000
        #self.PAYOUT = 960
        self.PAYOFF = -1000

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
        self.BORDER_DIV = 0.1

        # lgbm_make_data.pyで作成しておく正解リスト BORDER_DIV別にlgbm_make_data.pyでデータ作成するのを防ぐ為、まとめて作成する
        self.BORDER_DIV_LIST = [0.1,0.5,1.0,]

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
        #self.TUNER_TYPE = 'OPTUNA'
        self.TUNER_TYPE_STR = "_TUNERTYPE-" + self.TUNER_TYPE
        #BEST PARAMS: {'num_leaves': 4, 'max_bin': 831, 'min_data_in_leaf': 83, 'lambda_l1': 8.703771989018222, 'lambda_l2': 1.1885596206855467, 'min_sum_hessian_in_leaf': 0.0009888969994132815}
        self.LGBM_PARAM_DICT={
            'boosting_type' : 'gbdt',
            'device_type' :'cpu', #cpu or gpu
            'learning_rate':0.5,  # default 0.1
            'max_bin': 255,  # default 255 小さくすると精度は下がるが汎化性能が上がる。
            'max_depth':-1,  # default -1
            'min_data_in_leaf':50,  # default 20
            'num_leaves':31,  # default 31 一つの木の最大葉数
            'num_threads':15,  # LightGBMに用いるスレッド数 実際のCPUコア数にすれば良い
            'seed': 42,
            'bagging_fraction': 1, #フィルさん設定
            'bagging_freq':1, #フィルさん設定
            #'lambda_l1': 8.703771989018222,
            #'lambda_l2': 1.1885596206855467,
            #'min_sum_hessian_in_leaf': 0.0009888969994132815,
            'init_model': "MN2110",
        }

        # min_data_in_leaf=min_child_samplesをチューニングするときにはFalse
        if self.TUNER_TYPE == 'OPTUNA':
            self.LGBM_PARAM_DICT["feature_pre_filter"] = False

        self.LGBM_PARAM_STR = "_LGBM-PARAM"
        for k,v in self.LGBM_PARAM_DICT.items():
            self.LGBM_PARAM_STR += "-" + k + "_" + str(v)

        #OPTUNA用パラメータ
        self.LGBM_OPTUNA_PARAM_DICT={
            #'learning_rate': [0.01,0.1],
            #'num_leaves': [3, 300],
            'max_bin': [100, 10000],
            #'min_data_in_leaf': [2,200],
            #'lambda_l1': [1e-8, 10.0],
            #'lambda_l2': [1e-8, 10.0],
            #'min_sum_hessian_in_leaf':[0.0001,0.01],
        }

        self.LGBM_OPTUNA_PARAM_STR = "_LGBM-OPTUNA-PARAM"
        for k,v in self.LGBM_OPTUNA_PARAM_DICT.items():
            self.LGBM_OPTUNA_PARAM_STR += "-" + k + "_" + list_to_str(v, spl=":")

        self.LGBM_OPTUNA_PARAM_STR = "" if self.TUNER_TYPE != 'OPTUNA' else self.LGBM_OPTUNA_PARAM_STR

        #その他のパラメータ　使用しないパラメータはコメントアウトしないとモデル名に含まれてしまう
        self.OTHER_PARAM_DICT={
            #'gpu_device_id':0,
            'early_stopping_rounds': 100,
            'num_boost_round': 1000,
            'n_trials' : 100, #optuna用のパラメータ:探索回数
        }

        self.OTHER_PARAM_DICT_STR = "_OTHER-PARAM"
        for k,v in self.OTHER_PARAM_DICT.items():
            self.OTHER_PARAM_DICT_STR += "-" + k + "_" + str(v)

        self.FLOAT = "32" #データがfloat64なら64を入力

        #self.SUFFIX = "_20040101_20241201"
        self.SUFFIX = "_20160101_20241201"
        #self.SUFFIX = "_20100101_20241201"

        self.EVAL = "_20241201_20260502"
        #self.EVAL = "_20250501_20260124"

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
                              "_FL" + self.FLOAT + self.HOR_STR + self.REAL_SPREAD_STR + self.BORDER_STR  + self.OUTPUT_TYPE_STR + self.OUTPUT_DATA_STR + self.SUFFIX + self.EVAL + \
                              self.IND_STR + self.ANSWER_STR + self.EXCEPT_LIST_STR + self.EXCEPT_DIVIDE_STR + self.EXCEPT_DIVIDE_MIN_AFTER_STR + self.RANGE_FROM_RATE_STR + "_" + socket.gethostname() + "_" + self.FROM_DATA


    def numbering(self, train_file_score, test_file_score):
        self.FILE_PREFIX_DB = self.FILE_PREFIX_DB + "_TRAF" + str(train_file_score) + "_TESF" + str(test_file_score)
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
