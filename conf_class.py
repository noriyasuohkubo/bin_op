import datetime
import json
import os
import logging.config
from decimal import Decimal

import redis

from util import *
import socket

#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

FX_LOSS_PNALTY = 0
MSE_PENALTY = 0
INSENSITIVE_BORDER = 0

class ConfClass():
    def __init__(self):
        self.CONF_TYPE = "LSTM"

        self.JPY_FIX = 130 #固定のドル円レート　BTC用

        self.SEED = 0
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

        self.BET_TERM = 2 # betする間隔(秒)
        self.PRED_TERM = 20  # 予測する間隔:DB1_TERMが2でPRED_TERMが15なら30秒後の予測をする

        self.BET_SHIFT = 2 #予測実施時の秒シフト

        self.START_TERM = 0 #予測開始レートを何term後にするか thinkmarketsでは発注から約定まで数秒までかかるため、その対応としての設定
        self.END_TERM = 0  # 予測終了レートをPRED_TERMから何term後にするか thinkmarketsでは決済注文から約定まで数秒までかかるため、その対応としての設定

        self.START_TERM_STR = "" if self.START_TERM == 0 else "_ST" + str(self.START_TERM)
        self.END_TERM_STR = "" if self.END_TERM == 0 else "_ET" + str(self.END_TERM)

        self.DB_HOST = 'localhost'
        self.DB_NO = 3 #dukascopy data
        #self.DB_EVAL_NO = 0 #highlow data(GBPJPY)/oanda data(USDJPY)
        #self.DB_EVAL_NO = 1 #dukascopy
        self.DB_EVAL_NO = 2 #oanda data(GBPJPY)/oanda tick data(USDJPY)

        # Fake用DB 長い足を使わない場合とのテスト結果比較のため
        # 長い足を特徴量とする場合、テストデータが少なくなって、長い足を使わない場合とのテスト結果が正確ではなくなるため
        # 使用しない場合 0にする
        self.DB_FAKE_TERM = 0
        self.DB_FAKE_LIST = make_db_list(self.SYMBOL, self.DB_FAKE_TERM, self.BET_TERM)
        self.DB_FAKE_INPUT_LEN = 24
        print(self.DB_FAKE_LIST)

        self.DB_SYMBOLS = [self.SYMBOL, self.SYMBOL, self.SYMBOL, self.SYMBOL, self.SYMBOL,]

        self.DB_SYMBOLS_STR = "" #self.DB_SYMBOLSの中で、self.SYMBOLと異なる場合のみモデル名に反映させる

        # inputするデータの秒間隔
        self.DB1_TERM = 2
        self.TERM = get_decimal_multi(self.PRED_TERM, self.DB1_TERM)

        #DB1のデータを学習対象としない場合はTrue
        self.DB1_NOT_LEARN = False
        self.DB1_NOT_LEARN_STR = "_DB1NL" if self.DB1_NOT_LEARN else ""

        # 読み込み対象のDB名を入れていく
        # 例:db_termが60でsが2なら60秒間隔でデータ作成していることは変わらないが
        # 位相が"01:02,02:02,03:02..." と"01:00,02:00,03:00..."で異なっている
        self.DB1_LIST = make_db_list(self.DB_SYMBOLS[0], self.DB1_TERM, self.BET_TERM)
        print(self.DB1_LIST)

        self.DB2_TERM = 10
        self.DB2_LIST = make_db_list(self.DB_SYMBOLS[1], self.DB2_TERM, self.BET_TERM)
        print(self.DB2_LIST)
        if self.DB2_TERM != 0 and self.DB_SYMBOLS[1] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "2" + self.DB_SYMBOLS[1][0] + self.DB_SYMBOLS[1][3]

        self.DB3_TERM = 60
        self.DB3_LIST = make_db_list(self.DB_SYMBOLS[2], self.DB3_TERM, self.BET_TERM)
        print(self.DB3_LIST)
        if self.DB3_TERM != 0 and self.DB_SYMBOLS[2] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "3" + self.DB_SYMBOLS[2][0] + self.DB_SYMBOLS[2][3]

        self.DB4_TERM = 0
        self.DB4_LIST = make_db_list(self.DB_SYMBOLS[3], self.DB4_TERM, self.BET_TERM)
        print(self.DB4_LIST)
        if self.DB4_TERM != 0 and self.DB_SYMBOLS[3] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "4" + self.DB_SYMBOLS[3][0] + self.DB_SYMBOLS[3][3]

        self.DB5_TERM = 0
        self.DB5_LIST = make_db_list(self.DB_SYMBOLS[4], self.DB5_TERM, self.BET_TERM)
        print(self.DB5_LIST)
        if self.DB5_TERM != 0 and self.DB_SYMBOLS[4] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "5" + self.DB_SYMBOLS[4][0] + self.DB_SYMBOLS[4][3]

        if self.DB_SYMBOLS_STR != "":
            self.DB_SYMBOLS_STR = "_" + self.DB_SYMBOLS_STR

        # LSTM8用
        self.DB_VOLUME_TERM = 2
        self.DB_VOLUME_LIST = make_db_list(self.DB_SYMBOLS[0], self.DB_VOLUME_TERM, self.BET_TERM)
        self.DB_VOLUME_INPUT_LEN = 15
        print(self.DB_VOLUME_LIST)

        TMP_DB_TERMS = [self.DB1_TERM,self.DB2_TERM,self.DB3_TERM,self.DB4_TERM,self.DB5_TERM,]
        self.DB_TERMS = []
        for dt in TMP_DB_TERMS:
            if dt != 0:
                self.DB_TERMS.append(dt)
        self.DB_TERM_STR = list_to_str(self.DB_TERMS)

        self.INPUT_LEN = [
            300,
            300,
            240,
            #48,
            ]
        self.INPUT_LEN_STR = list_to_str(self.INPUT_LEN)

        self.LSTM_UNIT =[
            30,
            30,
            24,
            #5
        ]
        self.LSTM_UNIT_STR = list_to_str(self.LSTM_UNIT)

        self.DENSE_UNIT =[
            48,
            24,
            12,
        ]
        self.DENSE_UNIT_STR = list_to_str(self.DENSE_UNIT)

        #MT5で取得する分足を用いて特徴量とする
        #[足の長さ,データの長さ,unit数, 特徴量,DB名, 複数入力の場合に別のLSTMで学習する場合(SEPARATE_FLG):TRUE]を1セットとしてリスト形式で入力
        #self.FOOT_DBS = [[300, 300, 30, "d1_ehd1-1_eld1-1", "USDJPY_300_FOOT", True],]
        self.FOOT_DBS = [] #特徴量として利用しない場合は空のリストとする

        self.FOOT_STR = "_FDB" if len(self.FOOT_DBS) != 0 else ""
        for idx,db in enumerate(self.FOOT_DBS):
            d_term,d_len,d_unit,d_x,db_name,separate_flg = db
            s_str = "-SEP" if separate_flg else ""
            if idx == 0:
                self.FOOT_STR = self.FOOT_STR + str(d_term) + "-" + str(d_len) + "-" + str(d_unit) + "-" + d_x + s_str
            else:
                self.FOOT_STR = self.FOOT_STR + "_" + str(d_term) + "-" + str(d_len) + "-" + str(d_unit) + "-" + d_x + s_str

        self.DELETE_TEST_FLG = False
        self.DELETE_LEARN_FLG = True
        if socket.gethostname() == 'ub3':
            #ub3はメモリが多いので削除しない
            self.DELETE_LEARN_FLG = False

        #self.ANSWER_DB = "USDJPY_2_0_ANSWER60_30-satr-5_MARKET_MARKET_SPR1"
        self.ANSWER_DB = ""
        self.ANSWER_DB_FILE = "/db2/answer/" + self.ANSWER_DB + "_20221101-20230501.pickle"
        self.ANSWER_DB_TYPE = "MARKET_MARKET"

        """
        self.ANSWER_ATR_COL = "" #使用するatrの列名
        self.ANSWER_ATR = [] #絞り込むatrの値をハイフン区切りにする
        #self.ANSWER_ATR_NEED_LEN = 0 if self.ANSWER_ATR_COL == "" else calc_need_len([self.ANSWER_ATR_COL], self.BET_TERM) #すでにmake_fx_answerでデータがつながっていることは確認ずみ
        self.ANSWER_ATR_STR = "_" + self.ANSWER_ATR_COL + "_" + list_to_str(self.ANSWER_ATR, spl="_") if len(self.ANSWER_ATR) != 0 else ""
        """

        self.OPTIONS_DB = self.SYMBOL + "_" + str(self.BET_TERM) + "_IND"
        #self.OPTIONS_DB = self.SYMBOL + "_" + str(self.BET_TERM) + "_IND60"
        #使用するオプション(indexなど)を配列で格納
        #self.OPTIONS =["300-adx14",]
        self.OPTIONS =[]
        self.OPTIONS_NEED_LEN = calc_need_len(self.OPTIONS, self.BET_TERM)
        self.OPTIONS_STR = "" if len(self.OPTIONS) == 0 else "_OPT" + list_to_str(self.OPTIONS, "+")

        self.ATR_COL = ""
        #self.ATR_COL = "30-satr-5"  # 使用するatrの列名
        #self.ATR_COL = str(self.TERM) + "-satr-5"  # 使用するatrの列名
        #self.ATR_COL = str(self.TERM) + "-satr-1"  # 使用するatrの列名
        self.ATR = [] #絞り込むatrの値をハイフン区切りにする
        self.ATR_NEED_LEN = 0 if self.ATR_COL == "" else calc_need_len([self.ATR_COL], self.BET_TERM)
        self.ATR_STR = "_" + self.ATR_COL + "_" + list_to_str(self.ATR, spl="_") if len(self.ATR) != 0 else ""

        #各指標をテスト時に確認したり、データを絞るために使用
        self.IND_COLS_DB = self.SYMBOL + "_" + str(self.BET_TERM) + "_IND"

        """
        self.IND_COLS = [
            "10-satr-5","10-satr-15","10-satr-45",
            "60-satr-5","60-satr-15","60-satr-45","300-satr-5","300-satr-15","300-satr-45",
            "1800-fsatr-5", "1800-fsatr-15", "1800-fsatr-45", "1800-fsatr-70",
        ]
        """

        self.IND_COLS = []

        self.IND_NEED_LENS = []
        for col in self.IND_COLS:
            self.IND_NEED_LENS.append(calc_need_len([col], self.BET_TERM))

        # 絞り込む値をIND_COLS分ハイフン区切りにして指定する いずれかに合致すればOK
        self.IND_RANGES =  [
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [], []
        ]

        self.IND_STR = ""
        for i, col in enumerate(self.IND_COLS):
            ranges = self.IND_RANGES[i]
            if len(ranges) != 0:
                self.IND_STR = self.IND_STR + "_" + col

            for r in ranges:
                self.IND_STR = self.IND_STR + "_" + str(r)
        if self.IND_STR != "":
            self.IND_STR = "_IND_" + self.IND_STR

        #以下、意識すべき水平線上の過去レートの数を特徴量に加える場合
        self.HOR_LEARN_ON = False #True:特徴量に加える場合
        self.HOR_DB_CORE = "" #参照もしない場合は空文字にする
        #self.HOR_DB_CORE = "60_1440_0.01"
        self.HOR_DATA_NUM = 5 #特徴量を上下いくつ使用するか 5なら上5下5真ん中1で合計11
        self.HOR_DB = self.SYMBOL + "_" + self.HOR_DB_CORE + "_HOR"
        self.HOR_STR = "_HOR" + self.HOR_DB_CORE + "_" + str(self.HOR_DATA_NUM)if self.HOR_DB_CORE != "" else ""
        self.HOR_WIDTH = float(self.HOR_DB_CORE.split("_")[2]) if self.HOR_LEARN_ON else None #同一線上とみなすレート幅
        self.HOR_TERM = int(self.HOR_DB_CORE.split("_")[0]) if self.HOR_LEARN_ON  else None #データ作成に使用した分足(sec)
        if self.HOR_LEARN_ON and self.HOR_DB_CORE == "":
            print("HOR_LEARN_ON is True, but HOR_DB_CORE is empty")
            exit(1)

        #以下、高値安値との差を特徴量に加える場合
        self.HIGHLOW_DB_CORE = "" #特徴量に加えない場合は空文字にする
        #self.HIGHLOW_DB_CORE = "60_60_24"
        self.HIGHLOW_DATA_NUM = 24 #過去どれほど遡った高値安値を参照するか
        self.HIGHLOW_DB = self.SYMBOL + "_" + self.HIGHLOW_DB_CORE + "_HIGHLOW"
        self.HIGHLOW_STR = "_HIGHLOW" + self.HIGHLOW_DB_CORE + "_" + str(self.HIGHLOW_DATA_NUM) if self.HIGHLOW_DB_CORE != "" else ""
        self.HIGHLOW_TERM = int(self.HIGHLOW_DB.split("_")[1]) if self.HIGHLOW_DB_CORE != "" else ""

        #以下、LSTMに与えるデータをDenseに特徴量として与える場合
        self.NON_LSTM_LIST = [] #空のリストの場合は特徴量に加えない
        #リストに{"db_no":1, "inputs":[], "length":100}のような形で使用するDB番号と特徴量、データ長を列挙する
        #self.NON_LSTM_LIST = [
            #{"db_no":1, "inputs":["d1"], "length":300},
        #]
        self.NON_LSTM_STR = ""
        if len(self.NON_LSTM_LIST) != 0:
            for i, t in enumerate(self.NON_LSTM_LIST):
                if i == 0:
                    self.NON_LSTM_STR = "_NL" + str(t["db_no"]) + "_" + list_to_str(t["inputs"]) + "_" + str(t["length"])
                else:
                    self.NON_LSTM_STR = self.NON_LSTM_STR + "-" + str(t["db_no"]) + "_" + list_to_str(t["inputs"]) + "_" + str(t["length"])

        #以下、オアンダのオープンオーダーデータを特徴力として加える場合
        self.OANDA_ORD_DB = "" #特徴量に加えない場合は空文字にする
        #self.OANDA_ORD_DB = self.SYMBOL + "_OANDA_ORD"
        self.OANDA_ORD_NUM = 10 #現在レートが属するレンジより何本分上および下のデータを加えるか 例) 5なら上が5下が5、現在レート分が1なので合計11レンジのデータを特徴量とする
        self.OANDA_ORD_STR = "_OORD" + str(self.OANDA_ORD_NUM) if self.OANDA_ORD_DB != "" else ""

        #以下、オアンダのオープンポジションデータを特徴力として加える場合
        self.OANDA_POS_DB = "" #特徴量に加えない場合は空文字にする
        #self.OANDA_POS_DB = self.SYMBOL + "_OANDA_POS"
        self.OANDA_POS_NUM = 5 #現在レートが属するレンジより何本分上および下のデータを加えるか 例) 5なら上が5下が5、現在レート分が1なので合計11レンジのデータを特徴量とする
        self.OANDA_POS_STR = "_OPOS" + str(self.OANDA_POS_NUM) if self.OANDA_POS_DB != "" else ""

        # 以下、各分足での指標を特徴力として加える場合
        self.IND_FOOT_DB = self.SYMBOL + "_" + str(self.BET_TERM) + "_IND_FOOT"  # 各足の指標を保持するDB
        self.IND_FOOT_COL = []
        self.IND_FOOT_STR = "_IFC" + list_to_str(self.IND_FOOT_COL, spl="-") if len(self.IND_FOOT_COL) != 0 else ""

        self.SUB = [] #答えのレートと現在レートの差の絞り込む絶対値をハイフン区切りにする
        self.SUB_STR = "_SUB" + list_to_str(self.SUB, spl="_") if len(self.SUB) != 0 else ""

        self.DROP = 0.0
        self.DROP_STR = "" if (self.DROP == 0.0 or self.DROP == 0) else "_DROP" + str(self.DROP)

        self.L_K_RATE = "" #lstm kernel_regularizer # 正則化タイプ - 値
        self.L_R_RATE = "" #lstm recurrent_regularizer
        self.L_D_RATE = "" #dense kernel_regularizer

        self.L_K_STR = "_LK-" + str(self.L_K_RATE) if self.L_K_RATE != "" else ""
        self.L_R_STR = "_LR-" + str(self.L_R_RATE) if self.L_R_RATE != "" else ""
        self.L_D_STR = "_LD-" + str(self.L_D_RATE) if self.L_D_RATE != "" else ""

        self.LSTM_DO = 0.0
        self.LSTM_DO_STR = "" if (self.LSTM_DO == 0.0 or self.LSTM_DO == 0) else "_LSTMDO" + str(self.LSTM_DO)

        self.L_DO = 0.0
        self.L_DO_STR = "" if (self.L_DO == 0.0 or self.L_DO == 0) else "_LDO" + str(self.L_DO)

        self.L_RDO = 0.0
        self.L_RDO_STR = "" if (self.L_RDO == 0.0 or self.L_RDO == 0) else "_LRDO" + str(self.L_RDO)


        self.DB_EXTRA_1 = "" #1秒データを使用する場合DB名を入れる　使用しなければ空文字
        self.DB_EXTRA_1_LEN = 360
        self.DB_EXTRA_1_UNIT = 36

        self.FX = True
        self.FX_TICK_DB =  self.SYMBOL + "_" + str(self.BET_TERM) + "_0_TICK"
        self.FX_REAL_SPREAD_FLG = True #Oandaのスプレッドデータを使用する場合True

        ### 取引会社ごとの設定 START ###

        self.FX_FUND = 600000 #for oanda
        #self.FX_FUND = 900000 #for threetrader

        #self.FX_FUND = 600000
        self.FX_LEVERAGE = 25 #for 国内FX
        #self.FX_LEVERAGE = 500 #for threetrader
        #self.FX_LEVERAGE = 200 #for mexc feature
        #self.FX_LEVERAGE = 1 #for mexc spot

        #取引量(lot):
        # 0の場合は　FX_FUND * FX_LEVERAGE / rate / (TERM / BET_TERM)
        # 0でない場合は　FX_FIX_POSITION。 ただし、FX_FUND * FX_LEVERAGE - (既に保持しているポジション数 * FX_FIX_POSITION * rate)  の値が0以上であれば(資金余裕があれば)新規ポジションを持てる
        #self.FX_FIX_POSITION = 10000 #for oanda
        #self.FX_FIX_POSITION = 50000 #for threetrader
        #self.FX_FIX_POSITION = 100000 #for thinkmarkets
        #self.FX_FIX_POSITION = 50000 # moneypartners(USDJPY)
        self.FX_FIX_POSITION = 0

        #self.START_MONEY = 620000 #for oanda
        #self.START_MONEY = 3600000 #for oanda
        #self.START_MONEY = 3600000 #for threetrader
        self.START_MONEY = 1000000

        #self.ADJUST_PIPS = -0.0011 #testLstmFX2_answer testLstmFX2_rgr_limit用 海外FXなどの場合はoandaよりスプレッドが広いので、その分のスプレッドを加味する
        self.ADJUST_PIPS = -0.002
        #self.ADJUST_PIPS = -0.000005
        #self.ADJUST_PIPS = -0.004
        #self.ADJUST_PIPS = 0.0

        self.IGNORE_MINUS_SPREAD = True #マイナススプレッドを対象外とする testLstmFX2_answer testLstmFX2_rgr_limit用

        self.BTCUSD_SPREAD_PERCENT = 0.0  #BTCUSDのMEXCでの取引手数料パーセント
        #self.BTCUSD_SPREAD_PERCENT = 0.0  #BTCUSDのMEXCでの取引手数料パーセント

        ### 取引会社ごとの設定 END ###

        self.FX_MORE_BORDER_RGR = 0.0
        self.FX_MORE_BORDER_CAT = 0.3

        self.FX_SINGLE_FLG = False #ポジションは一度に1つしか持たない
        self.FX_TARGET_SHIFT = [] #betするシフト
        self.FX_NOT_EXT_FLG = False #延長しない場合True
        self.FX_TP_SIG = 3 #takeprofit, stoplossするシグマ
        self.FX_SL_SIG = 3
        self.FX_SL_D = 1
        self.FX_LIMIT_SIG = 1 #指値注文の場合に使用するシグマ

        self.FX_TAKE_PROFIT = 0.01
        self.FX_STOP_LOSS = 0.1

        self.FX_TP_ATR = 1 #atrをもとにtakeprofit, stoplossを決める場合に、atrに掛ける値
        self.FX_SL_ATR = 1

        self.FX_TP_PRED = 5 #予想レートをもとにtakeprofit, stoplossを決める場合に、予想レートに掛ける値
        self.FX_SL_PRED = 5

        self.FX_MIN_TP_SL = None #takeprofit, stoplossを決める場合にFX会社によって決められている最低限の値幅 fxtf:0.08

        self.FX_TAKE_PROFIT_FLG = False #takeprofitするか
        self.FX_STOP_LOSS_FLG = False #stoplossするか

        self.FX_BORDER_ATR = None # ATRが突然上がった場合決済するATR　Noneは設定なし

        self.FX_NOT_EXT_MINUS = None#Noneでない場合、初回延長判断時にこの値より利益が少ないなら延長しない。最初の予想が外れているなら、延長判断に使う予想も外れいている可能性が高い

        self.FX_MAX_TRADE_SEC = None #最大取引時間 設定しない場合はNone

        self.FX_STOPLOSS_PER_SEC_FLG = False #True:Betしてから一定秒数ごとにレート変化に応じて回復見込みが薄ければ損切りする
        self.FX_STOPLOSS_PER_SEC = 4 #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に損切り判定する秒間隔
        self.FX_STOPLOSS_PER_SEC_DB_NO = 10 #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に参照するDB_NO
        self.FX_STOPLOSS_PER_SEC_DB_LIST = [p for p in range(30) if p!= 0 and Decimal(p) % Decimal(self.FX_STOPLOSS_PER_SEC) == 0 ] #FX_STOPLOSS_PER_SEC_FLG ==Trueの場合に参照するDB(秒数)
        self.FX_STOPLOSS_PER_SEC_DB_PREFIX = "" #DB名の接頭語: border + _UP など
        self.FX_STOPLOSS_PER_SEC_CHK_TIMES = [] #約定から何秒経過後に損切りするか

        self.BUY_FLG = True
        self.SELL_FLG = True

        self.TRADE_SHIFT = 1 #この秒数で割り切れるシフトでしか取引しない.  None:設定なし testLstmFX2_answer testLstmFX2_rgr_limit用

        self.RESTRICT_FLG = False # 取引制限がかかった状態にする
        self.RESTRICT_SEC = 32 #取引制限がかかる秒数
        self.REFER_FLG = False

        self.SAME_SHIFT_NG_FLG = True #True:既に同じシフトの建玉がある場合は新規注文しない testLstmFX2_rgr_limit.py用
        self.NG_SHIFT = 48

        self.FX_MAX_POSITION_CNT = 12 #保持可能な最大ポジション数 Noneなら制限なし 0なら指定なし

        #FX_MAX_POSITION_CNTの指定がない場合
        if self.FX_SINGLE_FLG:
            self.FX_MAX_POSITION_CNT = 1
        else:
            if self.FX_MAX_POSITION_CNT == 0:
                #指定がない場合
                if self.RESTRICT_FLG:
                    self.FX_MAX_POSITION_CNT =  get_decimal_divide(self.TERM, self.RESTRICT_SEC)
                else:
                    self.FX_MAX_POSITION_CNT = get_decimal_divide(self.TERM, self.BET_SHIFT)

                if self.FX_NOT_EXT_FLG:
                    #延長しない場合はいつまでもポジションを持ち続けることがないので制限なしにする
                    self.FX_MAX_POSITION_CNT = None

        # 本番データのスプレッドを加味したテストをする場合
        # 正解を出すときに使うSPREADに実データのスプレッドを使用する
        self.REAL_SPREAD_FLG = False
        self.REAL_SPREAD_EVAL_FLG = False #eval時に本番データを使う場合
        self.TRADE_FLG = False #実トレードした結果を加味する場合
        self.TRADE_ONLY_FLG = False #実トレードした結果のみを表示する場合

        self.TARGET_SPREAD_LISTS = [] #学習時に除外しないスプレッド
        self.TARGET_SPREAD_STR = "_TS" + list_to_str(self.TARGET_SPREAD_LISTS) if len(self.TARGET_SPREAD_LISTS) != 0 else ""

        self.TARGET_SPREAD_LISTS_TEST = [] #テスト時に除外しないスプレッド
        self.TARGET_SPREAD_PERCENT_LISTS_TEST = [] #テスト時に除外しないスプレッド％

        self.FORMER_LIST = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,]
        self.LATTER_LIST = [32,34,36,38,40,42,44,46,48,50,52,54,56,58,0,]

        self.EXCEPT_LIST_BY_SPERAD = {} #スプレッドごとの対象外の時間
        #self.EXCEPT_LIST_BY_SPERAD = {1:[0,7,8,9,10,11,12,13,14,15,16,17],}
        self.EXCEPT_DIV_LIST = []
        self.EXCEPT_DIVIDE_MIN = 0
        self.EXCEPT_DIVIDE_MAX = 0 #予想divideの上限値、これより大きい場合は取引しない
        self.DB_TRADE_NO = 8
        self.DB_TRADE_NAME = "GBPJPY_30_SPR_TRADE"

        self.PAYOUT = 1300 #30秒 spread
        #self.PAYOUT = 1000
        #self.PAYOUT = 960
        self.PAYOFF = 1000

        # 学習対象外時間(ハイローがやっていない時間)
        # FXの場合 スプレッドを見る限り
        # 開始時間については
        # 0時0分(マーケット開始から2:00経過)から取引するのがよさそう(サマータイムでない時期)
        # 23時0分(マーケット開始から2:00経過)から取引するのがよさそう(サマータイム時期)
        # 終了時間については
        # 21時までがよさそう(サマータイムでない時期)
        # 20時までがよさそう(サマータイム時期)
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

        self.ZERO_SEC_FLG = False #0秒のときのみ学習対象とする
        self.ZERO_SEC_STR = "" if self.ZERO_SEC_FLG == False else "_ZSEC"

        self.DRAWDOWN_LIST = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),
                 "drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),
                 "drawdown9over": (-90000, -1000000),}
        self.SPREAD_LIST = {"spread0":(-1,0),"spread1":(0,1),"spread2":(1,2),"spread3":(2,3), "spread4":(3,4)
                ,"spread5":(4,5),"spread6":(5,6),"spread7":(6,7),"spread8":(7,8),"spread9":(8,9)
                , "spread10": (9, 10), "spread11": (10, 11),"spread12": (11, 12),"spread13": (12, 13), "spread14": (13, 14),"spread15": (14, 15), "spread16": (15, 16),"spread16Over":(16,1000),}
        self.DIVIDE_BEF_LIST = {"divide0":(-1,0),"divide0.1":(0,0.1),"divide0.2":(0.1,0.2),"divide0.3":(0.2,0.3),"divide0.4":(0.3,0.4),"divide0.5":(0.4,0.5),"divide0.5over": (0.5, 10000),}
        self.DIVIDE_PREV_LENGTH = 15
        self.DIVIDE_LIST = {"divide0.01":(-1,0.01),"divide0.02":(0.01,0.02),"divide0.03":(0.02,0.03),"divide0.04":(0.03,0.04), "divide0.05":(0.04,0.05)
                        ,"divide0.06":(0.05,0.06),"divide0.07":(0.06,0.07),"divide0.08":(0.07,0.08),"divide0.09":(0.08,0.09),
                        "divide0.1":(0.09,0.1),"divide0.2":(0.1,0.2),"divide0.3":(0.2,0.3),"divide0.4":(0.3,0.4), "divide0.5":(0.4,0.5)
                        ,"divide0.6":(0.5,0.6),"divide0.7":(0.6,0.7),"divide0.8":(0.7,0.8),"divide0.9":(0.8,0.9)
                        , "divide1.0": (0.9, 1.0), "divide1.1": (1.0, 1.1), "divide1.2": (1.1, 1.2), "divide1.3": (1.2, 1.3),"divide1.4":(1.3,1.4)
                        , "divide1.5": (1.4, 1.5), "divide1.6": (1.5, 1.6),"divide1.7":(1.6,1.7),"divide1.8":(1.7,1.8),"divide1.9":(1.8,1.9),"divide2.0":(1.9,2.0),"divide2.0over": (2.0, 10000),
                                   }
        self.DIVIDE_AFT_LIST = {"divide0.5":(-1,0.5),"divide1":(0.5,1.0),"divide2":(1,2),"divide3":(2,3), "divide4":(3,4)
                        ,"divide5":(4,5),"divide6":(5,6),"divide7":(6,7),"divide8":(7,8),"divide9":(8,9),"divide10":(9,10)
                        , "divide11": (10, 11),"divide12":(11,12),"divide13":(12,13),"divide13over": (13, 10000),
                                   }
        self.DIVIDE_DIV_LIST = { "divide-0.5": (-1.0, -0.5),"divide-1.0": (-2.0, -1.0),"divide-2.0": (-3.0, -2.0),"divide-3.0": (-4.0, -3.0),"divide-4.0": (-5.0, -4.0),
                "divide-5.0": (-6.0, -5.0),"divide-6.0": (-7.0, -6.0),
                "divide-7.0over": (-10000, -7.0),
                "divide0.5": (0.5, 1.0),"divide1.0": (1.0, 2.0),"divide2.0": (2.0, 3.0),"divide3.0": (3.0, 4.0),"divide4.0": (4.0, 5.0),
                "divide5.0": (5.0, 6.0),"divide6.0": (6.0, 7.0),
                "divide7.0over": (7.0, 10000),
                }
        # 学習方法
        self.METHODS ={
            "NORMAL":0, "LSTM":1, "LSTM2":2, #LSTM予想を参考にさらに予想させる RGR用
            "LSTM3":3,#秒データを参考にする
            "LSTM4":4,  # 分データを参考にする
            "LSTM5":5,  # 時間, 分データを参考にする
            "LSTM6":6,  # 秒、分データを参考にする
            "LSTM7":7,  # 秒、分、時データを参考にする
            "LSTM8":8,  # 秒、分、時データに加えてボリューム(tick数)も参考にする
            "LSTM9":9,  # LSTM予想を参考にさらに予想させる Category用
            "LSTM10":10,  # 秒、分、時, 曜日データを参考にする
            "TCN":11,
            "TCN7":12,  # 秒、分、時データを参考にする
            #"LSTM7-SEC10":13, # 秒(10秒ごとに区切った値：０〜5)、分、時データを参考にする。 効力なしなので実装から削除
            #"LSTM-MAXMIN":14, #高値安値に近いか否かを特徴量にする
            #"LSTM7-MAXMIN": 15,  # 高値安値に近いか否かを特徴量にする
        }

        self.METHOD = "LSTM7"

        self.METHOD_STR = "_M" + str(self.METHODS[self.METHOD])

        self.TCN_KERNEL_SIZE = 1
        self.TCN_NB_STACKS = 1
        self.TCN_ARG_STR = "" if (self.METHOD != "TCN" and self.METHOD != "TCN7") else \
            "_TCN-"  + str(self.TCN_KERNEL_SIZE) + "-" + str(self.TCN_NB_STACKS)

        self.RETURN_SEQ = False
        self.RETURN_SEQ_STR = "_RS" if self.RETURN_SEQ else ""

        #self.LSTM_TYPE = "LSTM" #LSTM or GRU or Bi(Bidirectional) or QRNN or TCN or LAYERNORM or PEEPHOLE
        #todo:LAYERNORMとPEEPHOLEはメモリを大量にくうので試したことがない(将来高性能GPUを揃えられたら試してみたい)
        #KSA-LSTM:keras-self-attentin → LSTMのモデル
        self.LSTM_TYPE = "LSTM"

        self.LSTM_LAYER_NUM = 1 #LSTM層の数

        ### self-attention設定ここから
        self.SELF_AT_LAYER_NUM = 1 #self-attentionの層の数
        self.SELF_AT_LAYER_NUM_STR = ""

        self.SELF_AT_INPUT_PLUS = True
        self.SELF_AT_INPUT_PLUS_STR = ""

        self.SELF_AT_NORMAL = "" #BATCH or LAYER
        self.SELF_AT_NORMAL_STR = ""

        self.MHA_UNIT_NUM = 2
        self.MHA_UNIT_NUM_STR = ""

        self.MHA_HEAD_NUM = 1
        self.MHA_HEAD_NUM_STR = ""

        self.KSA_UNIT_NUM = 2
        self.KSA_UNIT_NUM_STR = ""

        if self.LSTM_TYPE in ["MHA-LSTM","LSTM-MHA","LSTM-MHA-CNN", "MHA-TCN"]:
            self.MHA_UNIT_NUM_STR = "_MHAUN" + str(self.MHA_UNIT_NUM)
            self.MHA_HEAD_NUM_STR = "_MHAHN" + str(self.MHA_HEAD_NUM)

            self.SELF_AT_LAYER_NUM_STR = "_SALN" + str(self.SELF_AT_LAYER_NUM)

            self.SELF_AT_INPUT_PLUS_STR = "_SAIP" if self.SELF_AT_INPUT_PLUS else ""
            if self.SELF_AT_NORMAL == "BATCH":
                self.SELF_AT_NORMAL_STR = "_SANB"
            elif self.SELF_AT_NORMAL == "LAYER":
                self.SELF_AT_NORMAL_STR = "_SANL"

        if self.LSTM_TYPE in ["KSA-LSTM","LSTM-KSA","LSTM-KSA-CNN", "KSA-TCN"]:
            self.KSA_UNIT_NUM_STR = "_KSAUN" + str(self.KSA_UNIT_NUM)
            self.SELF_AT_LAYER_NUM_STR = "_SALN" + str(self.SELF_AT_LAYER_NUM)

            self.SELF_AT_INPUT_PLUS_STR = "_SAIP" if self.SELF_AT_INPUT_PLUS else ""
            if self.SELF_AT_NORMAL == "BATCH":
                self.SELF_AT_NORMAL_STR = "_SANB"
            elif self.SELF_AT_NORMAL == "LAYER":
                self.SELF_AT_NORMAL_STR = "_SANL"

        ### self-attention設定ここまで

        ### CNN 設定ここから
        self.CNN_INPUT_PLUS = True
        self.CNN_INPUT_PLUS_STR = ""

        self.CNN_UNIT_NUM = 4
        self.CNN_UNIT_NUM_STR = ""

        self.CNN_NORMAL = "" #BATCH or LAYER
        self.CNN_NORMAL_STR = ""

        if self.LSTM_TYPE in ["LSTM-KSA-CNN","LSTM-MHA-CNN", ]:
            self.CNN_UNIT_NUM_STR = "_CNNUN" + str(self.CNN_UNIT_NUM)

            self.CNN_INPUT_PLUS_STR = "_CNNIP" if self.CNN_INPUT_PLUS else ""
            if self.CNN_NORMAL == "BATCH":
                self.CNN_NORMAL_STR = "_CNNNB"
            elif self.CNN_NORMAL == "LAYER":
                self.CNN_NORMAL_STR = "_CNNNL"

        ### CNN 設定ここまで

        self.WINDOW_SIZE = 2
        self.WINDOW_SIZE_STR = "" if self.LSTM_TYPE != "QRNN" else "_WS" + str(self.WINDOW_SIZE)

        self.SEC_OH_LEN_FIX_FLG = True
        self.SEC_OH_LEN_FIX_STR = ""

        if self.SEC_OH_LEN_FIX_FLG:
            #BET_SHIFTで60を割らない
            self.SEC_OH_LEN = 60
            self.SEC_OH_LEN_FIX_STR = "_SFIX"
        else:
            self.SEC_OH_LEN = int(Decimal("60") / Decimal(str(self.BET_SHIFT))) # 秒データをone-hotで入力するためのone-hotのながさ

        self.MIN_OH_LEN = 60 # 分データをone-hotで入力するためのone-hotのながさ
        self.HOUR_OH_LEN = 24
        self.WEEK_OH_LEN = 35

        self.LSTM9_INPUTS = [5,10] #参考にする予想を、予想時からどれぐらい以前にするか 5なら2×5で10秒前の予想
        self.LSTM9_PRED_DBS = ["2D",] #参考にする予想のDB内でのキー test時は複数記述
        self.LSTM9_PRED_DB_DEF = "2D" #参考にする予想のDB内でのキー test時はtestLstm2.pyなどでDataSequence2.set_db9_nameメソッドを呼んで適宜変更する
        self.LSTM9_USE_CLOSE = False
        self.LSTM9_INPUTS_STR = ""
        if self.METHOD == "LSTM9":
            for idx, ipt in enumerate(self.LSTM9_INPUTS):
                if idx == 0:
                    self.LSTM9_INPUTS_STR = "_INPUT9" + "-" + str(ipt)
                else:
                    self.LSTM9_INPUTS_STR = self.LSTM9_INPUTS_STR + "-" + str(ipt)
            if self.LSTM9_USE_CLOSE:
                self.LSTM9_INPUTS_STR = self.LSTM9_INPUTS_STR + "-C"

        self.MIXTURE_NORMAL = False #混合ガウス過程
        self.MIXTURE_NORMAL_NUM = 30 # num_components
        self.MIXTURE_NORMAL_STR = "_MN" + str(self.MIXTURE_NORMAL_NUM) if self.MIXTURE_NORMAL else ""

        self.DIVIDE_MAX = 0
        self.DIVIDE_MIN = 0

        self.DIVIDE_ALL_FLG = False #直前のレートから最古のレートまでのdivideを特徴量とする
        self.DIRECT_FLG = False #divideではなく、直接レートを予想する そのため、特徴量はclose値とする
        self.NOW_RATE_FLG = False #入力に予想時のレートをくわえる

        # バッチサイズについて なるべく小さい方が精度がでる 最大で2048
        # see:https://wandb.ai/wandb_fc/japanese/reports/---Vmlldzo1NTkzOTg
        #self.BATCH_SIZE = 1024 * 10
        self.BATCH_SIZE = 1024 * 5
        #self.BATCH_SIZE = 1024 * 1 * 6 #テスト時にエラーでた場合はバッチサイズさげる

        #以下、モデルのfit_generatorの引数設定
        self.WORKERS = 0
        self.MAX_QUEUE_SIZE = 0

        self.WORKERS_STR = "_W" + str(self.WORKERS) if self.WORKERS != 0 else ""
        self.MAX_QUEUE_SIZE_STR = "_MQS" + str(self.MAX_QUEUE_SIZE) if self.MAX_QUEUE_SIZE != 0 else ""

        self.LEARNING_TYPES ={
            "CATEGORY":1, "CATEGORY_BIN":2, "CATEGORY_BIN_UP":3, "CATEGORY_BIN_DW":4, "CATEGORY_BIN_BOTH":5, "CATEGORY_BIN_FOUR":6,
            "REGRESSION_SIGMA":7, "REGRESSION":8, "REGRESSION_UP":9, "REGRESSION_DW":10, "REGRESSION_OCOPS":11,
            "CATEGORY_OCOPS":12, "CATEGORY_BIN_UP_IFD":13, "CATEGORY_BIN_DW_IFD":14, "CATEGORY_BIN_UP_IFO":15, "CATEGORY_BIN_DW_IFO":16,
            "CATEGORY_BIN_UP_IFDSF":17, "CATEGORY_BIN_DW_IFDSF":18, "CATEGORY_BIN_UP_TP":19, "CATEGORY_BIN_DW_TP":20,
            "CATEGORY_BIN_UP_OCO":21, "CATEGORY_BIN_DW_OCO":22, "CATEGORY_BIN_UP_OCOA":23, "CATEGORY_BIN_DW_OCOA":24,
        }

        self.LEARNING_TYPE = "CATEGORY"
        self.LEARNING_TYPE_STR = "_LT" + str(self.LEARNING_TYPES[self.LEARNING_TYPE])

        if self.LEARNING_TYPE in ["REGRESSION_UP", "REGRESSION_DW", "REGRESSION_OCOPS", "CATEGORY_OCOPS" ]:
            self.ANSWER_STR = "_ASW-" + self.ANSWER_DB_TYPE if self.ANSWER_DB != "" else ""
        else:
            self.ANSWER_STR = ""

        #self.INPUT_DATAS = ["d1",]
        self.INPUT_DATAS = ["d1","d1_ehd1-1_eld1-1","d1_ehd1-1_eld1-1",  ]

        if len(self.INPUT_LEN) != len(self.INPUT_DATAS):
            print("INPUT_DATAS length is incorrect!")
            exit(1)

        self.INPUT_DATA_LENGTHS = []
        for tmp_ipt in self.INPUT_DATAS:
            # INPUT_DATAを算出するのに必要なデータ長 例えば、d1なら2 -1 adx14なら14 * 2 = 28 -1 INPUT_DATAにnanが入っている可能性があるのでデータ抽出時に除外するため
            self.INPUT_DATA_LENGTHS.append(get_max_need_len(tmp_ipt.split("_")))

        #self.INPUT_SEPARATE_FLG = False #特徴量を別々のLSTMで学習させる
        self.INPUT_SEPARATE_FLG = True #特徴量を別々のLSTMで学習させる

        self.INPUT_DATA_STR = ""
        tmp_ipt_dict = {}
        for i, tmp_ipt in enumerate(self.INPUT_DATAS):
            if (tmp_ipt in tmp_ipt_dict.keys()) == False:
                tmp_ipt_dict[tmp_ipt] = [i + 1]
            else:
                tmp_ipt_dict[tmp_ipt].append(i + 1)

        for k,v in tmp_ipt_dict.items():
            self.INPUT_DATA_STR = self.INPUT_DATA_STR + "_" + k + "_" + list_to_str(v, spl="")

        for tmp_ipt in self.INPUT_DATAS:
            ipt_lists = tmp_ipt.split("_")
            if len(ipt_lists) > 1:
                if self.INPUT_SEPARATE_FLG:
                    self.INPUT_DATA_STR = self.INPUT_DATA_STR + "-SEP"
                    break

        # d:d(変化率)を求める sub:sub(差)を求める
        self.OUTPUT_TYPE = "d"
        self.OUTPUT_MULTI = 1 #Regression で　OUTPUT_TYPE=subの場合のsubの倍率

        if self.LEARNING_TYPE == "CATEGORY_BIN": #pips0以上と以下で分けるので強制的にsubとする
            self.OUTPUT_TYPE = "sub"

        self.OUTPUT_TYPE_STR = "_OT-" + self.OUTPUT_TYPE if self.OUTPUT_MULTI == 1 else "_OT-" + self.OUTPUT_TYPE + "-M" + str(self.OUTPUT_MULTI)

        #変化の基準の直近データをCloseにする場合にTrue　Falseの場合はOUTPUT_DATAを基準の直近データとする
        self.OUTPUT_DATA_BEF_C = False
        self.OUTPUT_DATA_BEF_C_STR = "_ODBC" if self.OUTPUT_DATA_BEF_C else ""

        # regression系の場合  c:変化の基準をcloseとする smam60:変化の基準を直近60秒のsmamと予想時間直前の60秒のsmamとする
        # category系の場合  c:up or dwの基準をcloseの値とする smam60:基準を直近60秒のsmamとする
        self.OUTPUT_DATA = "c"
        #self.OUTPUT_DATA_STR = "_" + self.OUTPUT_DATA if (self.OUTPUT_DATA != "" and self.LEARNING_TYPE in ["REGRESSION","REGRESSION_UP","REGRESSION_DW", "REGRESSION_OCOPS"])  else ""
        self.OUTPUT_DATA_STR = "_OD-" + self.OUTPUT_DATA + self.OUTPUT_DATA_BEF_C_STR
        self.OUTPUT_LIST = self.OUTPUT_DATA.split("_")

        # Category 予測において　SPREADではなく変化率で分類する場合
        # レートが153.846153846154(=0.001/0.0000065)のとき、0.001円上がるとDivideは0.065
        # ※Divie＝((X_after/X_before) -1) * 10000

        #BTCJPYの場合はスプレッドが大体0.01%なのでdivが0.1だと0.01%動いたことになる
        #USDJPYなら100.000でdiv0.1なら100.001に動いたことになる
        self.BORDER_DIV = 1.0
        self.SPREAD = 1

        self.BORDER_STR = ""
        if self.LEARNING_TYPE == "CATEGORY" or  self.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or self.LEARNING_TYPE == "CATEGORY_BIN_UP" or self.LEARNING_TYPE == "CATEGORY_BIN_DW":
            #"CATEGORY_BIN"の場合、強制的にSPREAD0以上か以下かで分けるのでself.BORDER_STR = ""のままとする
            if self.OUTPUT_TYPE == "sub":
                self.BORDER_STR = "_SPREAD" + str(self.SPREAD)
            elif self.OUTPUT_TYPE == "d":
                self.BORDER_STR = "_BDIV" + str(self.BORDER_DIV)

        self.OUTPUT = 0
        if self.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_OCOPS"]:
            self.OUTPUT = 3
        elif self.LEARNING_TYPE in ["CATEGORY_BIN","CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN_UP_IFD",
                                    "CATEGORY_BIN_DW_IFD", "CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO",
                                    "CATEGORY_BIN_UP_IFDSF","CATEGORY_BIN_DW_IFDSF","CATEGORY_BIN_UP_TP","CATEGORY_BIN_DW_TP",
                                    "CATEGORY_BIN_UP_OCO","CATEGORY_BIN_DW_OCO","CATEGORY_BIN_UP_OCOA","CATEGORY_BIN_DW_OCOA"]:
            self.OUTPUT = 2
        elif self.LEARNING_TYPE in ["REGRESSION_SIGMA", "REGRESSION_OCOPS"]:
            self.OUTPUT = 2  # 平均, β(=logα α=標準偏差) #1つ目がmuで2つ目がbeta(精度パラメーターとする)
        elif self.LEARNING_TYPE in ["REGRESSION","REGRESSION_UP","REGRESSION_DW"]:
            if self.OUTPUT_DATA != "":
                self.opt_lists = self.OUTPUT_DATA.split("_")
                self.OUTPUT = len(self.opt_lists)
            else:
                self.OUTPUT = 1

        self.LOSS_TYPES ={
            "C-ENTROPY":1, "B-ENTROPY":2, # ← for category
            "FXMSE":3, "FXMSE2":4, "RMSE":5, "MAE":6, "MSE":7, # ← for Regression
            "MSEC":8,  # MSEの誤差を自由に設定できる
            "IE":9, "NLL":10,# negative_log_likelihood
            "HUBER":11, "LOG_COSH":12, "HINGE":13, "SQUARED_HINGE":14, "POISSON":15,
        }

        self.LOSS_TYPE = "C-ENTROPY"

        self.LOSS_TYPE_STR = "_LT" + str(self.LOSS_TYPES[self.LOSS_TYPE])

        if self.MIXTURE_NORMAL:
            #MIXTURE_NORMALの場合強制的にNLL
            self.LOSS_TYPE = "NLL"

        if self.LOSS_TYPE in ["FXMSE", "FXMSE2"]:
            self.LOSS_STR = str(FX_LOSS_PNALTY)
        elif self.LOSS_TYPE in  ["MSEC"]:
            self.LOSS_STR = str(MSE_PENALTY)
        elif self.LOSS_TYPE in  ["IE"]:
            self.LOSS_STR = str(INSENSITIVE_BORDER)
        else:
            self.LOSS_STR = ""

        self.ACTIVATION_TYPES ={
            "relu":1, "elu":2,"selu":3,
            "gelu":4, "leaky_relu":5, "relu6":6,
            "crelu":7, "tanh":8, "sigmoid":9,
        }

        self.DENSE_ACTIVATION = "gelu"
        self.DENSE_ACTIVATION_STR = "_DA" + str(self.ACTIVATION_TYPES[self.DENSE_ACTIVATION])

        self.RNN_ACTIVATION = "tanh"
        self.RNN_ACTIVATION_STR = "_RA" + str(self.ACTIVATION_TYPES[self.RNN_ACTIVATION])

        self.RNN_REC_ACTIVATION = "sigmoid"
        self.RNN_REC_ACTIVATION_STR = "_RRA" + str(self.ACTIVATION_TYPES[self.RNN_REC_ACTIVATION])

        # optimizer:ADAM, ADABOUND, AMSBOUND
        self.OPT = "ADAM"

        self.INIT_STR = "_GU-GU-GU" #初期値の設定(R_I,D_I,O_I)によってモデル名のsuffixを変える
        #self.SUFFIX = "_200401_200912"
        #self.SUFFIX = "_201001_201612"
        self.SUFFIX = "_201601_202303"

        self.DIVIDE_LOG_FLG = False #変化率の対数をとる
        if self.DIVIDE_LOG_FLG:
            self.SUFFIX += "_DLOG"

        self.NORMAL_TYPE = "BATCH_NORMAL_LSTM2" # BATCH_NORMAL or LAYER_NORMAL or LAYER_NORMAL_LSTM(lstm計算後に適用) or BATCH_NORMAL_LSTM(lstm計算後に適用) or blank
        self.NORMAL_STR = ""

        if self.NORMAL_TYPE == "BATCH_NORMAL":
            self.NORMAL_STR = "_BN"
        elif self.NORMAL_TYPE == "LAYER_NORMAL":
            self.NORMAL_STR = "_LN"
        elif self.NORMAL_TYPE == "BATCH_NORMAL_LSTM":
            self.NORMAL_STR = "_BNL"
        elif self.NORMAL_TYPE == "LAYER_NORMAL_LSTM":
            self.NORMAL_STR = "_LNL"
        elif self.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
            self.NORMAL_STR = "_BNL2"
        elif self.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
            self.NORMAL_STR = "_LNL2"

        if self.DIVIDE_ALL_FLG:
            self.SUFFIX += "DIVIDE_ALL"

        if self.METHOD == "LSTM2":
            self.SUFFIX += "_GEN1"  # gen1はLSTM2の1世代目(2ならLSTM2の結果をさらに受けて予想する)

        if self.DIRECT_FLG:
            self.SUFFIX += "_DIRECT"

        if self.NOW_RATE_FLG:
            self.SUFFIX += "_NRATE"

        if self.DB_FAKE_TERM != 0:
            self.SUFFIX += "_DBFAKE-" + str(self.DB_FAKE_TERM) + "-" + str(self.DB_FAKE_INPUT_LEN)

        if self.METHOD == "LSTM8":
            self.SUFFIX += "_VOL-" + str(int(Decimal(str(self.DB_VOLUME_TERM)) * Decimal(str(self.DB_VOLUME_INPUT_LEN))))

        if self.DB_EXTRA_1 != "":
            self.SUFFIX += "_EXTRA1"

        self.FILE_PREFIX = ""
        self.MODEL_DIR = ""
        self.HISTORY_DIR = ""
        self.CHK_DIR = ""

        self.MODEL_DIR_LOAD = ""
        self.CHK_DIR_LOAD = ""
        self.HISTORY_PATH = ""

        self.LOAD_CHK_NUM = "0009"
        self.LOAD_CHK_PATH = ""

        self.DIVIDE_MAX_STR = "_DIVIDEMAX" + str(self.DIVIDE_MAX) if self.DIVIDE_MAX != 0 else ""
        self.DIVIDE_MIN_STR = "_DIVIDEMIN" + str(self.DIVIDE_MIN) if self.DIVIDE_MIN != 0 else ""

        self.EPOCH = 40
        self.LEARNING_RATE = 0.001
        self.LEARNING_RATE_STR = "_LRT" + str(self.LEARNING_RATE)

        self.TAG = ""
        if self.TAG != "":
            self.SUFFIX += "-" + self.TAG

        #学習データをシャッフル or ローテート
        self.DATA_SHUFFLES = {
            "SHUFFLE":1, "NOSHUFFLE":2, "ROTATE":3,
        }

        self.DATA_SHUFFLE = "SHUFFLE"
        self.DATA_SHUFFLE_STR = "_SHU" + str(self.DATA_SHUFFLES[self.DATA_SHUFFLE])

        self.SINGLE_FLG = False # CPUのみ、またはGPU1つしか搭載していない場合
        self.DEVICE = "1" # SINGLE_FLG = Trueのとき、学習するデバイスを指定 0 :RTX3090 1 :RTX3080

        # 0:新規作成
        # 1:modelからロード
        # 2:chekpointからロード
        self.LOAD_TYPE = 0
        self.LOADING_NUM = "40"
        self.LEARNING_NUM = "40"

        #LOAD_TYPEが1 or 2の場合は以下を設定する
        self.FILE_PREFIX_OLD = ""
        if self.LOAD_TYPE != 0 and self.FILE_PREFIX_OLD == "":
            print("FILE_PREFIX_OLD is not set")
            exit(1)

        #FILE_PREFIXを作成
        self.make_file_prefix()

        #self.myLogger = printLog(loggerConf)

    def change_sprad(self, spr):
        self.SPREAD = spr

    def change_restrict_sec(self, sec):
        self.RESTRICT_SEC = sec

    def change_real_spread_flg(self, flg):
        self.REAL_SPREAD_FLG = flg

    def change_fx_real_spread_flg(self, flg):
        self.FX_REAL_SPREAD_FLG = flg

    def change_learning_rate(self, rate):
        self.LEARNING_RATE = rate
        self.change_str()
        self.make_file_prefix()

    def change_lkrate_rate(self, rate):
        self.L_K_RATE = rate #lstm kernel_regularizer # 正則化タイプ - 値
        self.change_str()
        self.make_file_prefix()

    def change_lrrate_rate(self, rate):
        self.L_R_RATE = rate #lstm recurrent_regularizer
        self.change_str()
        self.make_file_prefix()

    def change_ldrate_rate(self, rate):
        self.L_D_RATE = ""  # dense kernel_regularizer
        self.change_str()
        self.make_file_prefix()

    def change_lstm_unit(self, lstm_unit_list):
        self.LSTM_UNIT = lstm_unit_list
        self.change_str()
        self.make_file_prefix()

    def change_dense_unit(self, dense_unit_list):
        self.DENSE_UNIT = dense_unit_list
        self.change_str()
        self.make_file_prefix()

    def change_drop_out(self, drop):
        self.DROP = drop
        self.change_str()
        self.make_file_prefix()

    def change_self_at_layer_num(self, self_at_layer_num):
        self.SELF_AT_LAYER_NUM = self_at_layer_num
        self.change_str()
        self.make_file_prefix()

    def change_self_at_input_plus(self, self_at_input_plus):
        self.SELF_AT_INPUT_PLUS = self_at_input_plus
        self.change_str()
        self.make_file_prefix()

    def change_self_at_normal(self, self_at_normal):
        self.SELF_AT_NORMAL = self_at_normal
        self.change_str()
        self.make_file_prefix()

    def change_mha_unit_num(self, mha_unit_num):
        self.MHA_UNIT_NUM = mha_unit_num
        self.change_str()
        self.make_file_prefix()

    def change_mha_head_num(self, mha_head_num):
        self.MHA_HEAD_NUM = mha_head_num
        self.change_str()
        self.make_file_prefix()

    def change_ksa_unit_num(self, ksa_unit_num):
        self.KSA_UNIT_NUM = ksa_unit_num
        self.change_str()
        self.make_file_prefix()

    def change_l_do(self, l_do):
        self.L_DO = l_do
        self.change_str()
        self.make_file_prefix()

    def change_l_rdo(self, l_rdo):
        self.L_RDO = l_rdo
        self.change_str()
        self.make_file_prefix()

    def change_lstm_do(self, lstm_do):
        self.LSTM_DO = lstm_do
        self.change_str()
        self.make_file_prefix()

    def change_db_terms(self, db1_term=0, db2_term=0, db3_term=0, db4_term=0, db5_term=0,):
        self.DB1_TERM = db1_term
        self.TERM = self.PRED_TERM * self.DB1_TERM

        self.DB1_LIST = make_db_list(self.DB_SYMBOLS[0], self.DB1_TERM, self.BET_TERM)
        print(self.DB1_LIST)

        self.DB2_TERM = db2_term
        self.DB2_LIST = make_db_list(self.DB_SYMBOLS[1], self.DB2_TERM, self.BET_TERM)
        print(self.DB2_LIST)
        if self.DB2_TERM != 0 and self.DB_SYMBOLS[1] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "2" + self.DB_SYMBOLS[1][0] + self.DB_SYMBOLS[1][3]

        self.DB3_TERM = db3_term
        self.DB3_LIST = make_db_list(self.DB_SYMBOLS[2], self.DB3_TERM, self.BET_TERM)
        print(self.DB3_LIST)
        if self.DB3_TERM != 0 and self.DB_SYMBOLS[2] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "3" + self.DB_SYMBOLS[2][0] + self.DB_SYMBOLS[2][3]

        self.DB4_TERM = db4_term
        self.DB4_LIST = make_db_list(self.DB_SYMBOLS[3], self.DB4_TERM, self.BET_TERM)
        print(self.DB4_LIST)
        if self.DB4_TERM != 0 and self.DB_SYMBOLS[3] != self.SYMBOL:
            self.DB_SYMBOLS_STR = self.DB_SYMBOLS_STR + "4" + self.DB_SYMBOLS[3][0] + self.DB_SYMBOLS[3][3]

        self.DB5_TERM = db5_term
        self.DB5_LIST = make_db_list(self.DB_SYMBOLS[4], self.DB5_TERM, self.BET_TERM)

    def change_str(self):
        self.L_DO_STR = "" if (self.L_DO == 0.0 or self.L_DO == 0) else "_LDO" + str(self.L_DO)
        self.L_RDO_STR = "" if (self.L_RDO == 0.0 or self.L_RDO == 0) else "_LRDO" + str(self.L_RDO)
        self.LSTM_DO_STR = "" if (self.LSTM_DO == 0.0 or self.LSTM_DO == 0) else "_LSTMDO" + str(self.LSTM_DO)
        self.LEARNING_RATE_STR = "_L-RATE" + str(self.LEARNING_RATE)
        self.L_K_STR = "_LK-" + str(self.L_K_RATE) if self.L_K_RATE != "" else ""
        self.L_R_STR = "_LR-" + str(self.L_R_RATE) if self.L_R_RATE != "" else ""
        self.L_D_STR = "_LD-" + str(self.L_D_RATE) if self.L_D_RATE != "" else ""
        self.LSTM_UNIT_STR = list_to_str(self.LSTM_UNIT)
        self.DENSE_UNIT_STR = list_to_str(self.DENSE_UNIT)
        self.DROP_STR = "" if (self.DROP == 0.0 or self.DROP == 0) else "_DROP" + str(self.DROP)

        if self.LSTM_TYPE in ["MHA-LSTM","LSTM-MHA","LSTM-MHA-CNN", "MHA-TCN"]:
            self.MHA_UNIT_NUM_STR = "_MHAUN" + str(self.MHA_UNIT_NUM)
            self.MHA_HEAD_NUM_STR = "_MHAHN" + str(self.MHA_HEAD_NUM)

            self.SELF_AT_LAYER_NUM_STR = "_SALN" + str(self.SELF_AT_LAYER_NUM)

            self.SELF_AT_INPUT_PLUS_STR = "_SAIP" if self.SELF_AT_INPUT_PLUS else ""
            if self.SELF_AT_NORMAL == "BATCH":
                self.SELF_AT_NORMAL_STR = "_SANB"
            elif self.SELF_AT_NORMAL == "LAYER":
                self.SELF_AT_NORMAL_STR = "_SANL"

        if self.LSTM_TYPE in ["KSA-LSTM","LSTM-KSA","LSTM-KSA-CNN", "KSA-TCN"]:
            self.KSA_UNIT_NUM_STR = "_KSAUN" + str(self.KSA_UNIT_NUM)
            self.SELF_AT_LAYER_NUM_STR = "_SALN" + str(self.SELF_AT_LAYER_NUM)

            self.SELF_AT_INPUT_PLUS_STR = "_SAIP" if self.SELF_AT_INPUT_PLUS else ""
            if self.SELF_AT_NORMAL == "BATCH":
                self.SELF_AT_NORMAL_STR = "_SANB"
            elif self.SELF_AT_NORMAL == "LAYER":
                self.SELF_AT_NORMAL_STR = "_SANL"

        if self.LSTM_TYPE in ["LSTM-KSA-CNN", "LSTM-MHA-CNN", ]:
            self.CNN_UNIT_NUM_STR = "_CNNUN" + str(self.CNN_UNIT_NUM)

            self.CNN_INPUT_PLUS_STR = "_CNNIP" if self.CNN_INPUT_PLUS else ""
            if self.CNN_NORMAL == "BATCH":
                self.CNN_NORMAL_STR = "_CNNNB"
            elif self.CNN_NORMAL == "LAYER":
                self.CNN_NORMAL_STR = "_CNNNL"

    def make_file_prefix(self):
        if self.METHOD in [
            "LSTM","BY","LSTM2","LSTM3","LSTM4","LSTM5","LSTM6","LSTM7","LSTM8","LSTM9","LSTM10", \
            "TCN","TCN7","CNN","CNN7"]:

            self.FILE_PREFIX = self.SYMBOL + self.LEARNING_TYPE_STR + self.ANSWER_STR + self.METHOD_STR + "_" + self.LSTM_TYPE + str(self.LSTM_LAYER_NUM) + self.TCN_ARG_STR+ self.RETURN_SEQ_STR + self.WINDOW_SIZE_STR + self.MIXTURE_NORMAL_STR + \
                          "_B" + str(self.BET_TERM)+ "_BS" + str(self.BET_SHIFT) + "_T" + str(self.TERM) + self.START_TERM_STR + self.END_TERM_STR + \
                          self.DB_SYMBOLS_STR + self.DB1_NOT_LEARN_STR + "_I" + self.DB_TERM_STR + \
                          "_IL" + self.INPUT_LEN_STR + "_LU" + self.LSTM_UNIT_STR + "_DU" + self.DENSE_UNIT_STR + self.FOOT_STR + \
                          self.SELF_AT_LAYER_NUM_STR + self.KSA_UNIT_NUM_STR + self.SELF_AT_INPUT_PLUS_STR + self.SELF_AT_NORMAL + \
                          self.MHA_UNIT_NUM_STR + self.MHA_HEAD_NUM_STR + \
                          self.CNN_UNIT_NUM_STR + self.CNN_INPUT_PLUS_STR + self.CNN_NORMAL_STR + self.LSTM_DO_STR + self.DROP_STR + self.L_DO_STR + self.L_RDO_STR +\
                          self.L_K_STR + self.L_R_STR + self.L_D_STR + self.ATR_STR + self.NORMAL_STR + self.DIVIDE_MAX_STR + self.DIVIDE_MIN_STR + \
                          self.BORDER_STR + self.SUFFIX + self.LEARNING_RATE_STR + self.LOSS_TYPE_STR + self.LOSS_STR + "_" + self.OPT + self.DENSE_ACTIVATION_STR + \
                          self.RNN_ACTIVATION_STR + self.RNN_REC_ACTIVATION_STR + self.LSTM9_INPUTS_STR + self.INPUT_DATA_STR + self.OUTPUT_TYPE_STR + self.OUTPUT_DATA_STR + \
                          "_BS" + str(self.BATCH_SIZE) + "_SD" + str(self.SEED) + self.DATA_SHUFFLE_STR + self.OPTIONS_STR+ self.TARGET_SPREAD_STR + self.ZERO_SEC_STR + self.SUB_STR + self.HOR_STR + self.HIGHLOW_STR + self.NON_LSTM_STR + self.OANDA_ORD_STR + self.OANDA_POS_STR + \
                          self.IND_FOOT_STR + self.IND_STR+ self.EXCEPT_LIST_STR + self.WORKERS_STR + self.MAX_QUEUE_SIZE_STR + self.SEC_OH_LEN_FIX_STR + "_" + socket.gethostname()

        elif self.METHOD == "NORMAL":
            self.FILE_PREFIX = self.SYMBOL + self.LEARNING_TYPE_STR + self.METHOD_STR + "_BET" + str(self.BET_TERM) + "_TERM" + str(self.TERM) + \
                          self.DB_SYMBOLS_STR + "_INPUT" + self.DB_TERM_STR + \
                          "_INPUT_LEN" + self.INPUT_LEN_STR + \
                          "_D-UNIT" + self.DENSE_UNIT_STR + self.FOOT_STR + self.DROP_STR + \
                          self.L_K_STR + self.L_R_STR + self.L_D_STR + self.ATR_STR + self.NORMAL_STR + self.DIVIDE_MAX_STR + self.DIVIDE_MIN_STR + \
                          self.BORDER_STR + self.SUFFIX + self.LEARNING_RATE_STR + "_LOSS-" + self.LOSS_TYPE + self.LOSS_STR + "_" + self.OPT + self.INPUT_DATA_STR + self.OUTPUT_TYPE_STR + self.OUTPUT_DATA_STR + "_BS" + \
                          str(self.BATCH_SIZE) + "_SEED" + str(self.SEED) + "_" + self.DATA_SHUFFLE + self.OPTIONS_STR + self.SUB_STR + self.EXCEPT_LIST_STR + "_" + socket.gethostname()

        #FILE_PREFIXを変更することのよって以下も変更する必要がある
        self.MODEL_DIR = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
        self.HISTORY_DIR = "/app/history/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
        self.CHK_DIR = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM

        self.MODEL_DIR_LOAD = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM
        self.CHK_DIR_LOAD = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM
        self.HISTORY_PATH = os.path.join(self.HISTORY_DIR, self.FILE_PREFIX)

        self.LOAD_CHK_PATH = os.path.join(self.CHK_DIR_LOAD, self.LOAD_CHK_NUM)

    def change_file_prefix(self, file_prefix):
        #"USDJPY_REGRESSION_LSTM_TYPE-LSTM1_BET5_TERM3600_INPUT300_INPUT_LEN180_L-UNIT30_D-UNIT_DROP0.0_201201_202012_L-RATE0.0004_LOSS-MSE_ADAM_md1-1_sub_IDL1_BS2048_SEED0_SHUFFLE_EL-20-21-22-23"
        self.FILE_PREFIX = file_prefix
        self.MODEL_DIR = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
        self.HISTORY_DIR = "/app/history/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
        self.CHK_DIR = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM

        self.MODEL_DIR_LOAD = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM
        self.CHK_DIR_LOAD = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM
        self.HISTORY_PATH = os.path.join(self.HISTORY_DIR, self.FILE_PREFIX)

        self.LOAD_CHK_PATH = os.path.join(self.CHK_DIR_LOAD, self.LOAD_CHK_NUM)

    def numbering(self):
        # win2のDBを参照してモデルのナンバリングを行う
        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        result = r.zrevrange("MODEL_NO", 0, -1, withscores=True)  # 全件取得
        if len(result) == 0:
            print("CANNOT GET MODEL_NO")
            exit(1)
        else:
            if self.LOAD_TYPE == 0:
                newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

                for line in result:
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)
                    tmp_name = tmps.get("name")
                    tmp_name = tmp_name.split("_MN")[0]
                    if tmp_name == self.FILE_PREFIX:
                        # 同じモデルがないが確認
                        print("The Model Already Exists!!!")
                        exit(1)

                self.FILE_PREFIX = self.FILE_PREFIX + "_MN" + str(newest_no)  # モデルに番号をつける

                # DBにモデルを登録
                child = {
                    'name': self.FILE_PREFIX,
                    'no': newest_no
                }
                r.zadd("MODEL_NO", json.dumps(child), newest_no)

                self.FILE_PREFIX = "MN" + str(newest_no)
            else:
                self.FILE_PREFIX = self.FILE_PREFIX_OLD
                
            self.MODEL_DIR = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
            self.HISTORY_DIR = "/app/history/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM
            self.CHK_DIR = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LEARNING_NUM

            self.MODEL_DIR_LOAD = "/app/model/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM
            self.CHK_DIR_LOAD = "/app/chk/bin_op/" + self.FILE_PREFIX + "-" + self.LOADING_NUM

            self.HISTORY_PATH = os.path.join(self.HISTORY_DIR, self.FILE_PREFIX)
            self.LOAD_CHK_PATH = os.path.join(self.CHK_DIR_LOAD, self.LOAD_CHK_NUM)

    def print_load_info(self):
        if self.LOAD_TYPE == 0:
            print("新規作成")
        elif self.LOAD_TYPE == 1:
            print("modelからロード")
            print("LOADING_NUM:", self.LOADING_NUM)
        elif self.LOAD_TYPE == 2:
            print("chekpointからロード")
            print("LOAD_CHK_NUM:", self.LOAD_CHK_NUM)

        print("Model is " , self.FILE_PREFIX)

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
