import os
import logging.config
from decimal import Decimal
from util import *
import datetime
from datetime import timedelta
from app_usdjpy_fx_predict4_lgbm_920_conf import *

from important_index import ImportantIndex

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_thinkm.conf"))
loggerConf = logging.getLogger("thinkm")

class ConfThinkM():
    def __init__(self):
        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win8"

        self.START_TIME = datetime.datetime(year=2025, month=12, day=17, hour=0, minute=2, second=0, microsecond=0)

        #経済指標発表時は除外
        self.IMPORTANCE = "importances_high"
        self.IMPORTANT_INDEX_RANGE = 60
        #当日と翌日分の経済指標を作成
        self.IMPORTANT_INDEX = ImportantIndex(importance=self.IMPORTANCE, range=self.IMPORTANT_INDEX_RANGE,
                                              startDt=datetime.datetime.now(),endDt=datetime.datetime.now() + datetime.timedelta(days = 1))
        self.LOGGER(self.IMPORTANT_INDEX.get_index())

        #self.ID = "HLMI499029"
        #self.PW = "Ep2VuZtU"

        self.CAPTCHA_API_KEY = "0b32aefd3a864bd261f8cd69affd136b"
        self.CAPTCHA_SITE_KEY = "db5b3047-b5f2-424f-8c07-4129a225d72d"
        self.CAPTCHA_URL = "https://web.thinktrader.com/account/login"

        self.ID = "reicou@i.softbank.jp"
        self.PW = "Reikou0129@"

        #self.ID = "yorikoiiduka.hl@gmail.com"
        #self.PW = "Yoriko2918&"

        self.BORDER_ATR = None

        self.EXCEPT_SEC = []
        self.EXCEPT_MIN = []

        # ワンクリック注文か詳細注文するか
        # 注文画面でのスプレッド 注文画面を使用する場合、ワンクリックトレードを無効にしておくこと！
        self.ORDER_TYPE = "ONECLICK" #ONECLICK or DETAIL

        self.DEAL_TYPE = "ONE" #ALL or ONE:決済を一括でする(どれか一つでも決済する場合は全て決済する)か、個別にするか

        #self.DIV_REG_LIST = [10, 60, 300]
        self.DIV_REG_LIST = []
        self.MAX_DIV = 10
        self.MAX_DIV_SEC = 10

        #取引間隔
        self.TRADE_TERM = 2 #注文間隔の制限
        #self.TRADE_TERM = 14
        self.SINGLE_FLG = False #True:ポジションは一つしか持たない
        self.EXT_FLG = True #True:延長判断する場合
        self.TRADE_EXT_TERM = 2 #LOOP_TERMの倍数
        self.TRADE_EXT_SHORT_TERM = 2 #LOOP_TERMの倍数
        self.TRADE_EXT_SHORT_NUM = 2 #TRADE_EXT_SHORT_TERMを適用する最大ポジション数

        self.TRADE_EXT_START = 12 #LOOP_TERMの倍数 約定するまでの許容秒数(ORDER_TAKE_SEC)より大きい数字にする必要がある
        self.TRADE_EXT_START_SHORT_TERM = 12 #LOOP_TERMの倍数
        self.TRADE_EXT_START_SHORT_NUM = 2 #TRADE_EXT_START_SHORT_TERMを適用する最大ポジション数

        self.lgbm_model_file = lgbm_model_file
        self.lgbm_model_file_suffix = lgbm_model_file_suffix
        self.lgbm_model_file_ext = lgbm_model_file_ext
        self.lgbm_model_file_suffix_ext = lgbm_model_file_suffix_ext

        self.BET_TYPE = "CATEGORY"
        self.BET_BORDER = 0.62
        self.BET_BORDER_EXT = 0.05

        self.TARGET_SPREAD = 5

        #self.MAX_TRADE_CNT = 540
        self.MAX_TRADE_CNT = 9999

        #取引額
        #self.AMT = "99000"
        #self.AMT_STR = "99,000"

        self.AMT = "1000"
        self.AMT_STR = "1,000"

        #self.REFER_TICK_SEC = 2 #0なら参照しない 過去何秒分のティックデータを参照するか。
        self.REFER_TICK_SEC = 0 #0なら参照しない 過去何秒分のティックデータを参照するか。
        self.REFER_TICK_MOVE_CNT = 1 # 最初のティックデータよりレートがベットする方に動いた回数

        #許容する最大ポジション数
        self.MAX_POSITION_NUM = 1

        self.MAX_TIME_OUT = 10
        self.MAX_EXCEPT_CNT = 30

        self.AI_MODEL_TERM = AI_MODEL_TERM  # AIモデルの最小データ間隔
        self.LOOP_TERM = 1
        #self.SWITCH_TERM = 330
        self.SWITCH_TERM = 10000

        self.ORDER_TAKE_SEC = self.LOOP_TERM * 3 #発注から約定するまでかかるであろう秒数 LOOP_TERMの倍数

        self.LIMIT_SATR_MUILT = 0 #逆指値をsatrの何倍にするか
        self.STOP_LOSS_FIX = 0.05
        self.STOP_LOSS_MANUAL = 0.05 #手動ストップロス用

        #self.ORDER_TOTAL_STOPLOSS = None
        self.ORDER_TOTAL_STOPLOSS = -0.7

        self.STOPLOSS_UPDATE_PIPS = None

        self.TRAIL_STOPLOSS = False
        self.TRAIL_STOPLOSS_PIPS = 0.2

        self.DEAL_TAKE_SEC = 4 #決済してから処理されるであろう秒数

        self.NO_DEAL_FLG = False
        self.DEMO_FLG = False


        #self.REQUEST_URL = "http://127.0.0.1:7001/"
        self.HOST = "127.0.0.1"
        self.DB_NO = 8
        self.FX_DB_NO = 0
        self.DB_KEY = ""
        self.DB_HISTORY_KEY = ""
        self.DB_ORDER_KEY = ""
        self.DB_FX_DATA_KEY = ""
        self.DB_FX_DATA_KEY_USDJPY = ""

        self.DB_KEY_HEART_BEAT = 'process_id'
        self.DB_NO_HEART_BEAT = 0

        self.PAIR = ""
        self.PRED_TERM = ""
        self.PRED_TERM_ADJUST = 0 #決済時間を調整
        self.ARG = ""
        self.RATE_FORMAT = ""

        self.PREDICT_REQUEST_HOST = PREDICT_REQUEST_HOST
        self.PREDICT_REQUEST_DB_NO = PREDICT_REQUEST_DB_NO
        self.PREDICT_REQUEST_KEY = PREDICT_REQUEST_KEY

        self.NG_SHIFT = 48

        self.REGIST_HISTORY_ONLY = False #True:処理が取引履歴DB登録のみの場合

        self.DEAL_ALL_ONLY = False #全決済だけさせる

        self.MAIN_LOOP_CNT = 1
        self.MAIN_LOOP_CNT_MAX = 15

        self.BET_CNT = 0
        self.DEAL_CNT = 0
        self.EXCEPT_CNT = 0
        self.TIMEOVER_CNT = 0
        self.RATE_ERR_CNT = 0
        self.RATE_ERR_MAX_CNT = 10
        self.SPREAD_OVER_CNT = 0
        self.BET_ERR_CNT = 0
        self.CLOSE_LOCAL_ERR_CNT = 0
        self.RELOAD_CNT = 0

        self.FX_DATA_MACHINE = "192.168.1.114" #win2
        self.FX_TICK_DB_NAME = "VANTAGE_USDJPY.p_TICK"

        self.PREV_TRADE_TIME = None

        self.MAX_LEN = MAX_CLOSE_LEN - 1
        self.MAX_LEN_SEC = self.MAX_LEN * self.AI_MODEL_TERM

        #self.FOOT_DICT = {30:71}#mt5から取得する分足の種類(min表記)をkey、長さをvalue
        self.FOOT_DICT = {}
        self.FOOT_DB_NAME_PREFIT = "Tradeview_USDJPY_M"

        self.LOOP_END_DATETIME_PRE = None
        self.LOOP_END_DATETIME = None

        self.END_DATETIME_PRE = None
        self.END_DATETIME = None

        self.predict_slow_send = False

        self.DELETE_CHART = '/html/body/div[3]'

        self.PAIR_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div[2]/div[1]/div/div/div[2]/div[1]/div'
        self.PAIR_PATH_ONECLICK = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[1]/div[2]/div[2]/div'

        self.PROFIT_PATH = '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[1]/div[3]/div[2]/div'

        self.PW1_PATH = "//*[@id=\"twoFactorAuthenticationCode_1\"]"
        self.PW2_PATH = "//*[@id=\"twoFactorAuthenticationCode_2\"]"
        self.PW3_PATH = "//*[@id=\"twoFactorAuthenticationCode_3\"]"
        self.PW4_PATH = "//*[@id=\"twoFactorAuthenticationCode_4\"]"
        self.PW5_PATH = "//*[@id=\"twoFactorAuthenticationCode_5\"]"
        self.PW6_PATH = "//*[@id=\"twoFactorAuthenticationCode_6\"]"
        self.PW_BUTTON = "//*[@id=\"root\"]/div/div/div[2]/div/form/div[3]/button"

        self.DEMO_SELECT_PATH = "//*[@id=\"root\"]/div/div[1]/div[2]/div/form/div[2]/div/div/div[1]"
        self.LIVE_SELECT_PATH = '//*[@id="root"]/div/div[1]/div[2]/div/div[2]/div[2]'
        self.ID_INPUT_PATH = "//*[@id=\"email\"]"
        self.PW_INPUT_PATH = "//*[@id=\"password\"]"
        self.LOGIN_PATH = '//*[@id="root"]/div/div[1]/div/div[2]/div/form/button[1]'

        self.MONEY_PATH = '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[3]/div/div'

        self.MY_BUTTON = "//*[@id=\"root\"]/div/div/div[2]/div[1]/div/div[2]/div[2]/div/div[2]"
        self.LOG_OUT_DEMO = "/html/body/div[5]/div/div[2]/div/div[4]/span[2]"
        self.LOG_OUT_LIVE = '/html/body/div[4]/div/div[2]/div/div[5]/div/span'
        self.LOG_OUT_LIVE2 = '/html/body/div[5]/div/div[2]/div/div[5]/div/span'
        self.LOG_OUT_LIVE3 = '/html/body/div[3]/div/div[2]/div/div[5]/div/span'

        self.LOG_OUT_OK = '//*[@id="logout"]/div[2]/footer/button[2]'

        self.ACCOUNT_PATH = '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[2]/div/div[2]'
        self.SWITCH_LIVE_PATH = "/html/body/div[4]/div/div[1]/div[2]/button"
        self.SWITCH_DEMO_PATH = "/html/body/div[4]/div/div[2]/div[2]/div[3]/span[2]"
        self.ACCOUNT_TYPE_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[1]/div/div[2]/div[2]/div/div[1]/div[1]"
        #USDJPYペア選択
        self.USDJPY_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div/div[1]/div[1]/div[1]"
        #EURUSDペア選択
        self.EURUSD_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div/div[1]/div[2]/div[1]"

        self.USDJPY_DELETE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[1]/div[1]/div/div/div[2]/div/div[1]/div[1]/div[1]/div[1]'
        self.EURUSD_DELETE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[1]/div[1]/div/div/div[2]/div/div[1]/div[2]/div[1]/div[1]'
        self.PAIR_DELETE_PATH = "/html/body/div[5]/div/div/div[3]/div/button[2]/span"

        self.TOP_CHART_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[1]/div/cq-context/div[1]/div[2]/div[2]'

        #ワンクリックにするかどうかのトグルボタン
        self.SELECT_ONECLICK_TOGGLE = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div[2]/div[1]/div[1]/div/label'
        # 注文画面表示
        self.ORDER_SELECT_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[3]/div/div[1]'
        # 注文画面閉じる
        self.ORDER_SELECT_CLOSE_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[4]/button[1]"
        # 注文画面での買い
        self.BUY_SELECT_BUTTON_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[3]"
        # 注文画面でのスプレッド 注文画面を使用する場合、ワンクリックトレードを無効にしておくこと！
        self.ORDER_SELECT_SPREAD_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[2]"
        # 注文画面での売り
        self.SELL_SELECT_BUTTON_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[1]"
        # 注文画面での注文ボタン
        self.TRADE_BUTTON_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[4]/button[2]"

        # 注文画面での注文ボタン確定
        self.TRADE_BUTTON2_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/button[2]"

        # 注文画面でのポジション数入力
        self.ORDER_POSITION_INPUT_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[2]/div[2]/div/input"
        # 注文画面での指値入力ボタン
        self.ORDER_PROFIT_BUTTON_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[6]/div/div[2]/div"
        # 注文画面での指値入力
        self.ORDER_PROFIT_INPUT_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[6]/div[2]/div[2]/div[1]/input"

        # 注文画面での逆指値入力ボタン
        self.ORDER_LIMIT_BUTTON_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[7]/div/div[2]/div"

        # 注文画面での逆指値入力
        self.ORDER_LIMIT_INPUT_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[7]/div[2]/div/div[1]/div/input'

        #注文画面でのキャンセル
        self.ORDER_CANCEL_BUTTON = '/html/body/div[NUM]/div/div/div/div/div[3]/button[1]'

        # ワンクリック有効化ボタン
        self.ONECLICK_BUTTON = "//*[@id=\"root\"]/div/div/div[2]/div[2]/div/div/div[2]/div[1]/div[1]/label/div"
        # ワンクリック有効化ボタンを押した後の確認画面での決定ボタン
        self.ONECLICK_BUTTON_KETTEI = "/html/body/div[6]/div/div/div[2]/div/button[2]"

        # ワンクリックでの売りレート
        self.ONECLICK_SELL_RATE_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div/div[1]/div[1]/div[2]/div[1]/div[2]/div"
        # ワンクリックでの買いレート
        self.ONECLICK_BUY_RATE_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div/div[1]/div[1]/div[2]/div[3]/div[2]/div"
        # ワンクリックでのスプレッド
        self.SPREAD_ONECLICK_PATH = "//*[@id=\"watchlistPanel\"]/div/div/div[2]/div/div[1]/div[1]/div[2]/div[2]"

        # ページ左部の買いレート
        self.BUY_RATE_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[3]/div/div/div'
        # ページ左部の売りレート
        self.SELL_RATE_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[1]/div/div/div'

         # ページ左部のスプレッド
        self.SPREAD_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/div/div[2]'

        # ページ上部右の買いボタン
        self.SCREEN_TOP_BUY_BUTTON = '/html/body/div[3]/div[1]/div/div/div[3]/div/div/div/div/div/div[16]/div/div/div[3]/div[1]'

        # ワンクリックでの買いレート
        self.BUY_RATE_PATH_ONECLICK = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[2]/div[3]/div/div[2]/div'
        # ワンクリックでのスプレッド
        self.SPREAD_PATH_ONECLICK = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[2]/div[2]'

        # ワンクリックでの買い
        self.BUY_ONECLICK_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[2]/div[3]'
        # ワンクリックでの売り
        self.SELL_ONECLICK_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[2]/div[1]'
        # ワンクリックでのポジション数入力
        self.ONECLICK_POSITION_INPUT_PATH = '//*[@id="watchlistPanel"]/div/div/div[2]/div/div[1]/div/div[3]/input'

        #メッセージ
        self.MSG_PATH = "//*[@id=\"root\"]/div/div/div[1]/div[4]/div/div/div/div/div/div/div[2]/div/p"

        #建玉数の表示
        #self.POSITION_NUM_PATH = '/html/body/div[1]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[1]/span[2]/span'
        self.POSITION_NUM_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[1]/span[2]/span'

        ##ポジション欄
        #建玉ID
        self.POSITION_ID_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/thead/tr/th[9]/div[1]/div'

        self.POSITION_TABLE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody'

        #2ポジション以上ある場合の建玉欄の展開ボタン
        self.POSITION_EXPAND_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[1]/div/svg'

        #ポジションが1つしかない場合の決済ボタン
        self.POSITION_1_DEAL_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/thead/tr/th[11]/div[1]/div'

        # ポジションが1つしかない場合の新規日付
        self.POSITION_1_START_DATE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[4]/span'
        # ポジションが1つしかない場合の新規時刻
        self.POSITION_1_START_TIME_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[8]/span'

        # ポジションが1つしかない場合の買い売り
        self.POSITION_1_TYPE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[2]/span'
        # ポジションが1つしかない場合の逆指値ボタン
        self.POSITION_1_STOPLOSS_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[6]/div/button'
        # ポジションが1つしかない場合の逆指値表示セル
        self.POSITION_1_STOPLOSS = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[6]/div/button/div/div'
        # ポジションが1つしかない場合の新規レート
        self.POSITION_1_NEW_RATE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr/td[5]/span'


        #ポジションが2つ以上ある場合のポジションの決済ボタン
        self.POSITION_2_DEAL_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[11]/div/div'

        # ポジションが2つ以上ある場合の新規時刻
        self.POSITION_2_START_TIME_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[8]/span'
        # ポジションが2つ以上ある場合の買い売り
        self.POSITION_2_TYPE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[2]/span'
        # ポジションが2つ以上ある場合の逆指値ボタン
        self.POSITION_2_STOPLOSS_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[6]/div/button'
        # ポジションが2つ以上ある場合の逆指値表示セル
        self.POSITION_2_STOPLOSS = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[6]/div/button/div/div'
        # ポジションが2つ以上ある場合の新規レート
        self.POSITION_2_NEW_RATE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]/td[5]/span'

        #逆指値入力
        self.POSITION_STOPLOSS_INPUT = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[2]/div[1]/div/div/div[3]/div/div[2]/div[2]/div[2]/div[1]/input'

        self.POSITION_STOPLOSS_ENTER_BUTTON = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[2]/div[1]/div/div/div[4]/button[2]'

        #ポジション全決済ボタン
        self.POSITION_ALL_DEAL_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/thead/tr/th[11]/div[1]/div'
        self.POSITION_ALL_DEAL_BUTTON_PATH_NEW = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/thead/tr/th[11]/div[1]/div'

        #ポジション決済の決定ボタン
        self.POSITION_DEAL_BUTTON_PATH = '/html/body/div[10]/div/div/div[2]/div/button[2]'

        self.MODAL_PATH = '/html/body/div[6]/div/div/div[1]'

        self.MODAL_CANCEL = '/html/body/div[6]/div/div/div[3]/button/span'

        #予約注文
        self.RESEARVE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/span[2]'

        # 履歴
        self.HISTORY_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[3]/span[2]'

        # 履歴の期間選択リスト
        self.HISTORY_LIST_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[2]/div[1]'
        # 履歴の1週間選択
        self.HISTORY_LIST_WEEK = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[2]/div[1]/div[2]/div[2]'
        # 履歴の1カ月選択
        self.HISTORY_LIST_MONTH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[2]/div[1]/div[2]/div[3]'
        # 履歴の件数
        self.HISTORY_NUM = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[3]/span[2]/span'
        # 履歴の一行
        self.HISTORY_TR_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody/tr[NUM]'

        # 履歴の親テーブル
        self.HISTORY_TABLE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/tbody'

        self.OPEN_DATE_COL = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[2]/table/tbody/tr/td[6]/div[1]/div'
        self.OPEN_DAY_SORT = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[2]/table/tbody/tr/td[6]/div[2]/span'

        self.POSITION_ID = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[2]/div[1]/div/div[2]/div/div[1]/div/table/thead/tr/th[13]/div[1]/div'

        # 履歴　ペア
        self.HISTORY_TR_PAIR = 'td[1]/div/span'
        # 履歴　売買
        self.HISTORY_BET_TYPE = 'td[2]/span'
        # 履歴　新規レート
        self.HISTORY_OPEN_RATE = 'td[4]/span'
        # 履歴　決済レート
        self.HISTORY_CLOSE_RATE = 'td[5]/span'
        # 履歴　開始日
        self.HISTORY_OPEN_DAY = 'td[6]/span'
        # 履歴　決済日
        self.HISTORY_CLOSE_DAY = 'td[7]/span'
        # 履歴　開始時間
        self.HISTORY_OPEN_TIME = 'td[11]/span'
        # 履歴　決済時間
        self.HISTORY_CLOSE_TIME = 'td[12]/span'
        # 履歴　逆指値
        self.HISTORY_STOPLOSS = 'td[8]/span'
        # 履歴　ポジション数
        self.HISITORY_POSITION_NUM = 'td[3]/span'


        # 別モーダル画面からの注文
        self.MODAL_CHOISE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[3]/div[1]/div[2]/div[1]/div/div/div[1]/div[1]/div[2]/div/div'

        self.MODAL_ORDER_POSITION_INPUT_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[2]/div[2]/div/input'
        self.MODAL_ORDER_LIMIT_BUTTON_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[8]/div/div[2]/div/label'
        self.MODAL_ORDER_LIMIT_INPUT_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[8]/div[2]/div/div[1]/div/input'
        self.MODAL_TRADE_BUTTON_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'
        self.MODAL_TRADE_BUTTON2_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'
        self.MODAL_ORDER_SELECT_CLOSE_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[1]'

        self.MODAL_NUMS = list(range(4,100))
        self.MODAL_NUM = 4

        #決済確定
        self.MODAL_POSITION_DEAL_BUTTON_PATH = '/html/body/div[NUM]/div/div/div[3]/button[2]'

        #全決済確認ボタン
        self.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH = '/html/body/div[NUM]/div/div/div[3]/div/button[2]'
        self.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH_NEW = '/html/body/div[NUM]/div/div/div[2]/div/button[2]'

        #逆指値入力
        self.MODAL_POSITION_STOPLOSS_INPUT = '/html/body/div[NUM]/div/div/div/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/input'
        #逆指値指定決定ボタン
        self.MODAL_POSITION_STOPLOSS_ENTER_BUTTON = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'


        #損益
        self.NOW_PROFIT = '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div'

        self.BET_TYPE_BUY = "買い"
        self.BET_TYPE_SELL = "売り"

    def initial(self):
        if self.FX_DATA_MACHINE == "192.168.1.114": #win2
            self.DB_FX_DATA_KEY = "Tradeview_" + self.PAIR + "_S1"
            self.DB_FX_DATA_KEY_USDJPY = "Tradeview_USDJPY_S1"
        elif self.FX_DATA_MACHINE == "192.168.1.15": #win5
            self.DB_FX_DATA_KEY = "XM_" + self.PAIR + "#_S1"
            self.DB_FX_DATA_KEY_USDJPY = "XM_USDJPY_S1"

        self.SECOND_FLG = True

        #360分で自動ログアウトしてしまうので、それにあわせて終了時間を設定する
        # 1回目は0時開始(日本時間9時)、5:52終了
        # 2回目は6時開始(日本時間15時)、11:52終了
        # 3回目は12時開始(日本時間21時)、13:52終了
        # 4回目は14時開始(日本時間23時)、19:52終了
        tmp_dt = datetime.datetime.now()
        if tmp_dt.hour == 23:
            tmp_dt = tmp_dt + timedelta(days=1)

        #self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,hour=15, minute=55, second=0, microsecond=0)
        self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day, hour=14, minute=58,second=0, microsecond=0)

        #予想時間分早く終わらせて、終了時に結果がでるまで待たせる
        self.END_DATETIME_PRE = self.END_DATETIME - timedelta(seconds=self.PRED_TERM)


        if self.DEMO_FLG:
            self.DB_KEY = self.DB_KEY + "_DEMO"

        if self.SERVER_NAME == "win8":

            self.top_buy_button_x = 1900
            self.top_buy_button_y = 168

            self.modal_change_button_x = 1845
            self.modal_change_button_y = 160

            self.set_amt_x1 = 380
            self.set_amt_y1 = 333
            self.set_amt_x2 = 330
            self.set_amt_y2 = 510

            self.click_order_detail_x = 1905
            self.click_order_detail_y = 145

            self.click_account_x = 1894
            self.click_account_y = 110

            self.move_random_x1 = 500
            self.move_random_y1 = 400

            self.move_random_x2 = 500
            self.move_random_y2 = 400

            self.obs_x1 = 860
            self.obs_y1 = 1060
            self.obs_x2 = 1650
            self.obs_y2 = 750

            self.regist_history_db_x = 1800
            self.regist_history_db_y = 650
            self.regist_history_db_scroll = -600

            self.reload_x = 93
            self.reload_y = 60

            self.expand_x = 593
            self.expand_y = 593

            self.logout_x = 1680
            self.logout_y = 535

