import os
import logging.config
from decimal import Decimal
from util import *
import datetime
from datetime import timedelta
from app_usdjpy_fx_predict40_lgbm_conf import *

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_thinkm.conf"))
loggerConf = logging.getLogger("thinkm")

class ConfThinkM():
    def __init__(self):
        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win8"

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

        self.START_TIME = datetime.datetime(year=2024, month=8, day=12, hour=1, minute=2, second=0, microsecond=0)

        #取引対象外の時間外を開始、終了それぞれのdatetimeでリストにする
        self.EXCEPT_DATETIME = [
            [
                datetime.datetime(year=2024, month=8, day=12, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=12, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=12, hour=6, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=12, hour=7, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=12, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=12, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=12, hour=17, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=12, hour=18, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=12, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=12, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=12, hour=23, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=0, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=0, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=0, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=1, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=1, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=6, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=7, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=8, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=9, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=9, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=9, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=13, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=13, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=1, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=2, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=6, minute=43, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=6, minute=48, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=8, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=9, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=10, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=11, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=14, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=14, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=1, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=1, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=1, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=2, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=4, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=4, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=6, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=6, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=7, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=8, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=13, minute=13, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=13, minute=18, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=14, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=15, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=15, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=4, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=4, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=6, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=6, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=8, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=8, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=8, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=9, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=12, minute=13, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=12, minute=18, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=8, day=16, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=8, day=16, hour=14, minute=3, second=0, microsecond=0),
            ],

        ]
        # ワンクリック注文か詳細注文するか
        # 注文画面でのスプレッド 注文画面を使用する場合、ワンクリックトレードを無効にしておくこと！
        self.ORDER_TYPE = "ONECLICK" #ONECLICK or DETAIL

        self.DEAL_TYPE = "ONE" #ALL or ONE:決済を一括でする(どれか一つでも決済する場合は全て決済する)か、個別にするか

        #self.DIV_REG_LIST = [10, 60, 300]
        self.DIV_REG_LIST = []
        self.MAX_DIV = 10
        self.MAX_DIV_SEC = 10

        #取引間隔
        self.TRADE_TERM = 4
        #self.TRADE_TERM = 14
        self.SINGLE_FLG = False #True:ポジションは一つしか持たない
        self.EXT_FLG = True #True:延長判断する場合
        self.TRADE_EXT_TERM = 12 #LOOP_TERMの倍数
        self.TRADE_EXT_SHORT_TERM = 4 #LOOP_TERMの倍数
        self.TRADE_EXT_SHORT_NUM = 2 #TRADE_EXT_SHORT_TERMを適用する最大ポジション数

        self.TRADE_EXT_START = 12 #LOOP_TERMの倍数
        self.TRADE_EXT_START_SHORT_TERM = 12 #LOOP_TERMの倍数
        self.TRADE_EXT_START_SHORT_NUM = 6 #TRADE_EXT_START_SHORT_TERMを適用する最大ポジション数

        self.lgbm_model_file = lgbm_model_file
        self.lgbm_model_file_suffix = lgbm_model_file_suffix
        self.lgbm_model_file_ext = lgbm_model_file_ext
        self.lgbm_model_file_suffix_ext = lgbm_model_file_suffix_ext

        self.BET_TYPE = "CATEGORY"
        self.BET_BORDER = 0.6
        self.BET_BORDER_EXT = 0.05

        #self.MAX_TRADE_CNT = 540
        self.MAX_TRADE_CNT = 9999

        #取引額
        #self.AMT = "90000"
        #self.AMT_STR = "90,000"

        self.AMT = "1000"
        self.AMT_STR = "1,000"

        #許容する最大ポジション数
        self.MAX_POSITION_NUM = 8

        self.MAX_TIME_OUT = 10
        self.MAX_EXCEPT_CNT = 30

        self.AI_MODEL_TERM = AI_MODEL_TERM  # AIモデルの最小データ間隔
        self.LOOP_TERM = 2
        #self.SWITCH_TERM = 330
        self.SWITCH_TERM = 10000

        self.ORDER_TAKE_SEC = 8 #発注から約定するまでかかるであろう秒数

        self.LIMIT_SATR_MUILT = 0 #逆指値をsatrの何倍にするか
        self.STOP_LOSS_FIX = 0.55
        self.STOP_LOSS_MANUAL = 0.5 #手動ストップロス用

        #self.ORDER_TOTAL_STOPLOSS = None
        self.ORDER_TOTAL_STOPLOSS = -0.7

        self.STOPLOSS_UPDATE_PIPS = None

        self.TRAIL_STOPLOSS = False
        self.TRAIL_STOPLOSS_PIPS = 0.2

        self.DEAL_TAKE_SEC = 8 #決済してから処理されるであろう秒数

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

        self.PAIR = ""
        self.PRED_TERM = ""
        self.PRED_TERM_ADJUST = 0 #決済時間を調整
        self.ARG = ""
        self.RATE_FORMAT = ""

        self.PREDICT_REQUEST_HOST = PREDICT_REQUEST_HOST
        self.PREDICT_REQUEST_DB_NO = PREDICT_REQUEST_DB_NO
        self.PREDICT_REQUEST_KEY = PREDICT_REQUEST_KEY

        self.NG_SHIFT = self.TRADE_EXT_TERM

        self.REGIST_HISTORY_ONLY = False #True:処理が取引履歴DB登録のみの場合

        self.MAIN_LOOP_CNT = 1
        self.MAIN_LOOP_CNT_MAX = 5

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

        self.FX_DATA_MACHINE = "192.168.1.114"

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
        self.LOGIN_PATH = "//*[@id=\"root\"]/div/div[1]/div[2]/div/form/div[6]/button"

        self.MONEY_PATH = '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[1]/div[4]/div/div[2]'

        self.MY_BUTTON = "//*[@id=\"root\"]/div/div/div[2]/div[1]/div/div[2]/div[2]/div/div[2]"
        self.LOG_OUT_DEMO = "/html/body/div[5]/div/div[2]/div/div[4]/span[2]"
        self.LOG_OUT_LIVE = '/html/body/div[4]/div/div[2]/div[3]/div[5]/div/span'
        self.LOG_OUT_LIVE2 = '/html/body/div[5]/div/div[2]/div[3]/div[5]/div/span'
        self.LOG_OUT_LIVE3 = '/html/body/div[3]/div/div[2]/div[3]/div[5]/div/span'

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

        self.TOP_CHART_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[1]/div/cq-context/div[1]/div[2]/div[2]'

        #ワンクリックにするかどうかのトグルボタン
        self.SELECT_ONECLICK_TOGGLE = '//*[@id="root"]/div/div/div[2]/div[2]/div/div/div[2]/div[1]/div[1]/div/label'
        # 注文画面表示
        self.ORDER_SELECT_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[3]/div/div[1]'
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
        self.ORDER_LIMIT_INPUT_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div[2]/div[7]/div[2]/div/div[1]/div/input'

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
        self.POSITION_NUM_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[1]/span[2]'
        #新規日付
        self.POSITION_DATE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[2]/table/tbody/tr/td[4]'

        #建玉欄の1行目
        self.POSITION_1_TR = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div/div/div[1]/div/table/tbody/tr[1]"
        #建玉欄の1行目の1セル目
        self.POSITION_1_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div/div/div[1]/div/table/tbody/tr[1]/td[1]"
        #2ポジション以上ある場合の建玉欄の展開ボタン
        self.POSITION_EXPAND_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[2]/table/tbody/tr[1]/td[1]/div[1]/div'

        #ポジションが1つしかない場合の決済ボタン
        self.POSITION_1_DEAL_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[2]/table/tbody/tr[1]/td[4]/div'
        # ポジションが1つしかない場合の新規時刻
        self.POSITION_1_START_TIME_PATH = "//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div/div/div[1]/div/table/tbody/tr[1]/td[5]"
        # ポジションが1つしかない場合の買い売り
        self.POSITION_1_TYPE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div/div/div[1]/div/table/tbody/tr[1]/td[2]'
        # ポジションが1つしかない場合の逆指値ボタン
        self.POSITION_1_STOPLOSS_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[1]/td[7]/div/button'
        # ポジションが1つしかない場合の逆指値表示セル
        self.POSITION_1_STOPLOSS = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[1]/td[7]/div/button/div/div'
        # ポジションが1つしかない場合の新規レート
        self.POSITION_1_NEW_RATE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[1]/td[6]'


        #ポジションが2つ以上ある場合のポジションの決済ボタン
        self.POSITION_2_DEAL_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[2]/table/tbody/tr[NUM]/td[4]/div'
        # ポジションが2つ以上ある場合の新規時刻
        self.POSITION_2_START_TIME_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]/td[5]'
        # ポジションが2つ以上ある場合の買い売り
        self.POSITION_2_TYPE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]/td[2]'
        # ポジションが2つ以上ある場合の逆指値ボタン
        self.POSITION_2_STOPLOSS_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]/td[7]/div/button'
        # ポジションが2つ以上ある場合の逆指値表示セル
        self.POSITION_2_STOPLOSS = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]/td[7]/div/button/div/div'
        # ポジションが2つ以上ある場合の新規レート
        self.POSITION_2_NEW_RATE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]/td[6]'

        #逆指値入力
        self.POSITION_STOPLOSS_INPUT = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[3]/div/div[2]/div[2]/div[2]/div[1]/input'
        #逆指値指定決定ボタン
        self.POSITION_STOPLOSS_ENTER_BUTTON = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[4]/button[2]'

        #ポジション全決済ボタン
        self.POSITION_ALL_DEAL_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[1]/table/tbody/tr/td[4]/div/div'
        #ポジション決済の決定ボタン
        self.POSITION_DEAL_BUTTON_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div[4]/div/button[2]'
        self.MODAL_PATH = "/html/body/div[5]/div/div/div[1]"

        self.MODAL_CANCEL = "/html/body/div[5]/div/div/div[3]/buttonn"

        #予約注文
        self.RESEARVE_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[2]/span[2]'

        # 履歴
        self.HISTORY_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[3]/span[2]'

        # 履歴の期間選択リスト
        self.HISTORY_LIST_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[2]/div[1]'
        # 履歴の1週間選択
        self.HISTORY_LIST_WEEK = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[2]/div[1]/div[2]/div[2]'
        # 履歴の件数
        self.HISTORY_NUM = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[3]/span[2]'
        # 履歴の一行
        self.HISTORY_TR_PATH = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[1]/div/div[1]/div/table/tbody/tr[NUM]'

        self.HISTORY_TABLE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[2]/table/tbody'

        self.OPEN_DATE_COL = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[2]/table/tbody/tr/td[6]/div[1]/div'
        self.OPEN_DAY_SORT = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[5]/div[2]/table/tbody/tr/td[6]/div[2]/span'
        # 履歴　ペア
        self.HISTORY_TR_PAIR = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[2]/div[1]/div/div[2]/div/div/div/div[6]/div[2]/table/tbody/tr[NUM]/td[1]/div[2]'
        # 履歴　売買
        self.HISTORY_BET_TYPE = 'td[2]'
        # 履歴　新規レート
        self.HISTORY_OPEN_RATE = 'td[4]'
        # 履歴　決済レート
        self.HISTORY_CLOSE_RATE = 'td[5]'
        # 履歴　開始日
        self.HISTORY_OPEN_DAY = 'td[6]/div'
        # 履歴　決済日
        self.HISTORY_CLOSE_DAY = 'td[7]/div'
        # 履歴　開始時間
        self.HISTORY_OPEN_TIME = 'td[10]'
        # 履歴　決済時間
        self.HISTORY_CLOSE_TIME = 'td[11]'
        # 履歴　逆指値
        self.HISTORY_STOPLOSS = 'td[12]'
        # 履歴　ポジション数
        self.HISITORY_POSITION_NUM = 'td[3]'


        # 別モーダル画面からの注文
        self.MODAL_CHOISE = '//*[@id="root"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[2]/div[1]/div/div/div[1]/div[1]/div[2]/div/div'

        self.MODAL_ORDER_POSITION_INPUT_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[2]/div[2]/div/input'
        self.MODAL_ORDER_LIMIT_BUTTON_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[8]/div/div[2]/div/label'
        self.MODAL_ORDER_LIMIT_INPUT_PATH = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[8]/div[2]/div/div[1]/div/input'
        self.MODAL_TRADE_BUTTON_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'
        self.MODAL_TRADE_BUTTON2_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'
        self.MODAL_ORDER_SELECT_CLOSE_PATH = '/html/body/div[NUM]/div/div/div/div/div[3]/button[1]'

        self.MODAL_NUM_FIRST = 4
        self.MODAL_NUM_SECOND = 5

        #決済確定
        self.MODAL_POSITION_DEAL_BUTTON_PATH = '/html/body/div[NUM]/div/div/div[3]/div/button[2]'

        #全決済確認ボタン
        self.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH = '/html/body/div[NUM]/div/div/div[2]/div/button[2]'

        #逆指値入力
        self.MODAL_POSITION_STOPLOSS_INPUT = '/html/body/div[NUM]/div/div/div/div/div[2]/div/div[2]/div[2]/div/div[1]/div/input'
        #逆指値指定決定ボタン
        self.MODAL_POSITION_STOPLOSS_ENTER_BUTTON = '/html/body/div[NUM]/div/div/div/div/div[3]/button[2]'

        #損益
        self.NOW_PROFIT = '//*[@id="root"]/div/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[3]/div[2]/div'


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
        self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day, hour=19, minute=55,second=0, microsecond=0)

        #予想時間分早く終わらせて、終了時に結果がでるまで待たせる
        self.END_DATETIME_PRE = self.END_DATETIME - timedelta(seconds=self.PRED_TERM)


        if self.DEMO_FLG:
            self.DB_KEY = self.DB_KEY + "_DEMO"

        if self.SERVER_NAME == "win8":
            self.MODAL_NUM_FIRST = 4
            self.MODAL_NUM_SECOND = 5

            self.top_buy_button_x = 1800
            self.top_buy_button_y = 150

            self.modal_change_button_x = 1847
            self.modal_change_button_y = 164

            self.set_amt_x1 = 380
            self.set_amt_y1 = 304
            self.set_amt_x2 = 515
            self.set_amt_y2 = 400

            self.click_order_detail_x = 1905
            self.click_order_detail_y = 140

            self.click_account_x = 1894
            self.click_account_y = 104

            self.move_random_x1 = 500
            self.move_random_y1 = 400

            self.move_random_x2 = 500
            self.move_random_y2 = 400

            self.obs_x1 = 1300
            self.obs_y1 = 1060
            self.obs_x2 = 1230
            self.obs_y2 = 780

            self.regist_history_db_x = 1800
            self.regist_history_db_y = 820
            self.regist_history_db_scroll = -200

            self.reload_x = 93
            self.reload_y = 60

        elif self.SERVER_NAME == "win5":
            self.MODAL_NUM_FIRST = 5
            self.MODAL_NUM_SECOND = 6

            self.top_buy_button_x = 1800
            self.top_buy_button_y = 150

            self.modal_change_button_x = 1847
            self.modal_change_button_y = 164

            self.set_amt_x1 = 380
            self.set_amt_y1 = 304
            self.set_amt_x2 = 515
            self.set_amt_y2 = 400
            self.click_order_detail_x = 1905
            self.click_order_detail_y = 140

            self.click_account_x = 1894
            self.click_account_y = 104

            self.move_random_x1 = 500
            self.move_random_y1 = 400

            self.move_random_x2 = 500
            self.move_random_y2 = 400

            self.obs_x1 = 1300
            self.obs_y1 = 1060
            self.obs_x2 = 1230
            self.obs_y2 = 780

            self.regist_history_db_x = 1800
            self.regist_history_db_y = 820
            self.regist_history_db_scroll = -200

            self.reload_x = 93
            self.reload_y = 60