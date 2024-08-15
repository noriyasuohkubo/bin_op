import os
import logging.config
from decimal import Decimal
from util import *
import datetime
from datetime import timedelta

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_highlow.conf"))
loggerConf = logging.getLogger("highlow")

class ConfHighlow():
    def __init__(self):
        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win4"
        self.CHROME_VER = 110

        #self.ID = "HL100005"
        #self.PW = "Reikou0129"

        #飯塚たかし
        self.ID = "HLMI728202"
        self.PW = "King5469?"

        #self.ID = "demo"
        #self.PW = "demo"

        self.SECOND_FLG = False

        #取引制限あり・なし
        self.RESTRICT_FLG = True
        self.TRADE_TERM = 32 #取引制限ありの場合の取引間隔秒

        #取引額
        self.AMT = "1000"
        self.AMT_STR = "1,000"

        self.EXCEPT_DATETIME = [
            [
                datetime.datetime(year=2024, month=6, day=16, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=16, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=17, hour=1, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=17, hour=2, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=17, hour=12, minute=13, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=17, hour=12, minute=18, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=17, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=17, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=4, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=4, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=8, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=9, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=12, minute=36, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=13, minute=13, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=13, minute=18, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=14, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=18, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=18, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=7, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=8, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=8, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=9, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=10, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=11, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=14, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=19, hour=23, minute=48, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=19, hour=23, minute=53, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=7, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=7, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=7, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=8, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=10, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=11, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=14, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=23, minute=0, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=23, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=20, hour=23, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=20, hour=23, minute=35, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=5, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=6, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=6, minute=43, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=6, minute=48, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=7, minute=13, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=7, minute=18, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=7, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=7, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=7, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=8, minute=3, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=8, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=8, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=12, minute=28, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=12, minute=33, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=13, minute=43, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=13, minute=48, second=0, microsecond=0),
            ],
            [
                datetime.datetime(year=2024, month=6, day=21, hour=13, minute=58, second=0, microsecond=0),
                datetime.datetime(year=2024, month=6, day=21, hour=14, minute=3, second=0, microsecond=0),
            ],

        ]

        self.OANDA_CLOSE_FLG = False

        self.MAX_PROFIT_DAY = 900000
        self.DRAW_UP_MONEY = 0

        self.WIN_FLG = True
        self.NO_DEAL_FLG = False
        self.DEMO_FLG = False

        self.REQUEST_URL = "http://127.0.0.1:5000/"
        self.HOST = "127.0.0.1"
        self.DB_NO = 8
        self.FX_DB_NO = 0
        self.DB_KEY = ""
        self.DB_KEY_TRADE = ""
        self.DB_FX_DATA_KEY = ""
        self.TYPE = ""
        self.PAIR = ""
        self.PRED_TIME = ""
        self.ARG = ""
        self.TERM = 2

        self.PAIR_CHK_STR = ""
        self.TIME_CHK_STR = ""

        self.BET_CNT = 0
        self.TIMEOVER_CNT = 0
        self.SPREAD_ERR_CNT = 0
        self.TRADE_FAIL_CNT = 0

        #self.FX_DATA_MACHINE = "localhost"
        self.FX_DATA_MACHINE = "192.168.1.15"

        self.TRADE_STOP_FLG = False

        self.PREV_TRADE_TIME = None

        self.TIME_MAP = {"30":"30秒", "60":"1分"}
        self.TYPE_MAP = {"TRB":"Turbo", "SPR":"Turboスプレッド"}

        self.MAX_LEN = 3600
        self.EXCEPT_LIST = [20,21,22]
        self.END_TIME = 19
        self.END_TIME_MINUTE = 52

        self.INI_MONEY = 0

        self.balanceValueID = "balanceValue"
        self.pairPath = "//*[@id=\"scroll_panel_1_content\"]/div[2]/div/div[1]/div[1]/div[1]/div/div[1]"
        self.amountPath = "//*[@id=\"scroll_panel_1_content\"]/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/div/input"
        self.optionPath =   "//*[@id=\"scroll_panel_1_content\"]/div[2]/div/div[1]/div[1]/div[2]/div[2]"

        self.upButtonID = "//*[@id=\"TradePanel_oneClickHighButton__3OAFf\"]/div"
        self.downButtonID = "//*[@id=\"TradePanel_oneClickLowButton__3Oq9p\"]/div"
        self.strikePath = "//*[@id=\"scroll_panel_1_content\"]/div[2]/div/div[1]/div[1]/div[1]/div/div[3]"
        self.pipRangePath = "//*[@id=\"scroll_panel_1_content\"]/div[2]/div/div[1]/div[1]/div[3]/div[2]"
        self.msgPath = "//*[@id=\"root\"]/div/div[8]/div/div/span"

        self.optionDetailID = "TOOLTIP_ANCHOR_ID_OPEN_TRADES_TOGGLE"

        self.menuPath = '//*[@id="accountMenuToggleButton"]'
        self.logoutButton = '//*[@id="logoutAppMenuButton"]'

    def initial(self):
        self.PAIR_CHK_STR = self.PAIR[0:3] + "/" + self.PAIR[3:6]
        self.TIME_CHK_STR = self.TIME_MAP[self.PRED_TIME]
        self.TYPE_CHK_STR = self.TYPE_MAP[self.TYPE]

        if self.FX_DATA_MACHINE == "192.168.1.114":  # win2
            self.DB_FX_DATA_KEY = "Tradeview_" + self.PAIR + "_S1"
        elif self.FX_DATA_MACHINE == "192.168.1.15":  # win5
            self.DB_FX_DATA_KEY = "XM_" + self.PAIR + "#_S1"

        if self.PAIR == "NZDJPY":
            self.END_TIME = 17 #nzdjpyのみ終了時間が午前3時

        if self.DEMO_FLG:
            self.DB_KEY = self.DB_KEY + "_DEMO"
            self.DB_KEY_TRADE = self.DB_KEY_TRADE + "_DEMO"

        if self.RESTRICT_FLG:
            self.TRADE_TERM = 32