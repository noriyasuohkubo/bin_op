import os
import logging.config
from decimal import Decimal
from util import *
import datetime
from datetime import timedelta
from app_usdjpy_fx_predict4_lgbm_907_conf import *

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_predict4_907.conf"))
loggerConf = logging.getLogger("predict4_907")

class Predict4_907():
    def __init__(self):
        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win3_predict4_907"

        self.START_TIME = datetime.datetime(year=2024, month=9, day=15, hour=23, minute=1, second=0, microsecond=0)

        self.lgbm_model_file = lgbm_model_file
        self.lgbm_model_file_suffix = lgbm_model_file_suffix
        self.lgbm_model_file_ext = lgbm_model_file_ext
        self.lgbm_model_file_suffix_ext = lgbm_model_file_suffix_ext

        self.AI_MODEL_TERM = AI_MODEL_TERM  # AIモデルの最小データ間隔
        self.LOOP_TERM = LOOP_TERM

        self.HOST = 'localhost'
        self.DB_NO = PREDICT_REQUEST_DB_NO
        self.DB_KEY = PREDICT_REQUEST_KEY

        self.REQUEST_URL = "http://127.0.0.1:8004/"
        #self.FX_DATA_MACHINE = "192.168.1.15"
        self.FX_DATA_MACHINE = "192.168.1.114"
        self.FX_DB_NO = 0
        self.DB_FX_DATA_KEY = ""

        self.PAIR = ""

        self.RATE_FORMAT = ""

        self.MAX_LEN = MAX_CLOSE_LEN - 1
        self.MAX_LEN_SEC = self.MAX_LEN * self.AI_MODEL_TERM

        self.END_DATETIME = None

        self.DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S'

    def initial(self):
        if self.FX_DATA_MACHINE == "192.168.1.15":
            self.DB_FX_DATA_KEY = "VANTAGE_" + self.PAIR + ".p_S1"
        elif self.FX_DATA_MACHINE == "192.168.1.114": #win2
            #self.DB_FX_DATA_KEY = "Tradeview_" + self.PAIR + "_S1"
            self.DB_FX_DATA_KEY = "THREETRADER_" + self.PAIR + "_S1"
            self.FX_DB_NO = 5
        elif self.FX_DATA_MACHINE == "192.168.1.115": #win4
            self.DB_FX_DATA_KEY = "XM_" + self.PAIR + "#_S1"

        tmp_dt = datetime.datetime.now()
        if tmp_dt.hour in [22,23]:
            tmp_dt = tmp_dt + timedelta(days=1)

        #取引時間
        self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                                   hour=19, minute=59, second=0, microsecond=0)
