import os
import sys
import time
import datetime
# MetaTrader5はubuntuになし
import json
import numpy as np
import redis
import requests

import send_mail as mail
from ctypes import windll
import pytz
from datetime import timedelta
from decimal import Decimal
from util import *
import pyautogui as pag
import pyperclip
from copy import deepcopy
from app_usdjpy_fx_predict4_lgbm_907_class import *

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_predict4_907.conf"))
loggerConf = logging.getLogger("predict4_907")

class Predict4_907():
    def __init__(self):
        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win5_predict4_907"

        self.START_TIME = datetime.datetime(year=2025, month=1, day=6, hour=0, minute=1, second=45, microsecond=0)

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
        self.FX_DATA_MACHINE = "192.168.1.114"
        #self.FX_DATA_MACHINE = "169.254.175.171"
        self.FX_DB_NO = 0
        self.DB_FX_DATA_KEY = "{:.3f}"

        self.PAIR = "USDJPY"

        self.RATE_FORMAT = ""

        self.MAX_LEN = MAX_CLOSE_LEN - 1
        self.MAX_LEN_SEC = self.MAX_LEN * self.AI_MODEL_TERM

        self.END_DATETIME = None

        self.DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S'

        if self.FX_DATA_MACHINE == "192.168.1.114": #win5
            self.DB_FX_DATA_KEY = "VANTAGE_" + self.PAIR + ".p_S1"

        tmp_dt = datetime.datetime.now()
        if tmp_dt.hour in [22,23]:
            tmp_dt = tmp_dt + timedelta(days=1)

        #取引時間
        self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                                   hour=19, minute=59, second=0, microsecond=0)


def get_remote_close(conf, base_t_just):
    cnt = 0
    return_close = []
    while True:
        cnt += 1
        if cnt > 30:
            break

        result = redis_fx_db.zrangebyscore(conf.DB_FX_DATA_KEY, base_t_just - (conf.LOOP_TERM - conf.AI_MODEL_TERM),
                                           base_t_just, withscores=True)
        # conf.LOGGER(result)
        if len(result) == (conf.LOOP_TERM - conf.AI_MODEL_TERM + 1):

            for i in range(0, len(result), conf.AI_MODEL_TERM):
                line = result[i]
                body = line[0]

                tmps = json.loads(body)
                ask = tmps["ask"]
                bid = tmps["bid"]

                return_close.append(float(get_decimal_divide(get_decimal_add(ask, bid), "2")))
            # conf.LOGGER(return_close)
            return return_close
        time.sleep(0.01)
    return return_close

def registRedis(conf, redis_db, score, child, key):
    # 既存レコードがなければ追加
    tmp_val = redis_db.zrangebyscore(key, score, score)
    if len(tmp_val) == 0:
        redis_db.zadd(key, json.dumps(child), score)
    #redis_db.zadd(key, json.dumps(child), score)

def main_loop(conf,):
    return_code = 1
    err_flg = 0

    try:
        tmp_dt = datetime.datetime.now()
        base_dt = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                    hour=tmp_dt.hour, minute=tmp_dt.minute, second=tmp_dt.second, microsecond=0)

        #開始時刻設定 数秒待つ
        base_dt = base_dt + timedelta(seconds=(conf.LOOP_TERM * 2 + (conf.LOOP_TERM - tmp_dt.second % conf.LOOP_TERM)))
        base_t = time.mktime(base_dt.timetuple()) + 0.01
        conf.LOGGER("base_t", base_t)

        while True:
            time.sleep(0.0001)
            # print(datetime.now().microsecond)
            if (base_t - time.time()) < 0.0005:  # time.timeの誤差を考慮して0.5ミリ秒早く起きる
                break
            # もし追い越してしまったらエラーとする
            if (base_t - time.time()) < -0.01:
                conf.LOGGER("TIME START FAILED!!", base_t, time.time())
                err_flg = True

        if err_flg:
            # ここまででエラーあったら終了
            mail.send_message(subject=conf.SERVER_NAME, msg="SYSTEM ERROR OCCURED! EXIT!!")
            return 3

        first_loop = True
        closes = []
        closes_org = []

        close_take = 0.0
        predict_take = 0.0
        order_take = 0.0
        deal_take = 0.0
        db_take = 0.0
        loop_take = 0.0

        sleep_time = 0.0

        time_over_flg = False

        while (True):
            base_t_just = int(base_t - 0.01)  # base_tは0.01秒遅くなっているため
            base_t_just_score = int(base_t_just)
            base_t_just_dt = datetime.datetime.fromtimestamp(base_t_just_score)

            start = time.perf_counter()

            offset = base_t - time.time()  # 起動すべき時間と起動した時間の差
            tmp_offset = offset
            if tmp_offset < 0:
                tmp_offset = tmp_offset * -1
            # offsetが1000ミリ秒以上の場合メール送信 早くても遅くても駄目
            if tmp_offset > 1:
                conf.LOGGER("offset over 1000milces", offset, close_take, predict_take, order_take, deal_take, db_take,
                            loop_take, sleep_time)
                conf.TIMEOVER_CNT += 1
                err_flg = True
                time_over_flg = True
                break

            tdt = datetime.datetime.now()

            # 終了時間になったらポジションあれば決済し、抜ける
            if tdt >= conf.END_DATETIME:
                conf.LOGGER("main loop end!!")
                return_code = 2
                break

            start_close = time.perf_counter()

            #リモートPCのレート取得
            close = get_remote_close(conf, base_t_just)
            if len(close) == 0:
                conf.LOGGER("get remote close failed!!!")
                err_flg = True
                break

            close_take = time.perf_counter() - start_close

            # 過去分レート取得
            if first_loop:
                conf.LOGGER("first_loop get remote rate start")
                end_t = base_t_just - conf.LOOP_TERM
                start_t = end_t - (conf.MAX_LEN_SEC - conf.LOOP_TERM)
                result = redis_fx_db.zrangebyscore(conf.DB_FX_DATA_KEY, start_t, end_t, withscores=True)

                scores = []
                # print(db ,len(result))
                for i in range(0, len(result), conf.AI_MODEL_TERM):  # AI_MODEL_TERM秒おきのデータのみ必要なのでLOOP_TERMおきに取得
                    line = result[i]
                    body = line[0]
                    tmp_score = int(line[1]) - conf.AI_MODEL_TERM  # oandaのデータはopenなのでcloseに合わせるためにスコア調整
                    scores.append(tmp_score)

                    tmps = json.loads(body)
                    ask = tmps["ask"]
                    bid = tmps["bid"]
                    tmp_close = float(get_decimal_divide(get_decimal_add(ask, bid), "2"))
                    closes.append(tmp_close)

                if len(closes) != conf.MAX_LEN - (int(get_decimal_divide(conf.LOOP_TERM, conf.AI_MODEL_TERM)) - 1):
                    conf.LOGGER("Data Short! length:", len(closes))
                    err_flg = True
                    break
                conf.LOGGER("first_loop get remote rate end")

            for tmp_c in close:
                closes.append(tmp_c)  # 最初に取得したレートを追加
                closes_org.append(tmp_c)


            while True:
                if len(closes_org) > conf.MAX_LEN + 1:
                    closes_org.pop(0)
                else:
                    break

            while True:
                if len(closes) > conf.MAX_LEN + 1:
                    closes.pop(0)
                else:
                    break

            now_rate = close[-1]

            # 3分間レートが変わっていなかったら異常発生としメール送信！
            if len(closes_org) >= 180:
                startInd = len(closes_org) - 179
                rate_err = True
                for j in range(179):
                    if closes_org[startInd - 1] != closes_org[startInd + j]:
                        # 変化あったらエラーなし
                        rate_err = False
                        break

                if rate_err:
                    conf.LOGGER("rate has not Changed for 3 min !")
                    err_flg = True
                    break


            # 予想取得
            start_predict = time.perf_counter()
            try:
                """
                json_data = {'score': base_t_just, 'vals': closes}
                response = requests.post(conf.REQUEST_URL, json=json_data)
                response_text = response.text
                """
                response_text = predict_class.get_predict(base_t_just, closes)

            except Exception as request_e:
                conf.LOGGER(tracebackPrint(request_e))
                if request_e.__str__() == "cannot get predict. restart":
                    #再起動させる
                    return 4
                else:
                    err_flg = True
                    break

            predict_take = time.perf_counter() - start_predict

            start_db = time.perf_counter()

            regist_score = int(base_t_just_score)
            regist_time_str = datetime.datetime.fromtimestamp(regist_score)
            child = {
                'response': response_text,
                'now_rate': now_rate,
                'predict_take': '{:.3f}'.format(predict_take),
                'time': str(regist_time_str),
                'loop_take': time.perf_counter() - start,
                'db_take': db_take,
                'close_take': close_take,
            }

            registRedis(conf, redis_db, regist_score, child, conf.DB_KEY)
            db_take = time.perf_counter() - start_db

            # 処理時間表示
            end = time.perf_counter()
            # 処理時間がconf.LOOP_TERM + 0.9以上の場合
            process_t = end - start
            if process_t > (conf.LOOP_TERM + 0.9):
                conf.LOGGER("time over:", process_t, close_take, predict_take, db_take, loop_take, sleep_time, )

            if first_loop:
                first_loop = False

            # 次に起動すべき時間
            base_t += conf.LOOP_TERM

            # 次のターンまでスリープする
            start_loop = time.perf_counter()

            sleep_time = base_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_take = time.perf_counter() - start_loop

        if err_flg:
            return_code = 3

    except Exception as e:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))

        return_code = 3

    return return_code


if __name__ == '__main__':

    # タイマー精度を1msec単位にする
    windll.winmm.timeBeginPeriod(1)

    err_flg = False

    conf = Predict4_907()

    conf.LOGGER("PROCESS ID:", os.getpid())

    conf.LOGGER("start init predict class")
    predict_class = AppUsdjpyFxPredict4Lgbm907Class(conf.LOGGER)
    conf.LOGGER("end init predict class")

    while True:
        now_dt = datetime.datetime.now()
        if now_dt< conf.START_TIME:
            time.sleep(60)
            #print("sleep")
        else:
            break


    redis_db = redis.Redis(host=conf.HOST, port=6379, db=conf.DB_NO, decode_responses=True)
    # RedisのオートSave設定を無効にする
    print(redis_db.config_set("save", ""))

    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=True)
    MAIN_LOOP_CNT = 0
    MAIN_LOOP_CNT_MAX = 10
    try:
        # メイン処理を繰り返す
        while True:

            return_code = main_loop(conf, )
            conf.LOGGER("return_code:", return_code)

            if return_code == 2:
                # 正常な処理終了
                # 不要な一時予想DBレコードを削除
                for tmp_db in base_models:
                    r_db = redis.Redis(host=tmp_db["db_host"], port=6379, db=tmp_db["db_no"], decode_responses=True)
                    r_db.delete(tmp_db["db_name"])
                    
                break
            elif return_code == 3:
                # 異常あり
                mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")
                break
            elif return_code == 4:
                # 再ループ
                MAIN_LOOP_CNT += 1
                conf.LOGGER("MAIN_LOOP_CNT:", MAIN_LOOP_CNT)
                if MAIN_LOOP_CNT >= MAIN_LOOP_CNT_MAX:
                    #ループ最大回数を越えたら終了
                    conf.LOGGER("main loop cnt over: " + str(MAIN_LOOP_CNT_MAX))
                    mail.send_message(conf.SERVER_NAME, msg="main loop cnt over")
                    break

    except Exception as e:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))
        mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")

    # タイマー精度を戻す
    windll.winmm.timeEndPeriod(1)

    exit(0)