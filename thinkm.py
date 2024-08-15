import os
import sys
import time
import datetime
# MetaTrader5はubuntuになし
import json
import numpy as np
import redis
import requests
import selenium

import send_mail as mail
from ctypes import windll
import pytz
from datetime import timedelta
from decimal import Decimal
import conf_thinkm
import conf_thinkm_eurusd
from util import *
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome import service as fs
import pyautogui as pag
import pyperclip
from copy import deepcopy
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

#起動時に1000だけ取引して、発注ボタンおしても反応しないのを防ぐ
def eurusd_init(conf, driver):
    # 取引量設定
    oneclick_positionE = driver.find_element(By.XPATH, conf.ONECLICK_POSITION_INPUT_PATH)
    # print(oneclick_positionE.get_attribute("value"))

    pag.moveTo(x=conf.set_amt_x1, y=conf.set_amt_y1)
    pag.click()
    time.sleep(0.1)
    # oneclick_positionE.clear()
    pag.press("backspace")
    time.sleep(0.1)
    pag.press("backspace")
    time.sleep(0.1)
    pag.press("backspace")
    time.sleep(0.1)
    pag.press("backspace")
    time.sleep(0.1)
    pag.press("backspace")
    time.sleep(0.1)
    pag.press("backspace")
    time.sleep(0.1)

    # conf.LOGGER("cleared!!!")
    oneclick_positionE.send_keys("1000")
    time.sleep(0.5)
    # conf.LOGGER("input!!!")
    pag.moveTo(x=conf.set_amt_x2, y=conf.set_amt_y2)
    pag.click()
    time.sleep(0.5)

    wait = WebDriverWait(driver, 1)

    wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_ONECLICK_PATH))).click()

    #pag.moveTo(x=conf.ANALOG_BUY_X, y=conf.ANALOG_BUY_Y, duration=0.2)
    #pag.click()

    pag.moveTo(x=conf.move_random_x2, y=conf.move_random_y2)

#過去24時間での取引回数取得
def get_trade_cnt(conf, redis_db, base_t_just):

    result = redis_db.zrangebyscore(conf.DB_ORDER_KEY, base_t_just - (3600 * 24), base_t_just,withscores=True)

    return len(result)

def get_pair(conf, driver, line_num):
    wait = WebDriverWait(driver, 4)

    tmp_path = conf.HISTORY_TR_PAIR.replace("NUM", str(line_num + 1))
    pair = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text

    tmp_path = conf.HISTORY_TR_PATH.replace("NUM", str(line_num + 1))
    history_trE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))

    bet_type = history_trE.find_element(By.XPATH, conf.HISTORY_BET_TYPE).text

    open_rate = history_trE.find_element(By.XPATH, conf.HISTORY_OPEN_RATE).text
    close_rate = history_trE.find_element(By.XPATH, conf.HISTORY_CLOSE_RATE).text

    open_date = history_trE.find_element(By.XPATH, conf.HISTORY_OPEN_DAY).text
    close_date = history_trE.find_element(By.XPATH, conf.HISTORY_CLOSE_DAY).text

    open_time = history_trE.find_element(By.XPATH, conf.HISTORY_OPEN_TIME).text
    close_time = history_trE.find_element(By.XPATH, conf.HISTORY_CLOSE_TIME).text

    stoploss_rate = history_trE.find_element(By.XPATH, conf.HISTORY_STOPLOSS).text

    position_number = history_trE.find_element(By.XPATH, conf.HISITORY_POSITION_NUM).text
    #print(pair,bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate)

    return [pair,bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate, position_number]


def get_history_num(conf, driver):
    wait = WebDriverWait(driver, 5)
    history_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_NUM)))
    history_num = int(history_numE.text.split("(")[1].split(")")[0])

    return history_num

def get_newest_history_score(conf, redis_db):
    db_data = redis_db.zrevrange(conf.DB_HISTORY_KEY, 0, 0, withscores=True)
    db_score = None
    # DB上の一番新しいスコアを取得
    if len(db_data) != 0:
        for d in db_data:
            body = d[0]
            db_score = int(d[1])

    return db_score

def regist_history_db(conf,driver, redis_db):
    wait = WebDriverWait(driver, 5)
    #建玉タブ表示
    wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH))).click()

    #履歴タブ表示
    wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_PATH))).click()

    retry_cnt = 0
    while True:
        try:
            #履歴期間選択リストボックス表示
            wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_LIST_PATH))).click()
            break
        except Exception as e:
            if retry_cnt >=10:
                #リトライ上限に達したらエラー扱い
                conf.LOGGER("regist_history_db retry_cnt over!!! ", retry_cnt)
                raise Exception("regist_history_db retry_cnt over!!!")
            else:
                #まだ一覧が開けていないのでスリープ後にリトライ
                time.sleep(10)
                retry_cnt +=1

    # 履歴期間で一週間を選択
    wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_LIST_WEEK))).click()
    time.sleep(5)
    # 履歴件数取得
    history_num = get_history_num(conf, driver)

    if history_num !=0:
        #開始日の降順にする
        sortE = wait.until(EC.presence_of_element_located((By.XPATH, conf.OPEN_DAY_SORT)))
        print(sortE.get_attribute("class"))

        if ('dx-sort-down' in sortE.get_attribute("class")) == False:

            #降順になってなかったら二回クリックして降順にする
            wait.until(EC.presence_of_element_located((By.XPATH, conf.OPEN_DATE_COL))).click()
            time.sleep(10)
            wait.until(EC.presence_of_element_located((By.XPATH, conf.OPEN_DATE_COL))).click()
            time.sleep(10)
            sortE = wait.until(EC.presence_of_element_located((By.XPATH, conf.OPEN_DAY_SORT)))
            if ('dx-sort-down' in sortE.get_attribute("class")) == False:
                #ソートできないのでエラー扱い
                conf.LOGGER("regist_history_db cannnot sort")
                raise Exception("regist_history_db cannnot sort")

    regist_cnt = 0

    print("history_num",history_num)

    db_score = get_newest_history_score(conf, redis_db)
    conf.LOGGER("newest db score:", db_score)
    conf.LOGGER("zremrangebyscore " + conf.DB_HISTORY_KEY + " " + str(db_score) + " " + str(db_score + 3600 * 24)) # in case of fail, remove sentence
    for line_num in range(history_num):
        pair,bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate, position_number = get_pair(conf, driver, line_num)

        tmp_list = [line_num, pair,bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate]

        while True:
            if "" in tmp_list:
                #取得できない値がある場合は下にスクロールして取得しなおす
                pag.moveTo(x=conf.regist_history_db_x, y=conf.regist_history_db_y)
                time.sleep(1)
                pag.scroll(conf.regist_history_db_scroll) #-400で大体10行
                time.sleep(2)
                pair, bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate, position_number = get_pair(conf, driver, line_num)
                tmp_list = [line_num, pair, bet_type, open_rate, close_rate, open_date, close_date, open_time,
                            close_time, stoploss_rate]
            else:
                break

        print(tmp_list)

        # レートをDBに登録
        open_dt = datetime.datetime.strptime(open_date + " " + open_time, '%Y/%m/%d %H:%M:%S')
        close_dt = datetime.datetime.strptime(close_date + " " + close_time, '%Y/%m/%d %H:%M:%S')

        open_score = int(time.mktime(open_dt.timetuple()))
        close_score = int(time.mktime(close_dt.timetuple()))

        regist_score = open_score - (open_score % conf.LOOP_TERM)  # 発注時から遅れて約定されるので発注時のスコアを登録する

        position_number = int(position_number.replace(",", ""))

        #既存レコードがないか確認
        if db_score == None or regist_score > db_score:
            if pair == conf.PAIR:
                if bet_type == '買い':
                    bet_str = 'buy'
                elif bet_type == '売り':
                    bet_str = 'sell'
                else:
                    conf.LOGGER("bet_type is incorrect", bet_type)
                    raise Exception("bet_type is incorrect")

                child = {
                    'bet_type': bet_str,
                    'open_rate': open_rate,
                    'close_rate': close_rate,
                    'stoploss_rate': stoploss_rate, #値が設定されていなければ半角空白が入る
                    'open_score': open_score,
                    'close_score': close_score,
                    'open_time': datetime.datetime.fromtimestamp(open_score).strftime('%Y/%m/%d %H:%M:%S'),
                    'position_number': position_number,
                }

                redis_db.zadd(conf.DB_HISTORY_KEY, json.dumps(child), regist_score)
                regist_cnt += 1
        else:
            break

    conf.LOGGER("history regist cnt:", regist_cnt)

    wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH))).click()

#最新の履歴情報の新規時刻を返す
def regist_latest_history(conf,driver):
    wait = WebDriverWait(driver, 2)
    try:
        return_val = None
        #履歴タブ表示
        wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_PATH))).click()

        # 履歴件数取得
        history_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_NUM)))
        history_num = int(history_numE.text.split("(")[1].split(")")[0])

        if history_num != 0:
            #最新の履歴情報取得
            try:
                pair, bet_type, open_rate, close_rate, open_date, close_date, open_time, close_time, stoploss_rate, position_number = get_pair(conf, driver, 0)
                return_val = open_time
            except selenium.common.exceptions.StaleElementReferenceException as e:
                #エラー発生時は-1を返す
                return_val = -1
            except selenium.common.exceptions.NoSuchElementException as e:
                return_val = -1
            except selenium.common.exceptions.TimeoutException as e:
                return_val = -1

    except selenium.common.exceptions.TimeoutException as e:
        return_val = -1

    wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH))).click()

    return  return_val

def set_amt(conf, driver):
    # 取引量設定
    oneclick_positionE = driver.find_element(By.XPATH, conf.ONECLICK_POSITION_INPUT_PATH)
    # print(oneclick_positionE.get_attribute("value"))
    if conf.AMT_STR != oneclick_positionE.get_attribute("value"):
        # oneclick_positionE = driver.find_element(By.XPATH, conf.ONECLICK_POSITION_INPUT_PATH)
        # oneclick_positionE.click()
        pag.moveTo(x=conf.set_amt_x1, y=conf.set_amt_y1)
        pag.click()
        time.sleep(0.1)
        # oneclick_positionE.clear()
        pag.press("backspace")
        time.sleep(0.1)
        pag.press("backspace")
        time.sleep(0.1)
        pag.press("backspace")
        time.sleep(0.1)
        pag.press("backspace")
        time.sleep(0.1)
        pag.press("backspace")
        time.sleep(0.1)
        pag.press("backspace")
        time.sleep(0.1)

        # conf.LOGGER("cleared!!!")
        oneclick_positionE.send_keys(conf.AMT)
        time.sleep(0.5)
        # conf.LOGGER("input!!!")
        pag.moveTo(x=conf.set_amt_x2, y=conf.set_amt_y2)
        pag.click()
        time.sleep(0.5)


def delete_chart(driver):
    conf.LOGGER("delete chart start")
    try:
        # チャートを削除するためUSDJPYのチャートだけ表示する
        #driver.find_element(By.XPATH,"//*[@id=\"root\"]/div/div/div[2]/div[3]/div[5]/div/div[2]/div[1]/div[1]/div/cq-context/div[4]/div[1]/cq-context-wrapper[1]/cq-context/div/div[3]").click()
        #time.sleep(3)

        # チャートを削除する
        chartE = driver.find_element(By.XPATH, conf.DELETE_CHART)
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, chartE)
    except Exception as e:
        conf.LOGGER(tracebackPrint(e))
        pass

def click_reaload(conf):
    pag.moveTo(x=conf.reload_x, y=conf.reload_y)
    pag.click()
    time.sleep(30)

def click_order_detail(conf):
    pag.moveTo(x=conf.click_order_detail_x, y=conf.click_order_detail_y)
    pag.click()


def click_account(conf, driver):
    pag.moveTo(x=conf.click_account_x, y=conf.click_account_y)
    pag.click()


def move_random(conf, driver):
    pag.moveTo(x=conf.move_random_x1, y=conf.move_random_y1)
    pag.click()
    pag.moveTo(x=conf.move_random_x2, y=conf.move_random_y2)
    wait = WebDriverWait(driver, 1)
    wait.until(EC.presence_of_element_located((By.XPATH, conf.RESEARVE_PATH))).click()
    wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH))).click()

def switch_demo_live(conf, driver):
    time.sleep(15)
    cnt = 0
    err = None
    while True:
        cnt += 1
        try:
            click_account(conf, driver)
            time.sleep(15)
            driver.find_element(By.XPATH, conf.SWITCH_LIVE_PATH).click()
            break
        except Exception as e:
            conf.LOGGER("click exception", cnt)
            time.sleep(2)
            err = e
        if cnt >= 5:
            conf.LOGGER(tracebackPrint(err))
            raise Exception("cannnot click account button")

    time.sleep(15)
    if driver.find_element(By.XPATH, conf.ACCOUNT_TYPE_PATH).text != "ライブ":
        raise Exception("cannnot switch")
    driver.find_element(By.XPATH, conf.ACCOUNT_PATH).click()
    time.sleep(15)
    driver.find_element(By.XPATH, conf.SWITCH_DEMO_PATH).click()
    time.sleep(15)

    if driver.find_element(By.XPATH, conf.ACCOUNT_TYPE_PATH).text != "デモ":
        raise Exception("cannnot switch")


def start_obs(conf):

    conf.LOGGER("obs start")
    #OBSを開く
    pag.moveTo(x=conf.obs_x1, y=conf.obs_y1)
    pag.click()
    time.sleep(5)

    pag.moveTo(x=conf.obs_x2, y=conf.obs_y2) #録画ボタン押下
    pag.click()
    time.sleep(5)


    #OBSをとじる
    pag.moveTo(x=conf.obs_x1, y=conf.obs_y1)
    pag.click()
    time.sleep(5)


def stop_obs(conf):
    conf.LOGGER("obs stop")
    #OBSを開く
    pag.moveTo(x=conf.obs_x1, y=conf.obs_y1)
    pag.click()
    time.sleep(5)

    pag.moveTo(x=conf.obs_x2, y=conf.obs_y2) #録画停止ボタン押下
    pag.click()
    time.sleep(5)


    #OBSをとじる
    pag.moveTo(x=conf.obs_x1, y=conf.obs_y1)
    pag.click()
    time.sleep(5)

def get_predict(conf, base_t_just_score):

    cnt = 0

    while True:
        cnt += 1
        if cnt > 30:
            break

        result = redis_predict_db.zrangebyscore(conf.PREDICT_REQUEST_KEY, base_t_just_score, base_t_just_score, withscores=True)
        # conf.LOGGER(result)
        if len(result) == 1:
            line = result[0]
            body = line[0]

            tmps = json.loads(body)
            response = tmps["response"]
            now_rate = tmps["now_rate"]

            return response, now_rate

        time.sleep(0.02)

    raise Exception("cannot get predict")

def get_close_local(conf, driver):
    wait = WebDriverWait(driver, 1)
    try:
        buy_rateE = wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_RATE_PATH)))
        buy_rate = float(buy_rateE.text)
        #print(buy_rate)
        """
        if conf.PAIR == 'USDJPY':
            buy_rateE = wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_RATE_PATH)))
            buy_rate1 = buy_rateE.find_element(By.XPATH, "span[1]").text
            buy_rate2 = buy_rateE.find_element(By.XPATH, "span[2]").text
            buy_rate3 = buy_rateE.find_element(By.XPATH, "span[3]").text
            buy_rate4 = buy_rateE.find_element(By.XPATH, "span[5]").text
            buy_rate5 = buy_rateE.find_element(By.XPATH, "span[6]").text
            buy_rate6 = buy_rateE.find_element(By.XPATH, "span[7]").text
            buy_rate = float(buy_rate1 + buy_rate2 + buy_rate3 + "." + buy_rate4 + buy_rate5 + buy_rate6)
            
        elif conf.PAIR == 'EURUSD':
            buy_rate1 = buy_rateE.find_element(By.XPATH, "span[1]").text
            buy_rate2 = buy_rateE.find_element(By.XPATH, "span[3]").text
            buy_rate3 = buy_rateE.find_element(By.XPATH, "span[4]").text
            buy_rate4 = buy_rateE.find_element(By.XPATH, "span[5]").text
            buy_rate5 = buy_rateE.find_element(By.XPATH, "span[6]").text
            buy_rate6 = buy_rateE.find_element(By.XPATH, "span[7]").text
            buy_rate = float(buy_rate1 + "."+ buy_rate2 + buy_rate3 +  buy_rate4 + buy_rate5 + buy_rate6)
        """

        return buy_rate
    except selenium.common.exceptions.StaleElementReferenceException as e:
        conf.CLOSE_LOCAL_ERR_CNT += 1
        if conf.CLOSE_LOCAL_ERR_CNT >= 300:
            raise Exception("CLOSE_LOCAL_ERR_CNT over 300!!!")
        return 0


def get_close_local_oneclick(conf, driver):
    wait = WebDriverWait(driver, 1)
    try:
        buy_rateE = wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_RATE_PATH_ONECLICK)))
        buy_rate = float(buy_rateE.text)

        return buy_rate
    except selenium.common.exceptions.StaleElementReferenceException as e:
        conf.CLOSE_LOCAL_ERR_CNT += 1
        if conf.CLOSE_LOCAL_ERR_CNT >= 300:
            raise Exception("CLOSE_LOCAL_ERR_CNT over 300!!!")
        return 0

def get_oanda_close(conf, base_t_just):
    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=False)

    cnt = 0
    return_close = []
    while True:
        cnt += 1
        if cnt > 5:
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
        time.sleep(0.05)
    return return_close

def get_oanda_close_usdjpy(conf):
    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=False)

    db_data = redis_fx_db.zrevrange(conf.DB_FX_DATA_KEY_USDJPY, 0, 0, withscores=True)
    return_close = None
    # DB上の一番新しいスコアを取得
    if len(db_data) != 0:
        line = db_data[0]
        body = line[0]
        tmps = json.loads(body)
        #print(tmps)
        ask = tmps["ask"]
        bid = tmps["bid"]

        return_close = float(get_decimal_divide(get_decimal_add(ask, bid), "2"))

    return return_close


def get_oanda_foot(conf):
    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=False)
    return_dict = {}

    for foot, foot_len in conf.FOOT_DICT.items():
        db_name_tmp = conf.FOOT_DB_NAME_PREFIT + str(foot)
        now_stamp = int(time.mktime(datetime.datetime.now().timetuple()))
        #最後に取得すべきデータのスコア
        last_regist_stamp = now_stamp - (now_stamp % (foot * 60)) - (foot * 60)

        close_list = []
        high_list = []
        low_list = []
        cnt = 0
        while True:
            cnt += 1
            if cnt > 10:
                break

            result = redis_fx_db.zrange(db_name_tmp, foot_len * -1, -1, withscores=True)
            # conf.LOGGER(result)
            if len(result) == foot_len:
                # 最後のデータのスコアをチェック
                last_line = result[foot_len -1]
                if last_line[1] != last_regist_stamp:
                    time.sleep(0.05)
                    continue
                else:
                    for i,line in enumerate(result):
                        body = line[0]
                        tmps = json.loads(body)
                        close_list.append(float(tmps["close"]))
                        high_list.append(float(tmps["high"]))
                        low_list.append(float(tmps["low"]))

                    return_dict[foot] = {
                        "close":close_list,
                        "high": high_list,
                        "low": low_list,
                    }
                    break
            else:
                conf.LOGGER("data lenght:", len(result))
                return None

        if (foot in return_dict.keys()) == False:
            #データ取得できていなかった場合Noneを返す
            return None

    return return_dict

def registRedis(conf, redis_db, score, child, key):
    # 既存レコードがなければ追加
    tmp_val = redis_db.zrangebyscore(key, score, score)
    if len(tmp_val) == 0:
        redis_db.zadd(key, json.dumps(child), score)


def login(conf, driver):
    try:
        if conf.DEMO_FLG:
            driver.find_element(By.XPATH, conf.DEMO_SELECT_PATH).click()
        else:
            driver.find_element(By.XPATH, conf.LIVE_SELECT_PATH).click()

        time.sleep(1)

        inputE = driver.find_element(By.XPATH, conf.ID_INPUT_PATH)
        inputE.clear()
        inputE.send_keys(conf.ID)
        time.sleep(1)

        inputE = driver.find_element(By.XPATH, conf.PW_INPUT_PATH)
        inputE.clear()
        inputE.send_keys(conf.PW)
        time.sleep(1)

        driver.find_element(By.XPATH, conf.LOGIN_PATH).click()
    except Exception as e:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))
        #すでにログイン状態の可能性があるので、ここで処理終了
        return 1

    return 0

def get_hcaptcha(conf, driver):
    try:
        hcaptchaE = driver.find_element(By.ID, "hcaptcha")
        conf.LOGGER("data-sitekey:", hcaptchaE.get_attribute("data-sitekey"))
        conf.LOGGER(driver.page_source)
        return True
    except selenium.common.exceptions.NoSuchElementException as e:
        # conf.LOGGER("NoSuchElementException:hcaptcha")
        return False

    return False


def get_position_num(conf, driver):
    wait = WebDriverWait(driver, 1)
    try:
        position_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH)))

        # position_numE = driver.find_element(By.XPATH, conf.POSITION_NUM_PATH)
        position_num = int(position_numE.text.split("(")[1].split(")")[0])
    except selenium.common.exceptions.StaleElementReferenceException as e:
        #conf.LOGGER("StaleElementReferenceException:conf.POSITION_NUM_PATH")
        position_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH)))
        # position_numE = driver.find_element(By.XPATH, conf.POSITION_NUM_PATH)
        position_num = int(position_numE.text.split("(")[1].split(")")[0])
    except IndexError as e:
        #conf.LOGGER(e)
        #conf.LOGGER("position_numE.text:",position_numE.text)
        position_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_NUM_PATH)))
        # position_numE = driver.find_element(By.XPATH, conf.POSITION_NUM_PATH)
        position_num = int(position_numE.text.split("(")[1].split(")")[0])

    return position_num

def get_history_num(conf, driver):
    wait = WebDriverWait(driver, 1)

    history_numE = wait.until(EC.presence_of_element_located((By.XPATH, conf.HISTORY_NUM)))
    history_num = int(history_numE.text.split("(")[1].split(")")[0])

    return history_num

def get_money(conf, driver):
    moneyE = driver.find_element(By.XPATH, conf.MONEY_PATH)
    return int(float(moneyE.text.split(" ")[1].replace(",", "")))

def get_stoploss(conf, driver, line_num=None):
    wait = WebDriverWait(driver, 1)
    if line_num == None:
        try:
            slE = driver.find_element(By.XPATH, conf.POSITION_1_STOPLOSS)
            sl_text = slE.text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            slE = driver.find_element(By.XPATH, conf.POSITION_1_STOPLOSS)
            sl_text = slE.text
        except Exception as e2:
            #まだストップロスが設定されていない
            return float(0)

    else:
        tmp_path = conf.POSITION_2_STOPLOSS.replace("NUM", str(line_num))
        try:
            slE = driver.find_element(By.XPATH, tmp_path)
            sl_text = slE.text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            slE = driver.find_element(By.XPATH, tmp_path)
            sl_text = slE.text
        except Exception as e2:
            # まだストップロスが設定されていない
            return float(0)

    return float(sl_text)

def get_start_text(conf, driver, line_num=None):
    wait = WebDriverWait(driver, 1)

    if line_num == None:
        try:
            start_text = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_START_TIME_PATH))).text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            start_text = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_START_TIME_PATH))).text
        except selenium.common.exceptions.TimeoutException as e:
            start_text = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_START_TIME_PATH))).text

        if len(start_text.split(":")) != 3:
            # 正しく取得できていない場合もう一度取得
            start_text = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_START_TIME_PATH))).text

        if len(start_text.split(":")) != 3:
            conf.LOGGER("start_text1:", start_text)
            raise Exception("cannot get start_text")
    else:
        tmp_path = conf.POSITION_2_START_TIME_PATH.replace("NUM", str(line_num))
        try:
            start_text = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            start_text = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text
        except selenium.common.exceptions.TimeoutException as e:
            start_text = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_START_TIME_PATH))).text

        if len(start_text.split(":")) != 3:
            # 正しく取得できていない場合もう一度取得
            start_text = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text

        if len(start_text.split(":")) != 3:
            conf.LOGGER("start_text2:", start_text)
            raise Exception("cannot get start_text")

    return start_text

def get_new_rate(conf, driver, line_num=None):
    wait = WebDriverWait(driver, 1)
    new_rate = None

    if line_num == None:
        try:
            new_rate = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_NEW_RATE))).text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            new_rate = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_NEW_RATE))).text
        except selenium.common.exceptions.TimeoutException as e:
            new_rate = wait.until(
                EC.presence_of_element_located((By.XPATH, conf.POSITION_1_NEW_RATE))).text

    else:
        tmp_path = conf.POSITION_2_NEW_RATE.replace("NUM", str(line_num))
        try:
            new_rate = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text
        except selenium.common.exceptions.StaleElementReferenceException as e:
            new_rate = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text
        except selenium.common.exceptions.TimeoutException as e:
            new_rate = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_NEW_RATE))).text

    new_rate = float(new_rate)
    return new_rate

def get_profit(order_start_rate_tm, now_close, position_type):
    if position_type == 0:
        #買いの場合
        profit = get_decimal_sub(now_close, order_start_rate_tm)
    else:
        profit = get_decimal_sub(order_start_rate_tm, now_close)

    return profit

def chk_pair(conf, driver):
    pair_text = driver.find_element(By.XPATH, conf.PAIR_PATH).text

    if conf.PAIR != pair_text:
        conf.LOGGER("pair is incorrect ", pair_text)
        return False
    else:
        return True


def logout(conf, driver):
    time.sleep(15)
    cnt = 0
    while True:
        cnt += 1
        try:
            click_account(conf, driver)
            time.sleep(15)
            if conf.DEMO_FLG:
                driver.find_element(By.XPATH, conf.LOG_OUT_DEMO).click()
            else:
                driver.find_element(By.XPATH, conf.LOG_OUT_LIVE).click()
            break
        except Exception as e:
            try:
                driver.find_element(By.XPATH, conf.LOG_OUT_LIVE2).click()
                break
            except Exception as e:
                try:
                    driver.find_element(By.XPATH, conf.LOG_OUT_LIVE3).click()
                    break
                except Exception as e:
                    conf.LOGGER("logout click exception", cnt)
                    time.sleep(2)
        if cnt >= 5:
            raise Exception("cannnot click logout button")
    time.sleep(10)
    driver.find_element(By.XPATH, conf.LOG_OUT_OK).click()

# 全て決済する
def do_deal_all(conf, driver, catch_except=True, sleep_time=5):

    if catch_except:
        #例外キャッチする場合:すでにほかのエラーが出ていて残っているポジションをキャッチする場合
        try:
            wait = WebDriverWait(driver, 3)
            position_num = get_position_num(conf, driver)
            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_ALL_DEAL_BUTTON_PATH))).click()

            if position_num == 1:
                time.sleep(2)
                tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
                try:
                    deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                    #deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                except selenium.common.exceptions.NoSuchElementException as e:
                    tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                    try:
                        deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                        # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                    except selenium.common.exceptions.NoSuchElementException as e:
                        #画面が自動更新されてモーダルでなくなった可能性がある
                        modal_change(conf)
                        tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
                        try:
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                            # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                        except selenium.common.exceptions.NoSuchElementException as e:
                            tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                            # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))

                deal_buttonE.click()

            elif position_num > 1:
                tmp_path = conf.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
                try:
                    deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                    #deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                except selenium.common.exceptions.NoSuchElementException as e:
                    tmp_path = conf.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                    try:
                        deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                        # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                    except selenium.common.exceptions.NoSuchElementException as e:
                        #画面が自動更新されてモーダルでなくなった可能性がある
                        modal_change(conf)
                        tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
                        try:
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                            # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
                        except selenium.common.exceptions.NoSuchElementException as e:
                            tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                            # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))

                deal_buttonE.click()

            time.sleep(sleep_time)  # 決済に要する時間待つ
        except Exception as e:
            conf.LOGGER("Error Occured!!:", e)
            conf.LOGGER(tracebackPrint(e))
    else:
        #例外キャッチしない場合:まだほかのエラーが発生していない場合
        wait = WebDriverWait(driver, 3)
        position_num = get_position_num(conf, driver)
        wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_ALL_DEAL_BUTTON_PATH))).click()
        #time.sleep(2)
        if position_num == 1:
            time.sleep(2)
            tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
            try:
                deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            except selenium.common.exceptions.NoSuchElementException as e:
                tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            deal_buttonE.click()

        elif position_num > 1:
            tmp_path = conf.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
            try:
                deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            except selenium.common.exceptions.NoSuchElementException as e:
                tmp_path = conf.MODAL_POSITION_ALL_DEAL_CONFIRM_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                # deal_buttonE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            deal_buttonE.click()
        time.sleep(sleep_time)  # 決済に要する時間待つ

#取引開始時間を見て決済時間が来ているならTrueを返す
#新規発注してから約定まで数秒かかるが、それがなかったものとして指定予想時間が経過しているかをチェックする
def do_deal(conf, order_score, base_t_just_dt, position_num):
    """
    trade_start_stamp = int(order_dt.timestamp())
    trade_start_just_stamp = trade_start_stamp - (trade_start_stamp % conf.ORDER_TAKE_SEC)
    """
    trade_start_just_stamp = order_score - (order_score % conf.LOOP_TERM)
    base_t_just_dt_stamp = int(base_t_just_dt.timestamp())

    passed = base_t_just_dt_stamp - trade_start_just_stamp

    trade_ext_start_tmp = conf.TRADE_EXT_START
    if position_num <= conf.TRADE_EXT_START_SHORT_NUM:
        trade_ext_start_tmp = conf.TRADE_EXT_START_SHORT_TERM

    if passed >= trade_ext_start_tmp:
        return True
    else :
        return False

#ストップロスに達しているかを判定
def do_manual_stoploss(conf, start_rate, now_close, position_type, start_text):
    profit = get_profit(start_rate, now_close, position_type)

    if (conf.STOP_LOSS_MANUAL  * -1) >= profit:
        conf.LOGGER("do_manual_stoploss profit:", profit, " start_text:", start_text)
        return True

    return False

def do_trail_stoploss(conf, start_rate, trail_stoploss, now_close, position_type, start_text):
    profit = get_profit(start_rate, now_close, position_type)
    trail_stoploss_flg = False

    if position_type == 0 and now_close <= trail_stoploss:
        trail_stoploss_flg = True

    elif position_type == 2 and now_close >= trail_stoploss:
        trail_stoploss_flg = True

    if trail_stoploss_flg:
        conf.LOGGER("do_trail_stoploss profit:", profit, " start_text:", start_text)
        return True

    return trail_stoploss

#延長判断する
def do_ext(conf, start_text, position_type, sign_ext, base_t_just_dt, probe_up, probe_dw, order_score, position_num):
    do_ext_flg = None #延長判定の結果、延長するばあいにTrue
    ext_flg = None #決済しない場合にTrue

    tmp_dt = datetime.datetime.now()

    """
    start_hour, start_min, start_sec = start_text.split(":")
    if start_hour == "23" and base_t_just_dt.hour == 0:
        tmp_dt = tmp_dt - timedelta(days=1)

    trade_start_datetime = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                             hour=int(start_hour), minute=int(start_min), second=int(start_sec), microsecond=0)
    trade_start_stamp = int(trade_start_datetime.timestamp())
    trade_start_just_stamp = trade_start_stamp - (trade_start_stamp % conf.ORDER_TAKE_SEC)
    """
    trade_start_just_stamp = order_score - (order_score % conf.LOOP_TERM)
    base_t_just_dt_stamp = int(base_t_just_dt.timestamp())

    passed = base_t_just_dt_stamp - trade_start_just_stamp

    trade_ext_term_tmp = conf.TRADE_EXT_TERM
    if position_num <= conf.TRADE_EXT_SHORT_NUM:
        trade_ext_term_tmp = conf.TRADE_EXT_SHORT_TERM

    trade_ext_start_tmp = conf.TRADE_EXT_START
    if position_num <= conf.TRADE_EXT_START_SHORT_NUM:
        trade_ext_start_tmp = conf.TRADE_EXT_START_SHORT_TERM

    if (passed - trade_ext_start_tmp) % trade_ext_term_tmp == 0:
        if position_type == 0 and sign_ext ==0:
            #conf.LOGGER("EXTEND:", start_text, probe_up)
            ext_flg = True
            do_ext_flg = True

        elif position_type == 2 and sign_ext ==2:
            #conf.LOGGER("EXTEND:", start_text, probe_dw)
            ext_flg = True
            do_ext_flg = True

        else:
            ext_flg = False
            do_ext_flg = False
    else:
        ext_flg = True
        do_ext_flg = False

    # トレード指定対象外時間なら延長しない
    if do_ext_flg == True or ext_flg == True:
        for time_range in conf.EXCEPT_DATETIME:
            start_time_range, end_time_range = time_range
            if start_time_range <= tmp_dt and tmp_dt <= end_time_range:
                do_ext_flg = False
                ext_flg = False
                break

    return ext_flg, do_ext_flg

#パスが都度変わる場合に再度エレメントを取得しなおす
def get_again(conf, driver, path):
    tmp_path = path.replace("NUM", str(conf.MODAL_NUM_FIRST))
    try:
        correct_num = conf.MODAL_NUM_FIRST
        element = driver.find_element(By.XPATH, tmp_path)
    except selenium.common.exceptions.NoSuchElementException as e:
        correct_num = conf.MODAL_NUM_SECOND
        tmp_path = path.replace("NUM", str(conf.MODAL_NUM_SECOND))
        element = driver.find_element(By.XPATH, tmp_path)

    return [element,correct_num]

#パスが都度変わる場合に再度クリックする
def click_again(conf, driver, path):
    tmp_path = path.replace("NUM", str(conf.MODAL_NUM_FIRST))
    try:
        correct_num = conf.MODAL_NUM_FIRST
        element = driver.find_element(By.XPATH, tmp_path)
        element.click()
    except selenium.common.exceptions.NoSuchElementException as e:
        correct_num = conf.MODAL_NUM_SECOND
        tmp_path = path.replace("NUM", str(conf.MODAL_NUM_SECOND))
        element = driver.find_element(By.XPATH, tmp_path)
        element.click()

    return correct_num

# ワンクリック注文
def oneclick_order(conf, driver, sign):
    wait = WebDriverWait(driver, 1)

    if sign == 0:  # 買い
        wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_ONECLICK_PATH))).click()
    elif sign == 2:  # 売り
        wait.until(EC.presence_of_element_located((By.XPATH, conf.SELL_ONECLICK_PATH))).click()


# 詳細注文画面から注文の場合準備しておく
def detail_order_pre(conf, driver):
    wait = WebDriverWait(driver, 1)
    # 注文ボタンを押下
    click_order_detail(conf)
    # wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_SELECT_BUTTON_PATH))).click()

    # ポジション数入力
    position_inputE = wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_POSITION_INPUT_PATH)))
    position_inputE.clear()
    position_inputE.send_keys(conf.AMT)

def modal_change(conf ):
    time.sleep(0.2)
    pag.moveTo(x=conf.modal_change_button_x, y=conf.modal_change_button_y)
    pag.click()
    time.sleep(0.2)

def oneclick_order_modal_test(conf, driver):
    wait = WebDriverWait(driver, 2)

    time.sleep(0.2)
    pag.moveTo(x=conf.top_buy_button_x, y=conf.top_buy_button_y)
    pag.click()
    time.sleep(0.2)

    modal_change(conf)
    click_again(conf, driver,conf.ORDER_CANCEL_BUTTON )

    """
    set_amt(conf, driver)
    #買いボタン押下
    wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_ONECLICK_PATH))).click()
    """

def detail_order_modal_test(conf, driver):
    wait = WebDriverWait(driver, 2)
    wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_SELECT_BUTTON_PATH))).click()
    #wait.until(EC.presence_of_element_located((By.XPATH, conf.MODAL_CHOISE))).click()
    modal_change(conf)

    position_inputE, correct_num = get_again(conf, driver, conf.MODAL_ORDER_POSITION_INPUT_PATH)
    position_inputE.clear()
    position_inputE.send_keys("1000")

    try:
        tmp_path = conf.MODAL_ORDER_LIMIT_BUTTON_PATH.replace("NUM", str(correct_num))
        driver.find_element(By.XPATH, tmp_path).click()
    except selenium.common.exceptions.NoSuchElementException as e:
        tmp_path = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[7]/div/div[2]/div/label'.replace("NUM", str(correct_num))
        driver.find_element(By.XPATH, tmp_path).click()

    # 売買ボタン押下
    tmp_path = conf.MODAL_TRADE_BUTTON_PATH.replace("NUM", str(correct_num))
    wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
    # 売買確定ボタン押下
    tmp_path = conf.MODAL_TRADE_BUTTON2_PATH.replace("NUM", str(correct_num))
    wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
    time.sleep(8)


# 詳細注文画面から注文
def detail_order(conf, driver, sign, stoploss):
    wait = WebDriverWait(driver, 2)

    # 注文ボタンを押下
    # click_order_detail()
    # wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_SELECT_BUTTON_PATH))).click()

    go_order = True
    first_num = conf.MODAL_NUM_FIRST
    second_num = conf.MODAL_NUM_SECOND
    if sign == 0:  # 買いを選択
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_SELECT_BUTTON_PATH))).click()
        except selenium.common.exceptions.StaleElementReferenceException as e:
            try:
                wait.until(EC.presence_of_element_located((By.XPATH, conf.BUY_SELECT_BUTTON_PATH))).click()
                #conf.LOGGER("StaleElementReferenceException:conf.BUY_SELECT_BUTTON_PATH")
            except selenium.common.exceptions.StaleElementReferenceException as e:
                conf.LOGGER("PASS BUY ORDER")
                go_order = False

    elif sign == 2:  # 売りを選択
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, conf.SELL_SELECT_BUTTON_PATH))).click()
        except selenium.common.exceptions.StaleElementReferenceException as e:
            try:
                wait.until(EC.presence_of_element_located((By.XPATH, conf.SELL_SELECT_BUTTON_PATH))).click()
                #conf.LOGGER("StaleElementReferenceException:conf.SELL_SELECT_BUTTON_PATH")
            except selenium.common.exceptions.StaleElementReferenceException as e:
                conf.LOGGER("PASS SELL ORDER")
                go_order = False

    #time.sleep(0.1)

    if go_order:
        # 別のモーダル画面からの注文
        # ポジション数入力

        tmp_path = conf.MODAL_ORDER_POSITION_INPUT_PATH.replace("NUM", str(first_num))
        try:
            position_inputE = driver.find_element(By.XPATH, tmp_path)
        except selenium.common.exceptions.NoSuchElementException as e:
            first_num = second_num
            tmp_path = conf.MODAL_ORDER_POSITION_INPUT_PATH.replace("NUM", str(first_num))
            position_inputE = driver.find_element(By.XPATH, tmp_path)

        position_inputE.clear()
        position_inputE.send_keys(conf.AMT)

        # 逆指値指定ボタン押下
        try:
            tmp_path = conf.MODAL_ORDER_LIMIT_BUTTON_PATH.replace("NUM", str(first_num))
            driver.find_element(By.XPATH, tmp_path).click()
            # 逆指値指定
            tmp_path = conf.MODAL_ORDER_LIMIT_INPUT_PATH.replace("NUM", str(first_num))
            limitE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            limitE.clear()
            limitE.send_keys(stoploss)
        except selenium.common.exceptions.NoSuchElementException as e:
            tmp_path = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[7]/div/div[2]/div/label'.replace("NUM",str(first_num))
            driver.find_element(By.XPATH, tmp_path).click()
            # 逆指値指定
            tmp_path = '/html/body/div[NUM]/div/div/div/div/div[2]/div[2]/div[7]/div[2]/div/div[1]/div/input'.replace("NUM", str(first_num))
            limitE = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path)))
            limitE.clear()
            limitE.send_keys(stoploss)

        # 売買ボタン押下
        tmp_path = conf.MODAL_TRADE_BUTTON_PATH.replace("NUM", str(first_num))
        wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()

        # 売買確定ボタン押下
        tmp_path = conf.MODAL_TRADE_BUTTON2_PATH.replace("NUM", str(first_num))
        wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()

        """
        #画面右の画面からの注文
        # ポジション数入力
        position_inputE = wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_POSITION_INPUT_PATH)))
        position_inputE.clear()
        position_inputE.send_keys(conf.AMT)

        # 逆指値指定ボタン押下
        wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_LIMIT_BUTTON_PATH))).click()
        # 逆指値指定
        limitE = wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_LIMIT_INPUT_PATH)))
        limitE.clear()
        if sign == 0:  # 買いを選択
            limitE.send_keys(conf.RATE_FORMAT.format(close - stoploss))
        elif sign == 2:  # 売りを選択
            limitE.send_keys(conf.RATE_FORMAT.format(close + stoploss))

        # 売買ボタン押下
        wait.until(EC.presence_of_element_located((By.XPATH, conf.TRADE_BUTTON_PATH))).click()
        # 売買確定ボタン押下
        wait.until(EC.presence_of_element_located((By.XPATH, conf.TRADE_BUTTON2_PATH))).click()
        """
    else:
        # キャンセルボタン押下
        tmp_path = conf.MODAL_ORDER_SELECT_CLOSE_PATH.replace("NUM", str(first_num))
        try:
            cancelE = driver.find_element(By.XPATH, tmp_path)
        except selenium.common.exceptions.NoSuchElementException as e:
            first_num = second_num
            tmp_path = conf.MODAL_ORDER_SELECT_CLOSE_PATH.replace("NUM", str(first_num))
            cancelE = driver.find_element(By.XPATH, tmp_path)

        cancelE.click()
        """
        #画面右の画面からの注文の場合
        wait.until(EC.presence_of_element_located((By.XPATH, conf.ORDER_SELECT_CLOSE_PATH))).click()
        """


def onemore_click(path, driver):
    try:
        driver.find_element(By.XPATH, path).click()
    except Exception as e:
        try:
            conf.LOGGER("error 1")
            time.sleep(0.01)
            driver.find_element(By.XPATH, path).click()
        except Exception as e:
            try:
                conf.LOGGER("error 2")
                time.sleep(0.01)
                driver.find_element(By.XPATH, path).click()
            except Exception as e:
                conf.LOGGER("onemore_click cannot get path:" + path)
                raise Exception("onemore_click cannot get path:" + path)


# 逆指値を設定する ワンクリック用
def set_stoploss(conf, driver, close):
    do_set_stoploss_flg = False

    stoploss = conf.STOP_LOSS_FIX

    wait = WebDriverWait(driver, 1)
    position_num = get_position_num(conf, driver)

    if position_num == 1:
        try:
            slE = driver.find_element(By.XPATH, conf.POSITION_1_STOPLOSS)
        except Exception as e:
            try:
                # 逆指値指定がない場合なので設定する
                try:
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
                except selenium.common.exceptions.StaleElementReferenceException as e:
                    #conf.LOGGER("StaleElementReferenceException:conf.POSITION_1_TYPE_PATH")
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
                except selenium.common.exceptions.TimeoutException as e:
                    #conf.LOGGER("TimeoutException:conf.POSITION_1_TYPE_PATH")
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text

                if position_type_str == "買い":
                    position_type = 0
                elif position_type_str == "売り":
                    position_type = 2

                driver.find_element(By.XPATH, conf.POSITION_1_STOPLOSS_BUTTON_PATH).click()

                first_num = conf.MODAL_NUM_FIRST
                second_num = conf.MODAL_NUM_SECOND
                tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                try:
                    limitE = driver.find_element(By.XPATH, tmp_path)
                except selenium.common.exceptions.NoSuchElementException as e:
                    first_num = second_num
                    tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                    limitE = driver.find_element(By.XPATH, tmp_path)

                limitE.clear()
                if position_type == 0:  # 買いを選択
                    limitE.send_keys(conf.RATE_FORMAT.format(close - stoploss))
                    # conf.LOGGER(close - stoploss)
                elif position_type == 2:  # 売りを選択
                    limitE.send_keys(conf.RATE_FORMAT.format(close + stoploss))

                # 決定ボタン押下
                tmp_path = conf.MODAL_POSITION_STOPLOSS_ENTER_BUTTON.replace("NUM", str(first_num))
                wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
                do_set_stoploss_flg = True

            except Exception as e2:
                conf.EXCEPT_CNT += 1

    elif position_num > 1:
        # 2ポジションある場合にポジションが展開表示されているか確認
        tr_expand(conf, driver)

        for line_num in range(position_num):
            line_num = line_num + 2
            try:
                tmp_path = conf.POSITION_2_STOPLOSS.replace("NUM", str(line_num))
                slE = driver.find_element(By.XPATH, tmp_path)
            except Exception as e:
                try:
                    # 逆指値指定がない場合なので設定する
                    tmp_path = conf.POSITION_2_STOPLOSS_BUTTON_PATH.replace("NUM", str(line_num))
                    driver.find_element(By.XPATH, tmp_path).click()

                    tmp_path = conf.POSITION_2_TYPE_PATH.replace("NUM", str(line_num))
                    try:
                        position_type_str = wait.until(
                            EC.presence_of_element_located((By.XPATH, tmp_path))).text
                    except selenium.common.exceptions.StaleElementReferenceException as e:
                        #conf.LOGGER("StaleElementReferenceException:conf.POSITION_2_TYPE_PATH")
                        position_type_str = wait.until(
                            EC.presence_of_element_located((By.XPATH, tmp_path))).text
                    except selenium.common.exceptions.TimeoutException as e:
                        #conf.LOGGER("TimeoutException:conf.POSITION_2_TYPE_PATH")
                        position_type_str = wait.until(
                            EC.presence_of_element_located((By.XPATH, tmp_path))).text

                    if position_type_str == "買い":
                        position_type = 0
                    elif position_type_str == "売り":
                        position_type = 2

                    first_num = conf.MODAL_NUM_FIRST
                    second_num = conf.MODAL_NUM_SECOND
                    tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                    try:
                        limitE = driver.find_element(By.XPATH, tmp_path)
                    except selenium.common.exceptions.NoSuchElementException as e:
                        first_num = second_num
                        tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                        limitE = driver.find_element(By.XPATH, tmp_path)

                    limitE.clear()
                    if position_type == 0:  # 買いを選択
                        limitE.send_keys(conf.RATE_FORMAT.format(close - stoploss))
                    elif position_type == 2:  # 売りを選択
                        limitE.send_keys(conf.RATE_FORMAT.format(close + stoploss))

                    # 決定ボタン押下
                    tmp_path = conf.MODAL_POSITION_STOPLOSS_ENTER_BUTTON.replace("NUM", str(first_num))
                    wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
                    do_set_stoploss_flg = True

                except Exception as e2:
                    conf.EXCEPT_CNT +=1

    return do_set_stoploss_flg

# 損切ラインを変更する
def stoploss_update(conf, driver, position_type, close, prev_rate, position_num, line_num):
    wait = WebDriverWait(driver, 1)

    return_rate = prev_rate

    do_change = False
    if position_type == 0:
        if close - prev_rate >= conf.STOPLOSS_UPDATE_PIPS:
            # 指定の利益が上がっていたら損切を変更する
            do_change = True
    elif position_type == 2:
        if prev_rate - close >= conf.STOPLOSS_UPDATE_PIPS:
            # 指定の利益が上がっていたら損切を変更する
            do_change = True

    if do_change:
        stoploss = conf.STOP_LOSS_FIX

        if position_num == 1:
            try:
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
            except selenium.common.exceptions.StaleElementReferenceException as e:
                #conf.LOGGER("StaleElementReferenceException:conf.POSITION_1_TYPE_PATH")
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
            except selenium.common.exceptions.TimeoutException as e:
                #conf.LOGGER("TimeoutException:conf.POSITION_1_TYPE_PATH")
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text

            if position_type_str == "買い":
                position_type = 0
            elif position_type_str == "売り":
                position_type = 2

            driver.find_element(By.XPATH, conf.POSITION_1_STOPLOSS_BUTTON_PATH).click()

            first_num = conf.MODAL_NUM_FIRST
            second_num = conf.MODAL_NUM_SECOND
            tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
            try:
                limitE = driver.find_element(By.XPATH, tmp_path)
            except selenium.common.exceptions.NoSuchElementException as e:
                first_num = second_num
                tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                limitE = driver.find_element(By.XPATH, tmp_path)

            limitE.clear()

            if position_type == 0:  # 買いを選択
                stoploss_rate = get_decimal_sub(close, stoploss)
            elif position_type == 2:  # 売りを選択
                stoploss_rate = get_decimal_add(close, stoploss)

            stoploss_rate = conf.RATE_FORMAT.format(stoploss_rate)
            limitE.send_keys(stoploss_rate)

            # 決定ボタン押下
            tmp_path = conf.MODAL_POSITION_STOMODAL_POSITION_STOPLOSS_ENTER_BUTTONPLOSS_INPUT.replace("NUM", str(first_num))
            wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
        elif position_num > 1:
            # 2ポジションある場合にポジションが展開表示されているか確認
            tr_expand(conf, driver)

            # 逆指値指定がない場合なので設定する
            tmp_path = conf.POSITION_2_STOPLOSS_BUTTON_PATH.replace("NUM", str(line_num))
            driver.find_element(By.XPATH, tmp_path).click()

            tmp_path = conf.POSITION_2_TYPE_PATH.replace("NUM", str(line_num))
            try:
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, tmp_path))).text
            except selenium.common.exceptions.StaleElementReferenceException as e:
                #conf.LOGGER("StaleElementReferenceException:conf.POSITION_2_TYPE_PATH")
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, tmp_path))).text
            except selenium.common.exceptions.TimeoutException as e:
                #conf.LOGGER("TimeoutException:conf.POSITION_2_TYPE_PATH")
                position_type_str = wait.until(
                    EC.presence_of_element_located((By.XPATH, tmp_path))).text

            if position_type_str == "買い":
                position_type = 0
            elif position_type_str == "売り":
                position_type = 2

            first_num = conf.MODAL_NUM_FIRST
            second_num = conf.MODAL_NUM_SECOND
            tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
            try:
                limitE = driver.find_element(By.XPATH, tmp_path)
            except selenium.common.exceptions.NoSuchElementException as e:
                first_num = second_num
                tmp_path = conf.MODAL_POSITION_STOPLOSS_INPUT.replace("NUM", str(first_num))
                limitE = driver.find_element(By.XPATH, tmp_path)
            limitE.clear()
            if position_type == 0:  # 買いを選択
                stoploss_rate = get_decimal_sub(close, stoploss)
            elif position_type == 2:  # 売りを選択
                stoploss_rate = get_decimal_add(close, stoploss)

            stoploss_rate = conf.RATE_FORMAT.format(stoploss_rate)
            limitE.send_keys(stoploss_rate)

            # 決定ボタン押下
            tmp_path = conf.MODAL_POSITION_STOMODAL_POSITION_STOPLOSS_ENTER_BUTTONPLOSS_INPUT.replace("NUM", str(first_num))
            wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()

        conf.LOGGER("prev_rate change", return_rate, close)
        return_rate = close

    return [return_rate, float(stoploss_rate)]


# 2ポジションある場合にポジション表示を展開
def tr_expand(conf, driver):
    tmp_cnt = 0
    while True:
        tmp_cnt += 1
        if tmp_cnt > 5:
            conf.LOGGER("tr element cannot get 5time over")
            raise Exception("tr element cannot get 5time over")
        try:
            trE = driver.find_element(By.XPATH, conf.POSITION_1_TR)
            tmp_text = trE.get_attribute("aria-expanded")
            # きちんと属性表示されるまで取得
            if tmp_text != None:
                break
            time.sleep(0.01)
        except selenium.common.exceptions.StaleElementReferenceException as e:
            time.sleep(0.01)

    if tmp_text == "false":
        # 展開されていない場合は展開する
        onemore_click(conf.POSITION_EXPAND_PATH, driver)


# ポジション欄を取引時間順にする
def tr_sort(conf, driver):
    wait = WebDriverWait(driver, 1)
    # 新規日付順になっているか確認する
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
    except selenium.common.exceptions.StaleElementReferenceException as e:
        # だめならもう一度トライ
        #conf.LOGGER("StaleElementReferenceException:conf.POSITION_DATE_PATH")
        wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
    except selenium.common.exceptions.TimeoutException as e:
        #conf.LOGGER("TimeoutException:conf.POSITION_DATE_PATH")
        wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
    time.sleep(0.1)  # ソートされる時間待つ

    tmp_cnt = 0
    while True:
        tmp_cnt += 1
        if tmp_cnt > 5:
            conf.LOGGER("p_idE element cannot get 5time over")
            raise Exception("p_idE element cannot get 5time over")
        try:
            # p_idE = driver.find_element(By.XPATH, conf.POSITION_DATE_PATH)
            tmp_text2 = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).get_attribute(
                "aria-sort")
            # きちんと属性表示されるまで取得
            if tmp_text2 != None:
                break
            time.sleep(0.01)
        except selenium.common.exceptions.StaleElementReferenceException as e:
            time.sleep(0.01)

    # conf.LOGGER("aria-sort", tmp_text)
    if tmp_text2 != "ascending":
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
        except selenium.common.exceptions.StaleElementReferenceException as e:
            #conf.LOGGER("StaleElementReferenceException2:conf.POSITION_DATE_PATH")
            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
        except selenium.common.exceptions.TimeoutException as e:
            #conf.LOGGER("TimeoutException2:conf.POSITION_DATE_PATH")
            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_DATE_PATH))).click()
        time.sleep(0.1)  # ソートされる時間待つ

# 現在の損益を取得
def get_now_profit_pips(conf, driver, position_num):
    wait = WebDriverWait(driver, 1)
    profit_pips = 0
    now_profit = wait.until(EC.presence_of_element_located((By.XPATH, conf.NOW_PROFIT))).text
    now_profit = int(float(now_profit.split(" ")[1].replace(",", "")))
    if position_num != 0:
        total_amt = int(conf.AMT) * position_num
        profit_pips = now_profit / total_amt

    return profit_pips

def get_now_profit_pips_eurusd(conf, driver, position_num, close_usdjpy):
    wait = WebDriverWait(driver, 1)
    profit_pips = 0
    now_profit = wait.until(EC.presence_of_element_located((By.XPATH, conf.NOW_PROFIT))).text
    now_profit = int(float(now_profit.split(" ")[1].replace(",", "")))
    if position_num != 0:
        total_amt = int(conf.AMT) * position_num
        profit_pips = now_profit / total_amt
        #eurusdの場合はさらにドル円レートで割る必要がある
        profit_pips = profit_pips / close_usdjpy

    return profit_pips


def deal_all(redis_db, conf, driver, position_num, start_text_stoploss_order_score_list, ordered_dict, now_close, spread, start_text_org, base_t_just_dt):
    # 一括決済の場合
    regist_score = float(datetime.datetime.now().timestamp())

    do_deal_all(conf, driver, catch_except=False, sleep_time=2.5)

    for line_num in reversed(range(position_num)):
        # 事前に取得したstart_text, stoploss, order_score
        start_text, stoploss, order_score = start_text_stoploss_order_score_list[line_num]
        if order_score == None:
            # 既に決済ずみなので何もしない
            pass
        else:
            start_hour, start_min, start_sec = start_text.split(":")
            tmp_dt = datetime.datetime.now()
            if start_hour == "23" and tmp_dt.hour == 0:
                tmp_dt = tmp_dt - timedelta(days=1)

            trade_start_datetime = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month,
                                                     day=tmp_dt.day,
                                                     hour=int(start_hour),
                                                     minute=int(start_min),
                                                     second=int(start_sec), microsecond=0)
            position_start_time_dt_timestamp = int(trade_start_datetime.timestamp())
            order_dt = datetime.datetime.fromtimestamp(order_score)
            order_dict = ordered_dict[order_score]
            order_dict["position_score"] = position_start_time_dt_timestamp
            order_dict["stoploss"] = stoploss

            order_dict["deal_score"] = regist_score
            order_dict["end_rate"] = now_close
            order_dict["amt"] = int(conf.AMT)
            order_dict["close_spread"] = spread

            redis_db.zadd(conf.DB_ORDER_KEY, json.dumps(order_dict), regist_score)
            # 決済したので管理から外す
            del ordered_dict[order_score]

            position_take = get_decimal_sub(position_start_time_dt_timestamp, order_score)

            if start_text != start_text_org:
                #全決済のもととなったポジション以外の決済ログを出力
                #もととなったポジションのログは出力済みのはず
                conf.LOGGER("ALL DEAL:", order_dt, start_text, base_t_just_dt, position_take, spread)

    return ordered_dict

# 決済
def deal(conf, driver, prev_deal_dict, base_t_just_dt, prev_deal_start_time, sign_ext, spread,
                     now_close, prev_rate_dict, start_rate_dict, probe_up ,probe_dw, ordered_dict, exception_cnt):
    # if len(prev_rate_dict.keys()) != 0:
    #    conf.LOGGER("prev_rate_dict:",prev_rate_dict)
    wait = WebDriverWait(driver, 1)

    position_times = []  # 現在保有中のポジションの開始時間

    deal_take1 = 0.0
    deal_take2 = 0.0
    deal_take3 = 0.0
    deal_take4 = 0.0
    deal_take5 = 0.0

    deal_flg = False
    do_ext_flg = False

    position_num = get_position_num(conf, driver)
    # 前回決済した時間から一定時間が過ぎていたら決済
    # LOOP_TERMごとに決済してしまうと、例えば2秒ごとに決済で、取引間隔が4秒なら、決済した取引が画面上から消されるまで時間がかかるため、残ってしまい次に古い取引が
    # 規定時間が過ぎていないのに前の取引の取引開始時間を取得してしまうことによって予期せぬ決済をしてしまうため
    # if prev_deal_datetime == None or ( prev_deal_datetime != None and prev_deal_datetime + timedelta(seconds=conf.DEAL_TAKE_SEC) <= base_t_just_dt):

    all_dealed_flg = False

    try:
        if position_num == 1:

            start_text =get_start_text(conf, driver, line_num=None)

            dealed_flg = False #決済済みフラグ
            stoploss = get_stoploss(conf, driver, line_num=None)
            #deal_key = start_text + "_" + str(stoploss)
            deal_key = start_text
            if deal_key in prev_deal_dict.keys():
                #既に決済済みと確定
                dealed_flg = True

            if dealed_flg == True:
                if prev_deal_dict[deal_key] + timedelta(seconds=conf.DEAL_TAKE_SEC) <= base_t_just_dt:
                    conf.LOGGER("same start_text!!", deal_key, prev_deal_dict[deal_key])
                    raise Exception("same start_text!!")

                position_times.append(start_text)
            else:
                start_hour, start_min, start_sec = start_text.split(":")
                tmp_dt = datetime.datetime.now()
                if start_hour == "23" and tmp_dt.hour == 0:
                    tmp_dt = tmp_dt - timedelta(days=1)

                trade_start_datetime = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                                         hour=int(start_hour), minute=int(start_min), second=int(start_sec),
                                                         microsecond=0)
                position_start_time_dt_timestamp = int(trade_start_datetime.timestamp())

                #発注した日時の古い順に並び替え
                try:
                    order_sorted = sorted(ordered_dict.items(), key=lambda x: x[0],reverse=False)
                    order_score, order_dict = order_sorted[0] #一番古い発注がポジション一覧にあるもの
                    order_dt = datetime.datetime.fromtimestamp(order_score)
                    order_start_rate = order_dict["start_rate"]
                    order_prev_rate = order_dict["prev_rate"]
                    order_stoploss = order_dict["stoploss"]
                    order_type = order_dict["sign"]
                    order_dict["position_score"] = position_start_time_dt_timestamp
                    order_start_rate_tm = order_dict.get("start_rate_tm")
                    if order_start_rate_tm == None:
                        order_start_rate_tm = get_new_rate(conf, driver, line_num=None)
                        ordered_dict[order_score]["start_rate_tm"] = order_start_rate_tm

                    order_prev_profit = order_dict["prev_profit"]
                    order_trail_stoploss = order_dict["trail_stoploss"]

                    #発注してから約定するまでの時間がながければ古い発注情報が残っていた可能性があるので、一旦エラーにして全決済させる
                    passed_sec = position_start_time_dt_timestamp - order_score
                    if passed_sec >= 60:
                        conf.LOGGER("order info too old ", order_dt, trade_start_datetime)
                        raise Exception("order info too old")
                except Exception as e:
                    #何かしら取得できない場合はエラー
                    conf.LOGGER("cannot get order info ", tracebackPrint(e))
                    conf.LOGGER("ordered_dict:", ordered_dict)
                    raise Exception("cannot get order info")

                order_dict["stoploss"] = stoploss
                """
                position_start_time_dt_timestamp_just = position_start_time_dt_timestamp - (
                            position_start_time_dt_timestamp % conf.LOOP_TERM)
                try:
                    prev_rate = prev_rate_dict[position_start_time_dt_timestamp_just]
                    start_rate = start_rate_dict[position_start_time_dt_timestamp_just]
                except Exception as e:
                    # 取得できなかったときは、発注してからループ秒以上経過して発注完了しているので損切アップをおこなえない 6秒ループなら発注してから7秒以上経過して発注完了している場合など
                    # 探索範囲を広げる
                    position_start_time_dt_timestamp_just2 = position_start_time_dt_timestamp - (
                            position_start_time_dt_timestamp % conf.LOOP_TERM) - conf.LOOP_TERM
                    try:
                        prev_rate = prev_rate_dict[position_start_time_dt_timestamp_just2]
                        start_rate = start_rate_dict[position_start_time_dt_timestamp_just2]
                    except Exception as e:
                        # 探索範囲を広げても取得できない場合
                        #conf.LOGGER("cannot get prev_rate or start_rate:", start_text,position_start_time_dt_timestamp_just, position_start_time_dt_timestamp_just2,prev_rate_dict, start_rate_dict)
                        prev_rate = None
                        start_rate = None
                        prev_rate_dict[position_start_time_dt_timestamp_just] = now_close  # 現在のレートを入れて次回損切アップできるようにする
                        start_rate_dict[position_start_time_dt_timestamp_just] = now_close
                """
                try:
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
                except selenium.common.exceptions.StaleElementReferenceException as e:
                    #conf.LOGGER("StaleElementReferenceException:conf.POSITION_1_TYPE_PATH")
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
                except selenium.common.exceptions.TimeoutException as e:
                    #conf.LOGGER("TimeoutException:conf.POSITION_1_TYPE_PATH")
                    position_type_str = wait.until(
                        EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text

                if position_type_str == "買い":
                    position_type = 0
                elif position_type_str == "売り":
                    position_type = 2
                else:
                    # 予期せぬ値なので再度トライ
                    position_type_str = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text
                    if position_type_str == "買い":
                        position_type = 0
                    elif position_type_str == "売り":
                        position_type = 2
                    else:
                        conf.LOGGER("position_type_str not invalid:", position_type_str)
                        raise Exception("position_type_str not invalid")

                deal_flg = do_deal(conf, order_score, base_t_just_dt, position_num)
                manual_stoploss_flg = do_manual_stoploss(conf, order_start_rate_tm, now_close, position_type, start_text)

                now_profit = get_profit(order_start_rate_tm, now_close, position_type)

                if conf.TRAIL_STOPLOSS and manual_stoploss_flg == False:
                    # トレイル判断
                    trail_stoploss_flg = do_trail_stoploss(conf, order_start_rate_tm, order_trail_stoploss, now_close, position_type, start_text)
                else:
                    trail_stoploss_flg = False

                if deal_flg or manual_stoploss_flg or trail_stoploss_flg:
                    deal_flg = False
                    ext_flg = False
                    if manual_stoploss_flg == False and trail_stoploss_flg == False:
                        if conf.EXT_FLG:
                            ext_flg, do_ext_flg = do_ext(conf, start_text, position_type, sign_ext, base_t_just_dt, probe_up, probe_dw, order_score, position_num)

                    if ext_flg == False or manual_stoploss_flg or trail_stoploss_flg:
                        #prev_deal_dictに登録　すでに値があったら偶然同じstart_text, stoplossのものがあったということなので一旦全決済
                        if deal_key in prev_deal_dict.keys():
                            conf.LOGGER("same dealkey exists ", deal_key)
                            raise Exception("same dealkey exists")
                        else:
                            prev_deal_dict[deal_key] = base_t_just_dt
                        deal_flg = True

                        try:
                            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_DEAL_PATH))).click()
                        except selenium.common.exceptions.StaleElementReferenceException as e:
                            #conf.LOGGER("StaleElementReferenceException:conf.POSITION_1_DEAL_PATH")
                            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_DEAL_PATH))).click()
                        except selenium.common.exceptions.TimeoutException as e:
                            #conf.LOGGER("TimeoutException:conf.POSITION_1_DEAL_PATH")
                            wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_DEAL_PATH))).click()

                        tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_FIRST))
                        try:
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                        except selenium.common.exceptions.NoSuchElementException as e:
                            tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM", str(conf.MODAL_NUM_SECOND))
                            deal_buttonE = driver.find_element(By.XPATH, tmp_path)

                        try:
                            position_take = get_decimal_sub(position_start_time_dt_timestamp, order_score)

                            if manual_stoploss_flg:
                                conf.LOGGER("MANUAL STOPLOSS:", order_dt, start_text, base_t_just_dt, position_take, spread)
                            elif trail_stoploss_flg:
                                conf.LOGGER("TRAIL STOPLOSS:", order_dt, start_text, base_t_just_dt, position_take, spread)
                            else:
                                conf.LOGGER("DEAL:", order_dt, start_text, base_t_just_dt, position_take, spread)
                            conf.DEAL_CNT += 1

                            #発注履歴DB登録
                            regist_score = float(datetime.datetime.now().timestamp())
                            order_dict["deal_score"] = regist_score
                            order_dict["end_rate"] = now_close
                            order_dict["amt"] = int(conf.AMT)
                            order_dict["close_spread"] = spread

                            deal_buttonE.click()

                            redis_db.zadd(conf.DB_ORDER_KEY, json.dumps(order_dict), regist_score)
                            #決済したので管理から外す
                            del ordered_dict[order_score]

                        except selenium.common.exceptions.ElementClickInterceptedException as e:
                            conf.LOGGER("deal retry")
                            conf.LOGGER(tracebackPrint(e))
                            time.sleep(2)
                            #ポップアップメッセージのせいだと思われるので、2秒待って再チャレンジ
                            try:
                                deal_buttonE.click()
                                redis_db.zadd(conf.DB_ORDER_KEY, json.dumps(order_dict), regist_score)
                                # 決済したので管理から外す
                                del ordered_dict[order_score]
                            except selenium.common.exceptions.ElementClickInterceptedException as e2:
                                raise Exception("deal cannot click!!")

                    else:
                        # 損切ライン変更
                        if conf.STOPLOSS_UPDATE_PIPS != None:
                            return_prev_rate, return_stoploss = stoploss_update(conf, driver, position_type, now_close, order_prev_rate, position_num,line_num=None)
                            ordered_dict[order_score]["prev_rate"] = return_prev_rate
                            ordered_dict[order_score]["stoploss"] = return_stoploss

                        # トレイルストップロス変更
                        if conf.TRAIL_STOPLOSS:
                            profit_sub = get_decimal_sub(now_profit, order_prev_profit)
                            if profit_sub > 0:
                                ordered_dict[order_score]["prev_profit"] = now_profit
                                if position_type == 0:
                                    #利益があがった分、トレイルストップレートもあげる
                                    ordered_dict[order_score]["trail_stoploss"] = get_decimal_add(order_trail_stoploss, profit_sub)
                                else:
                                    ordered_dict[order_score]["trail_stoploss"] = get_decimal_sub(order_trail_stoploss,profit_sub)

                        position_times.append(start_text)
                else:
                    # 損切ライン変更
                    if conf.STOPLOSS_UPDATE_PIPS != None:
                        return_prev_rate, return_stoploss = stoploss_update(conf, driver, position_type, now_close,order_prev_rate, position_num, line_num=None)
                        ordered_dict[order_score]["prev_rate"] = return_prev_rate
                        ordered_dict[order_score]["stoploss"] = return_stoploss

                    # トレイルストップロス変更
                    if conf.TRAIL_STOPLOSS:
                        profit_sub = get_decimal_sub(now_profit, order_prev_profit)
                        if profit_sub > 0:
                            ordered_dict[order_score]["prev_profit"] = now_profit
                            if position_type == 0:
                                #利益があがった分、トレイルストップレートもあげる
                                ordered_dict[order_score]["trail_stoploss"] = get_decimal_add(order_trail_stoploss, profit_sub)
                            else:
                                ordered_dict[order_score]["trail_stoploss"] = get_decimal_sub(order_trail_stoploss,profit_sub)

                    position_times.append(start_text)

        elif position_num >= 2:
            start_take1 = time.perf_counter()
            # 2ポジションある場合にポジションが展開表示されているか確認
            tr_expand(conf, driver)

            deal_take1 = time.perf_counter() - start_take1

            start_take2 = time.perf_counter()
            # ポジション欄を取引時間順にする
            #tr_sort(conf, driver)

            #先にstart_textと発注スコアのみ取得
            order_sorted = sorted(ordered_dict.items(), key=lambda x: x[0], reverse=False) #発注した日時の古い順に並び替え

            start_text_stoploss_list = []

            for line_num in range(position_num):
                start_text = get_start_text(conf, driver, line_num=line_num+2)
                stoploss = get_stoploss(conf, driver, line_num=line_num+2)
                start_text_stoploss_list.append([start_text, stoploss])

            try:
                start_text_stoploss_order_score_list = []
                tmp_cnt = 0  # 取得した発注情報の数
                for line_num, start_text_stoploss in enumerate(start_text_stoploss_list):
                    start_text, stoploss = start_text_stoploss
                    #deal_key_tmp = start_text + "_" + str(stoploss)
                    deal_key_tmp = start_text

                    if deal_key_tmp in prev_deal_dict.keys():
                        # 既に決済済みと確定 リストにNoneとして加える
                        start_text_stoploss_order_score_list.append([start_text, stoploss, None])
                        continue

                    order_score, order_dict = order_sorted[tmp_cnt]
                    start_text_stoploss_order_score_list.append([start_text, stoploss, order_score])
                    tmp_cnt += 1

            except Exception as e:
                # 何かしら取得できない場合はエラー
                conf.LOGGER("cannot get order info 2 ", tracebackPrint(e))
                conf.LOGGER("start_text_stoploss_list:", start_text_stoploss_list)
                conf.LOGGER("ordered_dict:", ordered_dict)
                conf.LOGGER("start_text_stoploss_order_score_list:", start_text_stoploss_order_score_list)
                raise Exception("cannot get order info")

            deal_take2 = time.perf_counter() - start_take2

            #reversedにして行の下から処理する 上の行から処理すると決済したら行が減ってしまうため
            for line_num in reversed(range(position_num)):
                start_take3 = time.perf_counter()

                #事前に取得したstart_text, stoploss, order_score
                start_text, stoploss, order_score = start_text_stoploss_order_score_list[line_num]
                #deal_key = start_text + "_" + str(stoploss)
                deal_key = start_text

                line_num = line_num + 2

                deal_take3 = time.perf_counter() - start_take3

                if order_score == None:

                    if deal_key in prev_deal_dict.keys() and prev_deal_dict[deal_key] + timedelta(seconds=conf.DEAL_TAKE_SEC) <= base_t_just_dt:
                        conf.LOGGER("same start_text!!", deal_key, prev_deal_dict[deal_key])
                        raise Exception("same start_text!!")
                    position_times.append(start_text)
                else:
                    start_hour, start_min, start_sec = start_text.split(":")
                    tmp_dt = datetime.datetime.now()
                    if start_hour == "23" and tmp_dt.hour == 0:
                        tmp_dt = tmp_dt - timedelta(days=1)

                    trade_start_datetime = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                                             hour=int(start_hour), minute=int(start_min),
                                                             second=int(start_sec), microsecond=0)
                    position_start_time_dt_timestamp = int(trade_start_datetime.timestamp())
                    try:
                        order_dict = ordered_dict[order_score]
                        order_dt = datetime.datetime.fromtimestamp(order_score)
                        order_start_rate = order_dict["start_rate"]
                        order_prev_rate = order_dict["prev_rate"]
                        order_stoploss = order_dict["stoploss"]
                        order_type = order_dict["sign"]
                        order_dict["position_score"] = position_start_time_dt_timestamp
                        order_dict["stoploss"] = stoploss

                        order_start_rate_tm = order_dict.get("start_rate_tm")
                        if order_start_rate_tm == None:
                            order_start_rate_tm = get_new_rate(conf, driver, line_num=line_num)
                            ordered_dict[order_score]["start_rate_tm"] = order_start_rate_tm

                        order_prev_profit = order_dict["prev_profit"]
                        order_trail_stoploss = order_dict["trail_stoploss"]

                        # 発注してから約定するまでの時間がながければ古い発注情報が残っていた可能性があるので、一旦エラーにして全決済させる
                        passed_sec = position_start_time_dt_timestamp - order_score
                        if passed_sec >= 60:
                            conf.LOGGER("order info too old ", order_dt, trade_start_datetime)
                            raise Exception("order info too old")
                    except Exception as e:
                        # 何かしら取得できない場合はエラー
                        conf.LOGGER("cannot get order info 3 ", tracebackPrint(e))
                        conf.LOGGER("ordered_dict:", ordered_dict)
                        conf.LOGGER("order_score:", order_score)
                        conf.LOGGER("start_text_stoploss_order_score_list:",start_text_stoploss_order_score_list)
                        raise Exception("cannot get order info")

                    tmp_path = conf.POSITION_2_TYPE_PATH.replace("NUM", str(line_num))
                    try:
                        position_type_str = wait.until(
                            EC.presence_of_element_located((By.XPATH, tmp_path))).text
                    except selenium.common.exceptions.StaleElementReferenceException as e:
                        position_type_str = wait.until(
                            EC.presence_of_element_located((By.XPATH, tmp_path))).text
                    except selenium.common.exceptions.TimeoutException as e:
                        position_type_str = wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_TYPE_PATH))).text

                    if position_type_str == "買い":
                        position_type = 0
                    elif position_type_str == "売り":
                        position_type = 2
                    else:
                        # 予期せぬ値なので再度トライ
                        position_type_str = wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).text
                        if position_type_str == "買い":
                            position_type = 0
                        elif position_type_str == "売り":
                            position_type = 2
                        else:
                            conf.LOGGER("position_type_str not invalid:", position_type_str)
                            raise Exception("position_type_str not invalid")

                    deal_flg = do_deal(conf, order_score, base_t_just_dt, position_num)
                    manual_stoploss_flg = do_manual_stoploss(conf, order_start_rate_tm, now_close, position_type,start_text)

                    now_profit = get_profit(order_start_rate_tm, now_close, position_type)

                    if conf.TRAIL_STOPLOSS and manual_stoploss_flg == False:
                        # トレイル判断
                        trail_stoploss_flg = do_trail_stoploss(conf, order_start_rate_tm, order_trail_stoploss,now_close, position_type, start_text)
                    else:
                        trail_stoploss_flg = False

                    if deal_flg or manual_stoploss_flg or trail_stoploss_flg:
                        deal_flg = False
                        ext_flg = False

                        if manual_stoploss_flg == False and trail_stoploss_flg == False:
                            if conf.EXT_FLG:
                                ext_flg, do_ext_flg_tmp = do_ext(conf, start_text, position_type, sign_ext, base_t_just_dt, probe_up, probe_dw, order_score, position_num)
                                if do_ext_flg_tmp:
                                    # do_ext_flg_tmpを直接do_ext_flgに代入してしまうと、
                                    # ループの中なので1番目が延長でも2番目が延長しなければFalseになる可能性があるためdo_ext_flg_tmppを用いる
                                    do_ext_flg = True

                        if ext_flg == False or manual_stoploss_flg or trail_stoploss_flg:
                            if deal_key in prev_deal_dict.keys():
                                conf.LOGGER("same dealkey exists ", deal_key)
                                raise Exception("same dealkey exists")
                            else:
                                prev_deal_dict[deal_key] = base_t_just_dt
                            deal_flg = True

                            start_take5 = time.perf_counter()
                            try:
                                position_take = get_decimal_sub(position_start_time_dt_timestamp, order_score)

                                if manual_stoploss_flg:
                                    conf.LOGGER("MANUAL STOPLOSS:", order_dt, start_text, base_t_just_dt, position_take,
                                                spread)
                                elif trail_stoploss_flg:
                                    conf.LOGGER("TRAIL STOPLOSS:", order_dt, start_text, base_t_just_dt, position_take,
                                                spread)
                                else:
                                    conf.LOGGER("DEAL:", order_dt, start_text, base_t_just_dt, position_take, spread)
                                conf.DEAL_CNT += 1

                                if position_take < 0:
                                    conf.LOGGER("position_take is minus",position_take)
                                    raise Exception("position_take is minus")

                                # 決済確定ボタン押下
                                if conf.DEAL_TYPE == 'ALL':
                                    #一括決済の場合
                                    ordered_dict = deal_all(redis_db, conf, driver, position_num, start_text_stoploss_order_score_list,
                                                            ordered_dict, now_close, spread, start_text, base_t_just_dt)
                                    position_times = []
                                    all_dealed_flg = True
                                    break

                                elif conf.DEAL_TYPE == 'ONE':
                                    # 発注履歴DB登録
                                    regist_score = float(datetime.datetime.now().timestamp())
                                    order_dict["deal_score"] = regist_score
                                    order_dict["end_rate"] = now_close
                                    order_dict["amt"] = int(conf.AMT)
                                    order_dict["close_spread"] = spread

                                    start_take4 = time.perf_counter()
                                    # 決済ボタン押下
                                    tmp_path = conf.POSITION_2_DEAL_PATH.replace("NUM", str(line_num))
                                    try:
                                        wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
                                    except selenium.common.exceptions.StaleElementReferenceException as e:
                                        # conf.LOGGER("StaleElementReferenceException:conf.POSITION_2_DEAL_PATH")
                                        wait.until(EC.presence_of_element_located((By.XPATH, tmp_path))).click()
                                    except selenium.common.exceptions.TimeoutException as e:
                                        # 二つの建玉を同時に決済しようとしている可能性が高いので一件しかないとして処理する
                                        wait.until(EC.presence_of_element_located((By.XPATH, conf.POSITION_1_DEAL_PATH))).click()

                                    deal_take4 = time.perf_counter() - start_take4

                                    tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM",str(conf.MODAL_NUM_FIRST))
                                    try:
                                        deal_buttonE = driver.find_element(By.XPATH, tmp_path)
                                    except selenium.common.exceptions.NoSuchElementException as e:
                                        tmp_path = conf.MODAL_POSITION_DEAL_BUTTON_PATH.replace("NUM",str(conf.MODAL_NUM_SECOND))
                                        deal_buttonE = driver.find_element(By.XPATH, tmp_path)

                                    deal_buttonE.click()

                                    redis_db.zadd(conf.DB_ORDER_KEY, json.dumps(order_dict), regist_score)
                                    # 決済したので管理から外す
                                    del ordered_dict[order_score]

                                if conf.DEAL_TYPE == 'ONE':
                                    # このターンでもう一度決済するかもしれないので少しスリープしてタイミングをずらす
                                    # スリープしないとモーダル画面が残ってしまい、うまく次のポジションの処理ができない
                                    time.sleep(0.15)

                            except selenium.common.exceptions.ElementClickInterceptedException as e:
                                conf.LOGGER("deal retry")
                                conf.LOGGER(tracebackPrint(e))
                                time.sleep(2)
                                # ポップアップメッセージのせいだと思われるので、2秒待って再チャレンジ
                                try:
                                    if conf.DEAL_TYPE == 'ALL':
                                        # 一括決済の場合
                                        ordered_dict = deal_all(redis_db, conf, driver, position_num,
                                                                start_text_stoploss_order_score_list,
                                                                ordered_dict, now_close, spread, start_text,
                                                                base_t_just_dt)
                                        position_times = []
                                        all_dealed_flg = True
                                        break

                                    elif conf.DEAL_TYPE == 'ONE':
                                        deal_buttonE.click()

                                        redis_db.zadd(conf.DB_ORDER_KEY, json.dumps(order_dict), regist_score)
                                        # 決済したので管理から外す
                                        del ordered_dict[order_score]

                                except selenium.common.exceptions.ElementClickInterceptedException as e2:
                                    raise Exception("deal cannot click!!")

                            deal_take5 = time.perf_counter() - start_take5

                        else:
                            # 損切ライン変更
                            if conf.STOPLOSS_UPDATE_PIPS != None:
                                return_prev_rate, return_stoploss = stoploss_update(conf, driver, position_type, now_close,order_prev_rate, position_num,line_num=None)
                                ordered_dict[order_score]["prev_rate"] = return_prev_rate
                                ordered_dict[order_score]["stoploss"] = return_stoploss

                            # トレイルストップロス変更
                            if conf.TRAIL_STOPLOSS:
                                profit_sub = get_decimal_sub(now_profit, order_prev_profit)
                                if profit_sub > 0:
                                    ordered_dict[order_score]["prev_profit"] = now_profit
                                    if position_type == 0:
                                        # 利益があがった分、トレイルストップレートもあげる
                                        ordered_dict[order_score]["trail_stoploss"] = get_decimal_add(
                                            order_trail_stoploss, profit_sub)
                                    else:
                                        ordered_dict[order_score]["trail_stoploss"] = get_decimal_sub(
                                            order_trail_stoploss, profit_sub)

                            position_times.append(start_text)

                    else:
                        # 損切ライン変更
                        if conf.STOPLOSS_UPDATE_PIPS != None:
                            return_prev_rate, return_stoploss = stoploss_update(conf, driver, position_type, now_close,order_prev_rate, position_num,line_num=None)
                            ordered_dict[order_score]["prev_rate"] = return_prev_rate
                            ordered_dict[order_score]["stoploss"] = return_stoploss

                        # トレイルストップロス変更
                        if conf.TRAIL_STOPLOSS:
                            profit_sub = get_decimal_sub(now_profit, order_prev_profit)
                            if profit_sub > 0:
                                ordered_dict[order_score]["prev_profit"] = now_profit
                                if position_type == 0:
                                    #利益があがった分、トレイルストップレートもあげる
                                    ordered_dict[order_score]["trail_stoploss"] = get_decimal_add(order_trail_stoploss, profit_sub)
                                else:
                                    ordered_dict[order_score]["trail_stoploss"] = get_decimal_sub(order_trail_stoploss,profit_sub)

                        position_times.append(start_text)

    except selenium.common.exceptions.StaleElementReferenceException as e:
        conf.LOGGER("deal exception:", e.__str__())
        conf.LOGGER("exception_cnt:", exception_cnt)
        if exception_cnt == 0:
            exception_cnt += 1
            #一回目のエラーなら再度deal処理させる
            return [deal_flg, [deal_take1, deal_take2, deal_take3, deal_take4, deal_take5], prev_deal_start_time,
                    do_ext_flg, position_times, prev_rate_dict, prev_deal_dict, start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg]
        else:
            #二回目のエラーなら例外をなげてmain_loopから抜ける
            raise e

    except selenium.common.exceptions.NoSuchElementException as e:
        conf.LOGGER("deal exception:", e.__str__())
        conf.LOGGER("exception_cnt:", exception_cnt)
        if exception_cnt == 0:
            exception_cnt += 1
            #一回目のエラーなら再度deal処理させる
            return [deal_flg, [deal_take1, deal_take2, deal_take3, deal_take4, deal_take5], prev_deal_start_time,
                    do_ext_flg, position_times, prev_rate_dict, prev_deal_dict, start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg]
        else:
            #二回目のエラーなら例外をなげてmain_loopから抜ける
            raise e

    except Exception as e:
        if e.__str__() == "cannot get start_text":
            conf.LOGGER("deal exception:", e.__str__())
            conf.LOGGER("exception_cnt:", exception_cnt)
            if exception_cnt == 0:
                exception_cnt += 1
                # 一回目のエラーなら再度deal処理させる
                return [deal_flg, [deal_take1, deal_take2, deal_take3, deal_take4, deal_take5], prev_deal_start_time,
                        do_ext_flg, position_times, prev_rate_dict, prev_deal_dict, start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg]
            else:
                # 二回目のエラーなら例外をなげてmain_loopから抜ける
                raise e
        else:
            raise e

    exception_cnt = 0
    return [deal_flg, [deal_take1, deal_take2, deal_take3, deal_take4, deal_take5], prev_deal_start_time, do_ext_flg,
            position_times, prev_rate_dict, prev_deal_dict, start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg]


def main_loop(conf, driver, second_flg=False):
    max_trade_cnt = False
    return_code = 1
    err_flg = 0
    wait = WebDriverWait(driver, 1)
    exception_cnt = 0

    try:
        #取引履歴DB登録
        start_regist_history = time.perf_counter()
        conf.LOGGER("start regist history")
        regist_history_db(conf, driver, redis_db)
        conf.LOGGER("finish regist history. take:", time.perf_counter() - start_regist_history)

        if conf.REGIST_HISTORY_ONLY:
            return_code = 5
            return return_code

        tmp_dt = datetime.datetime.now()
        conf.LOOP_END_DATETIME = tmp_dt + timedelta(minutes=conf.SWITCH_TERM)
        # 予想時間分早く終わらせて、終了時に結果がでるまで待たせる
        conf.LOOP_END_DATETIME_PRE = conf.LOOP_END_DATETIME - timedelta(
            seconds=(conf.TRADE_EXT_TERM + conf.PRED_TERM_ADJUST))

        conf.LOGGER("LOOP_END_DATETIME:", conf.LOOP_END_DATETIME)
        conf.LOGGER("MONEY:", get_money(conf, driver))

        # logout(conf, driver)
        if second_flg == False:
            """
            # 対象ペアを選択
            if conf.PAIR == "USDJPY":
                try:
                    driver.find_element(By.XPATH, conf.EURUSD_PATH).click()
                    time.sleep(2)
                    # 削除対象のペア
                    driver.find_element(By.XPATH, conf.EURUSD_DELETE_PATH).click()
                    time.sleep(2)
                    deleteE = driver.find_element(By.XPATH, conf.PAIR_DELETE_PATH).click()
                    time.sleep(2)
                except Exception as e:
                    pass
                driver.find_element(By.XPATH, conf.USDJPY_PATH).click()

            elif conf.PAIR == "EURUSD":
                try:
                    driver.find_element(By.XPATH, conf.USDJPY_PATH).click()
                    time.sleep(2)
                    # 削除対象のペア
                    driver.find_element(By.XPATH, conf.USDJPY_DELETE_PATH).click()
                    time.sleep(2)
                    deleteE = driver.find_element(By.XPATH, conf.PAIR_DELETE_PATH).click()
                    time.sleep(2)
                except Exception as e:
                    pass
                driver.find_element(By.XPATH, conf.EURUSD_PATH).click()
            """
            time.sleep(2)

            # チャートを削除するためUSDJPYのチャートだけ表示する
            #driver.find_element(By.XPATH,'//*[@id="WatchList"]/cq-context/div[4]/div[1]/cq-context-wrapper[1]/cq-context/div[1]/div[3]/span').click()
            time.sleep(3)

        #delete_chart(driver)
        time.sleep(5)

        if conf.PAIR == "EURUSD":
            if conf.ORDER_TYPE == "ONECLICK":
                #eurusdの場合のみ発注ボタン押しても反応しないことがあるので一度発注しておく
                eurusd_init(conf, driver)
                time.sleep(3)

        if chk_pair(conf, driver) == False:
            err_flg = True

        # 既にポジションを持っていたら前回決済できていないのでエラーとする
        tmp_p_num = get_position_num(conf, driver)
        if tmp_p_num != 0:
            conf.LOGGER("POSITION NUM IS NOT 0!!", tmp_p_num)

            # 決済する
            do_deal_all(conf, driver, catch_except=False, sleep_time=5)

        if conf.ORDER_TYPE == "ONECLICK":
            # ワンクリックでのポジション数を入力
            set_amt(conf, driver)
            oneclick_positionE = driver.find_element(By.XPATH, conf.ONECLICK_POSITION_INPUT_PATH)

            oneclick_position_value = oneclick_positionE.get_attribute("value")
            if oneclick_position_value != conf.AMT_STR:
                conf.LOGGER("oneclick_position incorrect!!", oneclick_position_value)
                err_flg = True

            #ワンクリックが有効か確認
            oneclick_buttonE = driver.find_element(By.XPATH, conf.SELECT_ONECLICK_TOGGLE)
            cl = oneclick_buttonE.get_attribute('class')
            if ('WtrToggle_active__1kW2l' in cl) == False:
                conf.LOGGER("oneclick toggle not set", cl)
                err_flg = True
        else:
            #ワンクリックが有効か確認
            oneclick_buttonE = driver.find_element(By.XPATH, conf.SELECT_ONECLICK_TOGGLE)
            cl = oneclick_buttonE.get_attribute('class')
            if ('WtrToggle_active__1kW2l' in cl) == True:
                conf.LOGGER("oneclick toggle set", cl)
                err_flg = True

        # conf.LOOP_TERM * 5 秒後に開始
        tmp_dt = datetime.datetime.now()
        base_dt = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                    hour=tmp_dt.hour, minute=tmp_dt.minute, second=tmp_dt.second, microsecond=0)

        # conf.LOOP_TERM
        base_dt = base_dt + timedelta(seconds=(conf.LOOP_TERM * 5 + (conf.LOOP_TERM - tmp_dt.second % conf.LOOP_TERM)))
        if base_dt.second == 0:
            #0秒ではじめてしまうと初回のループで他DBから分足の情報を取得する際にまだ登録されていない可能性があるのでずらす
            base_dt = base_dt + timedelta(seconds=(conf.LOOP_TERM))

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

        prev_order_time = None  # 直近で注文した時間
        prev_history_num = None #直近で注文した際の履歴件数
        latest_history_open_time = None

        prev_deal_datetime = None  # 直近で決済した時間
        prev_deal_start_time = None
        prev_set_stoploss_time = None
        first_loop = True
        closes = []
        closes_org = []
        closes_local = []

        rateErrSend = False

        close_take = 0.0
        predict_take = 0.0
        order_take = 0.0
        deal_take = 0.0
        db_take = 0.0
        loop_take = 0.0
        order_pre_take = 0.0

        take1 = 0.0
        take2 = 0.0

        end_flg = False  # 終了時間間際になったらTrueにして新規注文しない

        prev_spread = 0

        sleep_err = False
        sleep_time = 0.0
        # conf.LOOP_TERMの秒ごとのループ処理

        prev_rate_dict = {}
        start_rate_dict = {}
        prev_deal_dict = {}
        time_over_flg = False

        # 発注履歴保持
        ordered_dict = {}
        # ポジション数
        prev_position_num = 0

        while (True):
            base_t_just = int(base_t - 0.01)  # base_tは0.01秒遅くなっているため
            base_t_just_score = int(base_t_just)
            base_t_just_dt = datetime.datetime.fromtimestamp(base_t_just_score)
            # とりあえず予想をSAMEにしておく
            sign = 1
            sign_ext = 1
            start = time.perf_counter()

            offset = base_t - time.time()  # 起動すべき時間と起動した時間の差
            # offset_str = '{:.5f}'.format(offset)
            # print(offset_str)
            tmp_offset = offset
            if tmp_offset < 0:
                tmp_offset = tmp_offset * -1
            # offsetが1000ミリ秒以上の場合メール送信 早くても遅くても駄目
            if tmp_offset > 1:
                conf.LOGGER("offset over 1000milces", offset, close_take, predict_take, order_take, deal_take, db_take,
                            loop_take, sleep_time)
                # mail.send_message(subject=conf.SERVER_NAME, msg="SYSTEM ERROR OCCURED! EXIT!!")
                conf.TIMEOVER_CNT += 1
                err_flg = True
                time_over_flg = True
                break

            try:
                modal = driver.find_element(By.XPATH, conf.MODAL_PATH)
                if modal.text != "":
                    driver.find_element(By.XPATH, conf.MODAL_CANCEL).click()
                    mail.send_message(subject=conf.SERVER_NAME, msg="get modal!!!")
                    # conf.LOGGER(driver.page_source)
                    # time.sleep(300)
                    err_flg = True
                    # sleep_err = True
                    break

            except selenium.common.exceptions.NoSuchElementException as e:
                # conf.LOGGER("NoSuchElementException:modal")
                pass

            """
            #hcapthaが表示されているかチェック
            if get_hcaptcha(conf, driver):
                mail.send_message(subject=conf.SERVER_NAME, msg="get hcaptcha!!!")
                time.sleep(300)

                err_flg = True
                #sleep_err = True
                break
            """

            if conf.RATE_ERR_CNT >= conf.RATE_ERR_MAX_CNT:
                # レート取得失敗回数上限に達した場合エラー
                conf.LOGGER("RATE_ERR_MAX_CNT OVER !!!", conf.RATE_ERR_CNT)
                mail.send_message(subject=conf.SERVER_NAME, msg="RATE_ERR_MAX_CNT OVER !!!")
                err_flg = True
                break

            start_take1 = time.perf_counter()
            position_num = get_position_num(conf, driver)

            if prev_position_num != 0 and position_num == 0:
                #ポジション数が0になったら保持している発注履歴は初期化
                ordered_dict = {}

            take1 = time.perf_counter() - start_take1
            if take1 > 3:
                conf.LOGGER("take1 too late:", take1)
                err_flg = True
                sleep_err = True
                break

            if position_num > conf.MAX_POSITION_NUM:
                # 最大ポジション数上限に達した場合エラー
                # その数に達するということは決算処理がおいついていないため
                conf.LOGGER("MAX_POSITION_NUM OVER !!!", position_num, conf.MAX_POSITION_NUM)
                mail.send_message(subject=conf.SERVER_NAME, msg="MAX_POSITION_NUM OVER !!!")
                err_flg = True
                break

            start_take2 = time.perf_counter()
            # 1分ごとの処理
            if base_t_just_dt.second == 0:
                if conf.EXCEPT_CNT >= conf.MAX_EXCEPT_CNT:
                    conf.LOGGER("EXCEPT_CNT over" + str(conf.MAX_EXCEPT_CNT))
                    raise Exception("EXCEPT_CNT over" + str(conf.MAX_EXCEPT_CNT))
                    #err_flg = True
                    #break

            # ループ終了時間間際になったら新規注文をやめる。
            tdt = datetime.datetime.now()
            if tdt >= conf.LOOP_END_DATETIME_PRE:
                if end_flg != True:
                    end_flg = True
                    conf.LOGGER("loop end_flg true!")

            # ループ終了時間になったらポジションあれば決済し、抜ける
            if tdt >= conf.LOOP_END_DATETIME:
                # ポジションあったら決済
                if get_position_num(conf, driver) != 0:
                    # 決済する
                    do_deal_all(conf, driver, catch_except=False, sleep_time=5)

                conf.LOGGER("Total Money:" + str(get_money(conf, driver)) + " BET_CNT:" + str(conf.BET_CNT) +
                            " TIMEOVER_CNT:" + str(conf.TIMEOVER_CNT) + " EXCEPT_CNT:" + str(
                    conf.EXCEPT_CNT) + " RATE_ERR_CNT:" + str(conf.RATE_ERR_CNT))
                conf.LOGGER("trade end!!")
                return_code = 1
                break

            # 終了時間間際になったら新規注文をやめる。
            tdt = datetime.datetime.now()
            if tdt >= conf.END_DATETIME_PRE:
                if end_flg != True:
                    end_flg = True
                    conf.LOGGER("end_flg true!")

            # 終了時間になったらポジションあれば決済し、抜ける
            if tdt >= conf.END_DATETIME:
                # ポジションあったら決済
                if get_position_num(conf, driver) != 0:
                    # 決済する
                    do_deal_all(conf, driver, catch_except=False, sleep_time=5)

                conf.LOGGER("Total Money:" + str(get_money(conf, driver)) + " BET_CNT:" + str(conf.BET_CNT) +
                            " TIMEOVER_CNT:" + str(conf.TIMEOVER_CNT) + " EXCEPT_CNT:" + str(
                    conf.EXCEPT_CNT) + " RATE_ERR_CNT:" + str(conf.RATE_ERR_CNT))
                conf.LOGGER("trade end!!")
                return_code = 2
                break

            take2 = time.perf_counter() - start_take2


            #予想取得
            start_predict = time.perf_counter()

            response, now_close = get_predict(conf, base_t_just_score)
            #conf.LOGGER("response, now_close:",response, now_close)

            sign = 1
            sign_ext = 1

            signs = response.split("_")
            probe_up = float(signs[0])
            probe_same = float(signs[1])
            probe_dw = float(signs[2])

            if probe_up >= probe_dw and probe_up >= conf.BET_BORDER:
                sign = 0
            elif probe_dw > probe_up and probe_dw >= conf.BET_BORDER:
                sign = 2

            if probe_up >= probe_dw and probe_up >= conf.BET_BORDER_EXT:
                sign_ext = 0
            elif probe_dw > probe_up and probe_dw >= conf.BET_BORDER_EXT:
                sign_ext = 2

            if conf.NO_DEAL_FLG == True or end_flg == True:
                sign = 1
                sign_ext = 1

            predict_take = time.perf_counter() - start_predict
            #conf.LOGGER("predict_take:", predict_take)

            """
            #リモートのレート取得
            close = get_oanda_close(conf, base_t_just)
                        if len(close) == 0:
                conf.LOGGER("get oanda close failed!!!")
                err_flg = True
                break
            """
            start_close = time.perf_counter()
            if conf.ORDER_TYPE == 'DETAIL':
                close_local = get_close_local(conf, driver)
            elif conf.ORDER_TYPE == 'ONECLICK':
                close_local = get_close_local_oneclick(conf, driver)
            #print(close_local)

            try:
                # スプレッド取得
                if conf.ORDER_TYPE == 'DETAIL':
                    spread = float(wait.until(EC.presence_of_element_located((By.XPATH, conf.SPREAD_PATH))).text)
                elif conf.ORDER_TYPE == 'ONECLICK':
                    spread = float(wait.until(EC.presence_of_element_located((By.XPATH, conf.SPREAD_PATH_ONECLICK))).text)

                spread = int(get_decimal_multi(spread, 10))
                prev_spread = int(get_decimal_multi(spread, 10))
                # conf.LOGGER("spread:", spread)
            except selenium.common.exceptions.StaleElementReferenceException as e:
                conf.LOGGER("StaleElementReferenceException:SPREAD_PATH")
                spread = prev_spread
                conf.RATE_ERR_CNT += 1
            except selenium.common.exceptions.TimeoutException as e:
                conf.LOGGER("TimeoutException:SPREAD_PATH")
                spread = prev_spread
                conf.RATE_ERR_CNT += 1

            if conf.RATE_ERR_CNT >= 100:
                raise Exception("RATE_ERR_CNT over 100!!!")

            close_take = time.perf_counter() - start_close

            """
            # 過去二時間分レート取得
            if first_loop:
                conf.LOGGER("first_loop get onada start")
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
                conf.LOGGER("first_loop get oanda end")

            #mt5から分足データを取得する
            if len(conf.FOOT_DICT):
                foot_data_dict = get_oanda_foot(conf)
                if foot_data_dict == None:
                    conf.LOGGER("get oanda foot data failed!!!")
                    err_flg = True
                    break

            for tmp_c in close:
                closes.append(tmp_c)  # 最初に取得したレートを追加
                closes_org.append(tmp_c)
            """

            if close_local == 0:
                if len(closes_local) !=0:
                    closes_local.append(closes_local[-1])
            else:
                closes_local.append(close_local)

            """
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
            """

            while True:
                if len(closes_local) > conf.MAX_LEN + 1:
                    closes_local.pop(0)
                else:
                    break

            #now_close = close[-1]

            """
            # 3分間レートが変わっていなかったら異常発生としメール送信！
            if len(closes_org) >= 90:
                startInd = len(closes_org) - 89
                rate_err = True
                for j in range(89):
                    if closes_org[startInd - 1] != closes_org[startInd + j]:
                        # 変化あったらエラーなし
                        rate_err = False
                        break

                if rate_err and rateErrSend == False:
                    conf.LOGGER("rate has not Changed for 3 min !")
                    # mail.send_message(subject=conf.SERVER_NAME, msg="rate has not Changed for 3 min !")
                    rateErrSend = True
                    err_flg = True
                    break
            """

            if len(closes_local) >= 90:
                startInd = len(closes_local) - 89
                rate_err = True
                for j in range(89):
                    if closes_local[startInd - 1] != closes_local[startInd + j]:
                        # 変化あったらエラーなし
                        rate_err = False
                        break

                if rate_err:
                    conf.LOGGER("closes_local rate has not Changed for 3 min !")
                    conf.RELOAD_CNT += 1

                    if conf.RELOAD_CNT >= 5:
                        conf.LOGGER("closes_local rate has not Changed five times!")
                        err_flg = True
                        break
                    else:
                        raise Exception("closes_local rate has not Changed")

            """"
            # 予想取得
            start_predict = time.perf_counter()
            try:
                #json_data = {'spr': 0, 'vals': closes}
                json_data = {'score': base_t_just, 'vals': closes}

                if len(conf.FOOT_DICT):
                    for k ,v in foot_data_dict.items():
                        for name, v_list in v.items():
                            #keyをsec表記にして渡す
                            json_data[str(k * 60) + "_" + name] = v_list

                response = requests.post(conf.REQUEST_URL, json=json_data)
                signs = response.text.split("_")
                probe_up = float(signs[0])
                probe_same = float(signs[1])
                probe_dw = float(signs[2])

                if probe_up >= probe_dw and probe_up >= conf.BET_BORDER:
                    sign = 0
                elif probe_dw > probe_up and probe_dw >= conf.BET_BORDER:
                    sign = 2

                if probe_up >= probe_dw and probe_up >= conf.BET_BORDER_EXT:
                    sign_ext = 0
                elif probe_dw > probe_up and probe_dw >= conf.BET_BORDER_EXT:
                    sign_ext = 2

                # conf.LOGGER(response.text)
                # conf.LOGGER(json.loads(response.text))
                #sign = response.text
                #sign = int(sign)
            except Exception as request_e:
                err_flg = True
                conf.LOGGER("response", response.text)
                conf.LOGGER(tracebackPrint(request_e))
                break

            if (first_loop == False and conf.NO_DEAL_FLG == False and end_flg == False and conf.MAX_LEN + 1 == len(
                    closes)) == False:
                sign = 1

            predict_take = time.perf_counter() - start_predict
            """

            """
            #予想時間が遅くなったらメールで知らせる
            if predict_take > 0.2 and conf.predict_slow_send == False:
                conf.LOGGER("predict slow:" + str(predict_take))
                mail.send_message(subject=conf.SERVER_NAME, msg="predict slow:" + str(predict_take))
                conf.predict_slow_send = True
            """

            #発注処理
            start_order = time.perf_counter()

            do_trade_flg = True
            # トレード指定対象外時間ならトレードしない
            for time_range in conf.EXCEPT_DATETIME:
                start_time_range, end_time_range = time_range
                if start_time_range <= tdt and tdt <= end_time_range:
                    do_trade_flg = False
                    break

            if tdt.second in conf.EXCEPT_SEC:
                do_trade_flg = False

            if tdt.minute in conf.EXCEPT_MIN:
                do_trade_flg = False

            """
            bef_c = closes[-1 - int(get_decimal_divide(conf.MAX_DIV_SEC, conf.AI_MODEL_TERM))]
            div = get_divide(bef_c, closes[-1])

            if conf.MAX_DIV != None:
                if conf.MAX_DIV < abs(div):
                    do_trade_flg = False
            """

            # 既に同じシフトのポジションがあるか確認
            tmp_shift = base_t_just_score % conf.NG_SHIFT
            same_shift_flg = False
            for o_score in ordered_dict.keys():
                o_shift = o_score % conf.NG_SHIFT
                if o_shift == tmp_shift:
                    same_shift_flg = True

            #現在の損益が大きい場合は予想的中率が低いとみなし、取引しない
            if conf.ORDER_TOTAL_STOPLOSS != None:
                if conf.PAIR == 'EURUSD':
                    close_usdjpy = get_oanda_close_usdjpy(conf)
                    if close_usdjpy == None:
                        conf.LOGGER("get oanda close usdjpy failed!!!")
                        err_flg = True
                        break
                    else:
                        total_profit_pips = get_now_profit_pips_eurusd(conf, driver, position_num, close_usdjpy)
                else:
                    total_profit_pips = get_now_profit_pips(conf, driver, position_num)

                if total_profit_pips <= conf.ORDER_TOTAL_STOPLOSS:
                    conf.LOGGER("UNDER ORDER_TOTAL_STOPLOSS !! total_profit_pips :",total_profit_pips)
                    do_trade_flg = False

            #過去24時間の取引数の制限
            trade_cnt = get_trade_cnt(conf, redis_db, base_t_just)
            if trade_cnt + 1 >= conf.MAX_TRADE_CNT:
                #24時間での最大取引可能回数を超えた場合
                do_trade_flg = False
                if max_trade_cnt == False:
                    conf.LOGGER("MAX_TRADE_CNT OVER:", trade_cnt)
                    max_trade_cnt = True
            else:
                max_trade_cnt = False

            # トレード。
            if do_trade_flg == True and (sign == 0 or sign == 2) and (
                    (conf.SINGLE_FLG == False and position_num < conf.MAX_POSITION_NUM) or (
                    conf.SINGLE_FLG and position_num == 0)):
                # 注文間隔の制限がある場合は前回注文から指定時間経過しているか確認
                if prev_order_time == None or (prev_order_time != None and prev_order_time + timedelta(
                        seconds=conf.TRADE_TERM) <= base_t_just_dt):

                    # 既に同じシフトのポジションがある場合は発注しない
                    if conf.EXT_FLG == False or (conf.EXT_FLG and same_shift_flg == False):
                        # スプレッドが0以下の時のみトレード。
                        if spread <= 0:
                            prev_order_time = base_t_just_dt
                            prev_history_num = get_history_num(conf, driver)

                            conf.BET_CNT += 1
                            if sign == 0:
                                probe = probe_up
                                stoploss = get_decimal_sub(now_close, conf.STOP_LOSS_FIX)
                            elif sign == 2:
                                probe = probe_dw
                                stoploss = get_decimal_add(now_close, conf.STOP_LOSS_FIX)

                            if conf.ORDER_TYPE == "ONECLICK":
                                oneclick_order(conf, driver, sign)

                            elif conf.ORDER_TYPE == "DETAIL":
                                stoploss = conf.RATE_FORMAT.format(stoploss)
                                detail_order(conf, driver, sign, stoploss)

                            if sign == 0:
                                conf.LOGGER("BUY", probe_up, )
                            elif sign == 2:
                                conf.LOGGER("SELL", probe_dw, )

                            ordered_dict[base_t_just_score] = {
                                'bet_type': conf.BET_TYPE,
                                'bet_border': conf.BET_BORDER,
                                'bet_border_ext': conf.BET_BORDER_EXT,
                                'lgbm_model_file': conf.lgbm_model_file,
                                'lgbm_model_file_suffix': conf.lgbm_model_file_suffix,
                                'lgbm_model_file_ext': conf.lgbm_model_file,
                                'lgbm_model_file_suffix_ext': conf.lgbm_model_file_suffix,
                                "order_score": base_t_just_score,
                                "sign": sign,
                                "probe": probe,
                                "start_rate": now_close,
                                "prev_rate": now_close,  # 損切ライン変更用に保持するレート
                                "position_score": None,  # 約定したスコア
                                "stoploss": float(stoploss),
                                "prev_profit": 0,
                                "open_spread": spread,
                            }

                            if sign == 0:
                                ordered_dict[base_t_just_score]["trail_stoploss"] = get_decimal_sub(now_close,
                                                                                                    conf.TRAIL_STOPLOSS_PIPS)
                            elif sign == 2:
                                ordered_dict[base_t_just_score]["trail_stoploss"] = get_decimal_add(now_close,
                                                                                                    conf.TRAIL_STOPLOSS_PIPS)

                            prev_rate_dict[base_t_just_score] = now_close
                            start_rate_dict[base_t_just_score] = now_close
                            # conf.LOGGER(prev_rate_dict, start_rate_dict)

                        # スプレッドが1以上の時は参考のためログ出力。
                        else:
                            # conf.LOGGER("cannot order with spread:", spread)
                            conf.SPREAD_OVER_CNT += 1

            order_take = time.perf_counter() - start_order

            # 決済処理
            start_deal = time.perf_counter()
            deal_flg, deal_takes, prev_deal_start_time, do_ext_flg, position_times, prev_rate_dict, prev_deal_dict, \
            start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg = \
                deal(conf, driver, prev_deal_dict, base_t_just_dt, prev_deal_start_time, sign_ext, spread,
                     now_close, prev_rate_dict, start_rate_dict, probe_up ,probe_dw, ordered_dict, exception_cnt)

            if exception_cnt == 1:
                #処理中にページが変更されたので、再度決済処理
                deal_flg, deal_takes, prev_deal_start_time, do_ext_flg, position_times, prev_rate_dict, prev_deal_dict, \
                start_rate_dict, ordered_dict, exception_cnt, all_dealed_flg = \
                    deal(conf, driver, prev_deal_dict, base_t_just_dt, prev_deal_start_time, sign_ext, spread,
                         now_close, prev_rate_dict, start_rate_dict, probe_up, probe_dw, ordered_dict,exception_cnt)

            deal_take = time.perf_counter() - start_deal

            # ストップロス設定
            if conf.ORDER_TYPE == "ONECLICK":
                do_set_stoploss_flg = set_stoploss(conf, driver, now_close)
                if do_set_stoploss_flg:
                    #ストップロス設定したタイムスタンプを保持
                    prev_set_stoploss_time = base_t_just_score

            #ベット出来ているかチェック
            if conf.PAIR == "USDJPY" or conf.PAIR == "EURUSD":
                if prev_order_time != None and all_dealed_flg == False:
                    #発注したばかりのポジションが一括決済された可能性があるので　all_dealed_flg == Falseとしている
                    prev_order_score = int(prev_order_time.timestamp())
                    if prev_order_score + 4 == base_t_just_score:
                        if len(position_times) == 0:
                            conf.BET_ERR_CNT += 1
                            conf.LOGGER("maybe cannnot bet:", prev_order_time, position_times)
                            raise Exception("maybe cannnot bet")

            start_db = time.perf_counter()

            # レートをDBに登録
            #position_num = get_position_num(conf, driver)

            regist_score = int(base_t_just) - conf.LOOP_TERM  # closeを登録するので2秒ひく
            regist_time_str = datetime.datetime.fromtimestamp(regist_score)
            child = {
                #'position_num': position_num,
                #'profit': tmp_profit,
                'bet_type': conf.BET_TYPE,
                'bet_border': conf.BET_BORDER,
                'predict_up': probe_up,
                'predict_same': probe_same,
                'predict_dw': probe_dw,
                'spread': spread,
                # 'close_take': '{:.3f}'.format(close_take),
                'predict_take': '{:.3f}'.format(predict_take),
                # 'order_take': '{:.3f}'.format(order_take),
                # 'deal_take': '{:.3f}'.format(deal_take),
                # 'db_take': '{:.3f}'.format(db_take),
                'time': str(regist_time_str),
                # 'score':regist_score,
                'loop_take': '{:.3f}'.format(loop_take),
            }
            for d in conf.DIV_REG_LIST:
                bef_c = closes[-1 - int(get_decimal_divide(d, conf.AI_MODEL_TERM))]
                div = get_divide(bef_c, closes[-1])
                child["div" + str(d)] = div

            registRedis(conf, redis_db, regist_score, child, conf.DB_KEY)
            db_take = time.perf_counter() - start_db

            # 10分ごとの処理
            if base_t_just_dt.minute % 10 == 0 and base_t_just_dt.second == 0:
                conf.LOGGER("Total Money:" + str(get_money(conf, driver)) + " BET_CNT:" + str(
                    conf.BET_CNT) + " DEAL_CNT:" + str(conf.DEAL_CNT) +  " TIMEOVER_CNT:" + str(
                    conf.TIMEOVER_CNT) + " EXCEPT_CNT:" + str(conf.EXCEPT_CNT) + " RATE_ERR_CNT:" +
                            str(conf.RATE_ERR_CNT) + " SPREAD_OVER_CNT:" + str(conf.SPREAD_OVER_CNT) + " BET_ERR_CNT:" + str(conf.BET_ERR_CNT) + \
                            " CLOSE_LOCAL_ERR_CNT:", str(conf.CLOSE_LOCAL_ERR_CNT))

            start_post1 = time.perf_counter()
                        
            # 1分ごとの処理
            if base_t_just_dt.second == 0:
                """
                if conf.PAIR == "USDJPY":
                    db_score = get_newest_history_score(conf, redis_db)
                    if base_t_just_score - db_score >= 3600 * 2:
                        #最新の履歴から2時間以上経っていたら登録
                        #ログインしなおすと履歴は最大512件しか表示されないので、こまめに登録する
                        conf.LOGGER("regist history:", db_score)
                        raise Exception("regist history")
                """
                if chk_pair(conf, driver) == False:
                    err_flg = True
                    break
                """
                if conf.PAIR == "EURUSD":
                    now_latest_history_open_time = regist_latest_history(conf, driver)

                    if prev_order_time != None:
                        prev_order_score = int(prev_order_time.timestamp())
                        if base_t_just_score - 60 < prev_order_score and prev_order_score + 13 <= base_t_just_score:

                            if now_latest_history_open_time != -1:
                                now_position_num = get_position_num(conf, driver)
                                if now_position_num == 0 and (now_latest_history_open_time == None or latest_history_open_time == now_latest_history_open_time):
                                    # 注文から13秒経過していてもポジション一覧に表示されず、かつ最新履歴の新規時刻が以前チェック時と同じまま
                                    # 発注しても約定していない可能性が高いのでエラーとする
                                    conf.BET_ERR_CNT += 1
                                    conf.LOGGER("maybe cannnot bet:", prev_order_time, latest_history_open_time,now_latest_history_open_time)
                                    if conf.BET_ERR_CNT >=1:
                                        #エラーが一定数以上になったら処理停止
                                        raise Exception("maybe cannnot bet")
                    if now_latest_history_open_time != -1:
                        latest_history_open_time = now_latest_history_open_time
                """

                if conf.ORDER_TYPE == "ONECLICK":
                    # ポジション設定チェック
                    try:
                        oneclick_position_value = wait.until(EC.presence_of_element_located((By.XPATH, conf.ONECLICK_POSITION_INPUT_PATH))).\
                            get_attribute("value")

                        if oneclick_position_value != conf.AMT_STR:
                            conf.LOGGER("oneclick_position incorrect!!", oneclick_position_value)
                            err_flg = True
                            break
                    except selenium.common.exceptions.StaleElementReferenceException as e:
                        pass

            post1_take = time.perf_counter() - start_post1
            start_post2 = time.perf_counter()

            """
            # 5分ごとの処理
            if base_t_just_dt.minute % 5 == 0 and base_t_just_dt.second == 0 :
                move_random(conf,driver)
            """
            prev_position_num = get_position_num(conf, driver)

            post2_take = time.perf_counter() - start_post2

            # 処理時間表示
            end = time.perf_counter()
            # 処理時間が1.8秒以上の場合 TIMEOVER_CNTを増やす
            process_t = end - start
            loop_take = process_t
            if process_t > (conf.LOOP_TERM + 0.9):
                conf.LOGGER("time over:", process_t, take1, take2, close_take, predict_take, order_take, deal_take,
                            db_take, loop_take, sleep_time, deal_takes, post1_take, post2_take)

            if first_loop:
                first_loop = False

            # 次に起動すべき時間
            base_t += conf.LOOP_TERM

            # 次のターンまでスリープする
            start_sleep = time.perf_counter()

            sleep_time = base_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            sleep_take = time.perf_counter() - start_sleep

            """

            while True:
                time.sleep(0.001)
                time.sleep(base_t - time.time())
                # print(datetime.now().microsecond)
                if (base_t - time.time()) < 0.005:  # time.timeの誤差を考慮して5ミリ秒早く起きる

                    break
            """
        # if rateErrSend:
        #    time.sleep(60 * 60 * 3)#状況をみるため3時間スリープ

        if err_flg:

            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)

            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)

            time.sleep(4)
            if time_over_flg and get_position_num(conf, driver) == 0:
                return_code = 7
            else:
                return_code = 3

            conf.LOGGER(
                "Total Money:" + str(get_money(conf, driver)) + " BET_CNT:" + str(
                    conf.BET_CNT) + " TIMEOVER_CNT:" + str(
                    conf.TIMEOVER_CNT) + " EXCEPT_CNT:" + str(conf.EXCEPT_CNT) + " RATE_ERR_CNT:" + str(
                    conf.RATE_ERR_CNT))

    except selenium.common.exceptions.ElementClickInterceptedException as e0:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))

        #mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")
        """
        #接続が切れた場合と思われるので確認する
        try:
            modal = driver.find_element(By.XPATH, conf.MODAL_PATH)
            if modal.text != "":
                conf.LOGGER("modal.text:",modal.text)
                #接続をキャンセルする
                driver.find_element(By.XPATH, conf.MODAL_CANCEL).click()
                return 4
        except selenium.common.exceptions.NoSuchElementException as e:
            # conf.LOGGER("NoSuchElementException:modal")
            pass
        """
        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
        except Exception as e:
            conf.LOGGER("Error Occured!!:",tracebackPrint(e))
            return 3

        return 4
    except selenium.common.exceptions.NoSuchElementException as e:
        conf.EXCEPT_CNT += 1
        conf.LOGGER("Error Occured!!:",tracebackPrint(e))
        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
        except Exception as e:
            conf.LOGGER("Error Occured!!:",tracebackPrint(e))
            return 3

        return 7

    except selenium.common.exceptions.StaleElementReferenceException as e:
        conf.EXCEPT_CNT += 1
        conf.LOGGER("Error Occured!!:",tracebackPrint(e))
        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
        except Exception as e:
            conf.LOGGER("Error Occured!!:",tracebackPrint(e))
            return 3

        return 7

    except selenium.common.exceptions.TimeoutException as e:
        conf.EXCEPT_CNT += 1
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))
        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
        except Exception as e:
            conf.LOGGER("Error Occured!!:", tracebackPrint(e))
            return 3

        return 7

    except IndexError as e:
        conf.EXCEPT_CNT += 1
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))
        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
        except Exception as e:
            conf.LOGGER("Error Occured!!:", tracebackPrint(e))
            return 3

        return 4

    except Exception as e:
        conf.EXCEPT_CNT += 1
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))

        try:
            # 接続が切れた場合と思われるので確認する
            modal = driver.find_element(By.XPATH, conf.MODAL_PATH)
            if modal.text != "":
                conf.LOGGER("modal.text:", modal.text)
                # 接続をキャンセルする
                driver.find_element(By.XPATH, conf.MODAL_CANCEL).click()

                return 4

        except selenium.common.exceptions.NoSuchElementException as e1:
            # conf.LOGGER("NoSuchElementException:modal")
            pass
        except Exception as e:
            conf.LOGGER("Error Occured!!:", tracebackPrint(e))

        try:
            # ポジションあったら決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)
            time.sleep(10)  # 直前にポジションを持った可能性があるので待って再度決済
            if get_position_num(conf, driver) != 0:
                # 決済する
                do_deal_all(conf, driver, catch_except=True, sleep_time=5)

            conf.LOGGER("Total Money:" + str(get_money(conf, driver)) + " BET_CNT:" + str(conf.BET_CNT) + " TIMEOVER_CNT:" + str(
                    conf.TIMEOVER_CNT) + " EXCEPT_CNT:" + str(conf.EXCEPT_CNT) + " RATE_ERR_CNT:" + str(conf.RATE_ERR_CNT))

            if e.__str__() == "maybe cannnot bet":
                if conf.BET_ERR_CNT >=5:
                    #一定回数ベット出来ない状態が続けばエラーとして処理停止
                    mail.send_message(conf.SERVER_NAME, "maybe cannnot bet 5times")
                    return 3

                if get_position_num(conf, driver) == 0:
                    return 4
                else:
                    return 3
            elif e.__str__() == "tr element cannot get 5time over":
                if get_position_num(conf, driver) == 0:
                    return 7
                else:
                    return 3
            elif e.__str__() == "deal cannot click!!":
                conf.LOGGER("deal cannot click!!")
                return 3
            elif e.__str__() == "same start_text!!":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    mail.send_message(conf.SERVER_NAME, "same start_text!!")
                    return 3
            elif e.__str__() == "cannot get start_text":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    return 3
            elif e.__str__() == "position_type_str not invalid":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    return 3
            elif e.__str__() == "order info too old":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    return 3
            elif e.__str__() == "cannot get order info":
                #mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    return 3
            elif e.__str__() == "same dealkey exists":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    return 7
                else:
                    return 3
            elif e.__str__() == "closes_local rate has not Changed":
                if get_position_num(conf, driver) == 0:
                    # きちんとポジション決済していればもう一度ループさせる
                    click_reaload(conf)
                    time.sleep(60)
                    if conf.ORDER_TYPE == 'ONECLICK':
                        # モーダル画面に変更
                        oneclick_order_modal_test(conf, driver)
                        # ペアチェック用のパスをワンクリック用に変更
                        conf.PAIR_PATH = conf.PAIR_PATH_ONECLICK
                    else:
                        # 一度取引してモーダルでの取引に変更すると同時に、きちんと注文できるかチェックする
                        detail_order_modal_test(conf, driver)

                    return 7
                else:
                    return 3

            elif e.__str__() =="EXCEPT_CNT over 30!!":
                return 6

            elif e.__str__() == "regist history":
                conf.LOGGER("start regist history")
                regist_history_db(conf, driver, redis_db)

                return 7

            else:
                return 3

        except Exception as e:
            conf.LOGGER("Error Occured!!:", tracebackPrint(e))
            return 3

    return return_code


if __name__ == '__main__':

    # タイマー精度を1msec単位にする
    windll.winmm.timeBeginPeriod(1)

    err_flg = False

    args = sys.argv  # yorikoiiduka.hl@gmail.com EURUSD_30 live second
    tmp_pair = args[2].split("_")[0]
    if tmp_pair == "EURUSD":
        conf = conf_thinkm_eurusd.ConfThinkM()
    elif tmp_pair == "USDJPY":
        conf = conf_thinkm.ConfThinkM()
    else:
        err_flg = True

    conf.LOGGER("ARG:", args)
    # args[0]は本ファイル名
    if len(args) == 5:
        if conf.ID != args[1]:
            conf.LOGGER("ID IS NOT CORRECT:", args[1])
            err_flg = True

        conf.ARG = args[2]
        arg_str = conf.ARG.split("_")
        conf.PAIR = arg_str[0]  # exp: GBPJPY
        conf.PRED_TERM = int(arg_str[1])  # exp: 30 sec
        conf.DB_KEY = conf.PAIR + "_TM"
        conf.DB_HISTORY_KEY = conf.PAIR + "_TM_HISTORY"
        conf.DB_ORDER_KEY = conf.PAIR + "_TM_ORDER"

        if args[3] == "demo":
            conf.DEMO_FLG = True
        else:
            conf.DEMO_FLG = False

        if args[4] == "second":
            conf.SECOND_FLG = True
        else:
            conf.SECOND_FLG = False

        if conf.PAIR == 'USDJPY':
            conf.RATE_FORMAT = "{:.3f}"
        elif conf.PAIR == 'EURUSD':
            conf.RATE_FORMAT = "{:.5f}"

    else:
        conf.LOGGER("ARG IS NOT CORRECT:", args)
        err_flg = True

    if err_flg:
        # ここまででエラーあったら終了
        mail.send_message(subject=conf.SERVER_NAME, msg="SYSTEM ERROR OCCURED! EXIT!!")
        windll.winmm.timeEndPeriod(1)
        exit(1)

    while True:
        now_dt = datetime.datetime.now()
        if now_dt< conf.START_TIME:
            time.sleep(60)
            #print("sleep")
        else:
            break

    # 初期化
    conf.initial()
    second_flg = conf.SECOND_FLG

    redis_db = redis.Redis(host=conf.HOST, port=6379, db=conf.DB_NO, decode_responses=True)
    # RedisのオートSave設定を無効にする
    print(redis_db.config_set("save", ""))

    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=True)

    redis_predict_db = redis.Redis(host=conf.PREDICT_REQUEST_HOST, port=6379, db=conf.PREDICT_REQUEST_DB_NO, decode_responses=True,
                              socket_keepalive=True)

    try:
        # ドライバー指定でChromeブラウザを開く
        CHROME_DRIVER = "C:\app\chromedriver.exe"
        # chrome_service = fs.Service(executable_path=CHROME_DRIVER)
        chrome_service = Service()
        options = webdriver.ChromeOptions()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        driver = webdriver.Chrome(service=chrome_service, options=options)
        wait = WebDriverWait(driver, 0.5)
        time.sleep(2)

        if second_flg == False:
            driver.get("https://google.com/")
            time.sleep(2)
            driver.maximize_window()  # maximize window size
            time.sleep(5)

            # ログイン画面にてcaptchaがあるので、手動でログイン情報を入力する
            driver.get("https://web.thinktrader.com/account/login")

            time.sleep(5)
            login(conf, driver)
            # conf.LOGGER("login yourself!!!")

            time.sleep(60)
            # ログインボタン押下
            # driver.find_element(By.XPATH, "//*[@id=\"root\"]/div/div/div[2]/div/form/div[7]/button").click()
            # time.sleep(30)

            # OBSで録画開始
            # start_obs(conf)

        # 明示的な待機
        # driver.implicitly_wait(1)
        demo_liveM = driver.find_element(By.XPATH,
                                         '//*[@id="root"]/div/div/div[2]/div[1]/div/div[2]/div[1]/div[6]/div[2]/div/span[1]')

        if conf.DEMO_FLG:
            conf.LOGGER("DEMO", demo_liveM.text)
        else:
            conf.LOGGER("HONBAN", demo_liveM.text)

        conf.LOGGER("PAIR:", conf.PAIR)
        conf.LOGGER("TRADE_TERM:", conf.TRADE_TERM)
        conf.LOGGER("LOOP_TERM:", conf.LOOP_TERM)
        conf.LOGGER("PRED_TERM:", conf.PRED_TERM)
        conf.LOGGER("END_HOUR:", conf.END_DATETIME.hour)
        conf.LOGGER("END_MINUTE:", conf.END_DATETIME.minute)
        conf.LOGGER("MONEY:", get_money(conf, driver))
        conf.LOGGER("AMT:", conf.AMT)

        if conf.ORDER_TYPE == 'ONECLICK':
            #モーダル画面に変更
            oneclick_order_modal_test(conf, driver)

            #ペアチェック用のパスをワンクリック用に変更
            conf.PAIR_PATH = conf.PAIR_PATH_ONECLICK
        else:
            #一度取引してモーダルでの取引に変更すると同時に、きちんと注文できるかチェックする
            detail_order_modal_test(conf, driver)

        # メイン処理を繰り返す
        while True:
            #過去24時間の取引数の制限
            base_t = time.mktime(datetime.datetime.now().timetuple())
            trade_cnt = get_trade_cnt(conf, redis_db, base_t)
            conf.LOGGER("trade_cnt:", trade_cnt)

            return_code = main_loop(conf, driver, second_flg, )
            conf.LOGGER("return_code:", return_code)
            if return_code == 1:

                time.sleep(15)
                # ログアウト
                logout(conf, driver)
                time.sleep(15)
                login(conf, driver)
                time.sleep(60)
                second_flg = False

                if conf.ORDER_TYPE == 'ONECLICK':
                    # モーダル画面に変更
                    oneclick_order_modal_test(conf, driver)
                    # ペアチェック用のパスをワンクリック用に変更
                    conf.PAIR_PATH = conf.PAIR_PATH_ONECLICK
                else:
                    # 一度取引してモーダルでの取引に変更すると同時に、きちんと注文できるかチェックする
                    detail_order_modal_test(conf, driver)

            elif return_code == 2:
                conf.LOGGER("start regist history")
                regist_history_db(conf, driver, redis_db)

                # 正常な6時間ごとのログアウト
                break
            elif return_code == 3:
                # 異常あり
                mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")
                break
            elif return_code == 4:
                # 一定回数ベット出来ない状態
                time.sleep(2)
                driver.get("https://web.thinktrader.com/account/login")
                time.sleep(5)
                login(conf, driver)
                time.sleep(60)
                second_flg = False
                if conf.ORDER_TYPE == 'ONECLICK':
                    # モーダル画面に変更
                    oneclick_order_modal_test(conf, driver)
                    # ペアチェック用のパスをワンクリック用に変更
                    conf.PAIR_PATH = conf.PAIR_PATH_ONECLICK
                else:
                    # 一度取引してモーダルでの取引に変更すると同時に、きちんと注文できるかチェックする
                    detail_order_modal_test(conf, driver)

            elif return_code == 5:
                #取引履歴DB登録のみなので終了
                break
            elif return_code == 6:
                # 異常あり
                mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")
                break

            if return_code == 7:
                second_flg = False

            conf.MAIN_LOOP_CNT += 1

            if conf.MAIN_LOOP_CNT >= conf.MAIN_LOOP_CNT_MAX:
                raise Exception("main loop cnt over: " + str(conf.MAIN_LOOP_CNT_MAX))

        if conf.DEMO_FLG:
            pass

    except Exception as e:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))
        mail.send_message(conf.SERVER_NAME, "Error Occured!! see log!!!")

    # タイマー精度を戻す
    windll.winmm.timeEndPeriod(1)
    # ログアウト
    logout(conf, driver)
    # obs録画停止
    stop_obs(conf)

    driver.quit()

    exit(0)