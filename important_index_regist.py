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

"""
外為どっとこむの経済指標カレンダー(https://www.gaitame.com/markets/calendar/)から
指定の指標の日時をDB登録する
"""

END_YEAR_MONTH = '2026年4月'

YEAR_MONTH = '/html/body/div/div/div/div[2]/ul/li[3]'
NEXT_BUTTON = '/html/body/div/div/div/div[2]/div[2]'

TBODY = '//*[@id="calendar-table"]/tbody'
MONTH_DAY = 'td[1]/div'
DAY_TIME = 'td[2]'
COUNTRY = 'td[3]/div/div/img'
EVENT = 'td[4]/label'
IMPORTANCE = 'td[5]/span/img'

#webページに時間の表示がないもので、以下のkey_wordにひっかかるものは時間を補足してDB登録する
TIMELESS_INDEX_LIST = [
    {
        "key_word_list":["日銀金融政策決定会合、終了後政策金利発表"],#	日銀金融政策決定会合、終了後政策金利発表
        "default_hour":"11",
        "default_min":"00",
    },

]

if __name__ == '__main__':
    format = '%Y/%m/%d %H:%M'

    DB_HOST = "win2"
    DB_NO = 2
    DB_KEY = "IMPORTANT_INDEX"

    redis_db = redis.Redis(host=DB_HOST, port=6379, db=DB_NO, decode_responses=True)

    try:
        # ドライバー指定でChromeブラウザを開く
        # 月表示にして開始する年月のページを手動で開いておく
        CHROME_DRIVER = "C:\app\chromedriver.exe"
        # chrome_service = fs.Service(executable_path=CHROME_DRIVER)
        chrome_service = Service()
        options = webdriver.ChromeOptions()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9111")
        driver = webdriver.Chrome(service=chrome_service, options=options)
        wait = WebDriverWait(driver, 0.5)
        time.sleep(2)

        iframe = driver.find_element(By.ID, "parentframe")
        driver.switch_to.frame(iframe)

        year_month = driver.find_element(By.XPATH, YEAR_MONTH).text.strip()
        print("開始年月:", year_month)

        regist_cnt = 0
        prev_month_day = None
        now_year = None

        while True:
            year_month = driver.find_element(By.XPATH, YEAR_MONTH).text.strip()
            print(year_month)

            now_year =year_month.split("年")[0]

            tdobyE = driver.find_element(By.XPATH, TBODY)
            trs = tdobyE.find_elements(By.XPATH, 'tr')

            for tr in trs:
                try:
                    month_day = tr.find_element(By.XPATH, MONTH_DAY)
                    month_day = month_day.text.strip()
                    #print(month_day)
                    prev_month_day = month_day

                except selenium.common.exceptions.NoSuchElementException as e:
                    #print("month_day なし")
                    month_day = prev_month_day

                tmp_month = month_day.split("/")[0]
                tmp_day = month_day.split("/")[1].split("(")[0]

                tmp_month = f'{int(tmp_month):02}'  # 0埋めで2文字
                tmp_day = f'{int(tmp_day):02}'  # 0埋めで2文字

                event = tr.find_element(By.XPATH, EVENT).text.strip()
                print(event)
                if event == '※本日掲載の指標はありません。' or event == '休場':
                    continue

                else:
                    day_time = tr.find_element(By.XPATH, DAY_TIME).text.strip()
                    #print(day_time)

                    if day_time == "":
                        print(month_day,event)

                        regist_flg = False
                        for timeless_idx in TIMELESS_INDEX_LIST:
                            key_word_list = timeless_idx["key_word_list"]
                            #キーワードにすべてヒットするなら時間を補足してDB登録する
                            hit_flg = True
                            for k in key_word_list:
                                if (k in event) == False:
                                    hit_flg = False
                                    break

                            if hit_flg:
                                tmp_hour = timeless_idx["default_hour"]
                                tmp_min = timeless_idx["default_min"]
                                regist_flg = True
                                break

                        if regist_flg == False:
                            continue

                    else:
                        tmp_hour, tmp_min = day_time.split(":")
                        tmp_hour = f'{int(tmp_hour):02}'  # 0埋めで2文字
                        if int(tmp_hour) >= 24:
                            tmp_hour = int(tmp_hour) - 24
                            tmp_hour = f'{int(tmp_hour):02}'
                        tmp_min = f'{int(tmp_min):02}'  # 0埋めで2文字

                    dt_str = now_year + "/" + tmp_month + "/" + tmp_day + " " + tmp_hour + ":" + tmp_min
                    dt = datetime.datetime.strptime(dt_str, format)
                    dt = dt + timedelta(hours=-9)#標準時間に変換
                    score = dt.timestamp()

                    country = tr.find_element(By.XPATH, COUNTRY).get_attribute("title")
                    try:
                        importance = tr.find_element(By.XPATH, IMPORTANCE).get_attribute("class")
                    except selenium.common.exceptions.NoSuchElementException as e:
                        importance = 'importances_low'
                    print(dt_str, event, country)

                    child ={
                        'time': dt.strftime(format = format),
                        'event': event,
                        'country': country,
                        'importance': importance,
                        'score': score,
                    }
                    regist_score = score
                    while True:
                        tmp_val = redis_db.zrangebyscore(DB_KEY, regist_score, regist_score)
                        if len(tmp_val) == 0:
                            redis_db.zadd(DB_KEY, json.dumps(child), regist_score)
                            break

                        else:
                            # 既存レコードがあればscoreに0.1足す
                            regist_score += 0.1

                    regist_cnt += 1


            if year_month == END_YEAR_MONTH:
                break
            else:
                nextUL = driver.find_element(By.XPATH, NEXT_BUTTON)
                nextUL.click()
                time.sleep(2)

                #右に表示される年月をクリックする
                driver.find_element(By.XPATH, YEAR_MONTH).click()
                time.sleep(2)

        print("regist_cnt:", regist_cnt)

    except Exception as e:
        print("Error Occured!!:", tracebackPrint(e))



    mail.send_message("important_index_regist finished!!!")