import redis
from datetime import datetime
from datetime import timedelta
import time
from util import *
import send_mail as mail
from datetime import date

"""
think_marketsの取引用プロセス(thinkm.py)が起動しているかの確認
レコードがDB登録されているかで確認する
"""

#チェック対象
#以下をリストで保持. host,db_no,db_name
targets = [
            ["localhost", 8 , "USDJPY_30_TM", ],
        ]

dt_now = datetime.now()

#今から3分前の0秒のレコードの存在チェック(0秒ならどんなループ間隔であっても通るため)
check_dt = datetime.datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day,
                                      hour=dt_now.hour, minute=dt_now.minute, second=0, microsecond=0)

check_dt_1_start = datetime.datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day,
                                      hour=1, minute=10, second=0, microsecond=0)
check_dt_1_end = datetime.datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day,
                                      hour=19, minute=55, second=0, microsecond=0)

check_dt_2_start = datetime.datetime(year=dt_now.year, month=dt_now.month, day=dt_now.day,
                                      hour=23, minute=10, second=0, microsecond=0)


check_dt = check_dt + timedelta(minutes=-3)
check_score= int(time.mktime(check_dt.timetuple()))

#print(check_score)

for vals in targets:
    host = vals[0]
    db_no = vals[1]
    db_name = vals[2]

    try:
        redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

        result_data = redis_db.zrangebyscore(db_name, check_score, check_score, withscores=False)
        #print(result_data)
        if len(result_data) == 0:
            weekday = dt_now.weekday()
            # 0が月曜 6は日曜
            ok_flg = True
            if weekday == 0:
                if check_dt_1_start <= dt_now and dt_now <= check_dt_1_end:
                    ok_flg = False
                elif check_dt_2_start <= dt_now:
                    ok_flg = False

            elif weekday in [1,2,3,]:
                if dt_now <= check_dt_1_end:
                    ok_flg = False
                elif check_dt_2_start <= dt_now:
                    ok_flg = False

            elif weekday == 4:
                if dt_now <= check_dt_1_end:
                    ok_flg = False

            if ok_flg == False:
                mail.send_message(host + ":" + db_name,  "check_thinkm_ps: record not exists! ")

    except Exception as e:
        mail.send_message(host + ":" + db_name + " check_thinkm_ps: Exception Occured!", tracebackPrint(e))
