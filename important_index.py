from datetime import datetime,timedelta ,date
import time

s_format = '%Y/%m/%d %H:%M'

indexs ={
    "雇用統計":[
        datetime.strptime("2023/4/7 12:30", s_format),
        datetime.strptime("2023/5/5 12:30", s_format),
        datetime.strptime("2023/6/2 12:30", s_format),
        datetime.strptime("2023/7/7 12:30", s_format),
        datetime.strptime("2023/8/4 12:30", s_format),
        datetime.strptime("2023/9/1 12:30", s_format),
        datetime.strptime("2023/10/6 12:30", s_format),
        datetime.strptime("2023/11/3 12:30", s_format),
        datetime.strptime("2023/12/8 13:30", s_format),
        datetime.strptime("2024/1/5 13:30", s_format),
        datetime.strptime("2024/2/2 13:30", s_format),
        datetime.strptime("2024/3/8 13:30", s_format),
        datetime.strptime("2024/4/5 12:30", s_format),
        datetime.strptime("2024/5/3 12:30", s_format),
        datetime.strptime("2024/6/7 12:30", s_format),
        datetime.strptime("2024/7/5 12:30", s_format),
        datetime.strptime("2024/8/2 12:30", s_format),
    ],
    "CPI": [
        datetime.strptime("2023/4/12 12:30", s_format),
        datetime.strptime("2023/5/10 12:30", s_format),
        datetime.strptime("2023/6/13 12:30", s_format),
        datetime.strptime("2023/7/12 12:30", s_format),
        datetime.strptime("2023/8/10 12:30", s_format),
        datetime.strptime("2023/9/13 12:30", s_format),
        datetime.strptime("2023/10/12 12:30", s_format),
        datetime.strptime("2023/11/14 13:30", s_format),
        datetime.strptime("2023/12/12 13:30", s_format),
        datetime.strptime("2024/1/11 13:30", s_format),
        datetime.strptime("2024/2/13 13:30", s_format),
        datetime.strptime("2024/3/12 12:30", s_format),
        datetime.strptime("2024/4/10 12:30", s_format),
        datetime.strptime("2024/5/15 12:30", s_format),
        datetime.strptime("2024/6/12 12:30", s_format),
        datetime.strptime("2024/7/11 12:30", s_format),
        datetime.strptime("2024/8/14 12:30", s_format),
    ],
    "ISM製造業景況指数": [
        datetime.strptime("2023/4/3 14:00", s_format),
        datetime.strptime("2023/5/1 14:00", s_format),
        datetime.strptime("2023/6/1 14:00", s_format),
        datetime.strptime("2023/7/3 14:00", s_format),
        datetime.strptime("2023/8/1 14:00", s_format),
        datetime.strptime("2023/9/1 14:00", s_format),
        datetime.strptime("2023/10/2 14:00", s_format),
        datetime.strptime("2023/11/1 14:00", s_format),
        datetime.strptime("2023/12/1 15:00", s_format),
        datetime.strptime("2024/1/3 15:00", s_format),
        datetime.strptime("2024/2/1 15:00", s_format),
        datetime.strptime("2024/3/1 15:00", s_format),
        datetime.strptime("2024/4/1 14:00", s_format),
        datetime.strptime("2024/5/1 14:00", s_format),
        datetime.strptime("2024/6/3 14:00", s_format),
        datetime.strptime("2024/7/1 14:00", s_format),
        datetime.strptime("2024/8/1 14:00", s_format),
    ],
    "GDP": [
        datetime.strptime("2023/4/27 12:30", s_format),
        datetime.strptime("2023/7/27 12:30", s_format),
        datetime.strptime("2023/10/26 12:30", s_format),
        datetime.strptime("2024/1/25 13:30", s_format),
        datetime.strptime("2024/3/28 12:30", s_format),
        datetime.strptime("2024/4/25 12:30", s_format),
        datetime.strptime("2024/7/25 12:30", s_format),
    ],
    "ADP雇用統計": [
        datetime.strptime("2023/4/5 12:15", s_format),
        datetime.strptime("2023/5/3 12:15", s_format),
        datetime.strptime("2023/6/1 12:15", s_format),
        datetime.strptime("2023/7/6 12:15", s_format),
        datetime.strptime("2023/8/2 12:15", s_format),
        datetime.strptime("2023/8/30 12:15", s_format),
        datetime.strptime("2023/10/4 12:15", s_format),
        datetime.strptime("2023/11/1 12:15", s_format),
        datetime.strptime("2023/12/6 13:15", s_format),
        datetime.strptime("2024/1/31 13:15", s_format),
        datetime.strptime("2024/3/6 13:15", s_format),
        datetime.strptime("2024/4/3 12:15", s_format),
        datetime.strptime("2024/5/1 12:15", s_format),
        datetime.strptime("2024/6/5 12:15", s_format),
        datetime.strptime("2024/7/3 12:15", s_format),
        datetime.strptime("2024/7/31 12:15", s_format),
    ],
    "ISM非製造業景況指数": [
        datetime.strptime("2023/4/5 14:00", s_format),
        datetime.strptime("2023/5/3 14:00", s_format),
        datetime.strptime("2023/6/5 14:00", s_format),
        datetime.strptime("2023/7/6 14:00", s_format),
        datetime.strptime("2023/8/3 14:00", s_format),
        datetime.strptime("2023/9/6 14:00", s_format),
        datetime.strptime("2023/10/4 14:00", s_format),
        datetime.strptime("2023/11/3 14:00", s_format),
        datetime.strptime("2023/12/5 15:00", s_format),
        datetime.strptime("2024/1/5 15:00", s_format),
        datetime.strptime("2024/2/5 15:00", s_format),
        datetime.strptime("2024/3/5 15:00", s_format),
        datetime.strptime("2024/4/3 14:00", s_format),
        datetime.strptime("2024/5/3 14:00", s_format),
        datetime.strptime("2024/6/5 14:00", s_format),
        datetime.strptime("2024/7/3 14:00", s_format),
        datetime.strptime("2024/8/5 14:00", s_format),
    ],
    "小売売上高": [
        datetime.strptime("2023/4/14 12:30", s_format),
        datetime.strptime("2023/5/16 12:30", s_format),
        datetime.strptime("2023/6/15 12:30", s_format),
        datetime.strptime("2023/7/18 12:30", s_format),
        datetime.strptime("2023/8/15 12:30", s_format),
        datetime.strptime("2023/9/14 12:30", s_format),
        datetime.strptime("2023/10/17 12:30", s_format),
        datetime.strptime("2023/11/15 13:30", s_format),
        datetime.strptime("2023/12/14 13:30", s_format),
        datetime.strptime("2024/1/17 13:30", s_format),
        datetime.strptime("2024/2/15 13:30", s_format),
        datetime.strptime("2024/3/14 12:30", s_format),
        datetime.strptime("2024/4/15 12:30", s_format),
        datetime.strptime("2024/5/15 12:30", s_format),
        datetime.strptime("2024/6/18 12:30", s_format),
        datetime.strptime("2024/7/16 12:30", s_format),
        datetime.strptime("2024/8/15 12:30", s_format),
    ],
    "新築住宅販売件数": [
        datetime.strptime("2023/4/25 14:00", s_format),
        datetime.strptime("2023/5/23 14:00", s_format),
        datetime.strptime("2023/6/27 14:00", s_format),
        datetime.strptime("2023/7/26 14:00", s_format),
        datetime.strptime("2023/8/23 14:00", s_format),
        datetime.strptime("2023/9/26 14:00", s_format),
        datetime.strptime("2023/10/25 14:00", s_format),
        datetime.strptime("2023/11/27 15:00", s_format),
        datetime.strptime("2023/12/22 15:00", s_format),
        datetime.strptime("2024/1/25 15:00", s_format),
        datetime.strptime("2024/2/26 15:00", s_format),
        datetime.strptime("2024/3/25 14:00", s_format),
        datetime.strptime("2024/4/23 14:00", s_format),
        datetime.strptime("2024/5/23 14:00", s_format),
        datetime.strptime("2024/6/26 14:00", s_format),
        datetime.strptime("2024/7/24 14:00", s_format),
    ],
    "個人消費支出": [
        datetime.strptime("2023/4/28 12:30", s_format),
        datetime.strptime("2023/5/26 12:30", s_format),
        datetime.strptime("2023/6/30 12:30", s_format),
        datetime.strptime("2023/7/28 12:30", s_format),
        datetime.strptime("2023/8/30 12:30", s_format),
        datetime.strptime("2023/9/29 12:30", s_format),
        datetime.strptime("2023/10/27 12:30", s_format),
        datetime.strptime("2023/11/30 13:30", s_format),
        datetime.strptime("2023/12/22 13:30", s_format),
        datetime.strptime("2024/1/26 13:30", s_format),
        datetime.strptime("2024/2/29 13:30", s_format),
        datetime.strptime("2024/3/29 12:30", s_format),
        datetime.strptime("2024/4/26 12:30", s_format),
        datetime.strptime("2024/5/31 12:30", s_format),
        datetime.strptime("2024/6/28 12:30", s_format),
        datetime.strptime("2024/7/26 12:30", s_format),
    ],
    "FOMC金利発表": [
        datetime.strptime("2023/5/3 18:00", s_format),
        datetime.strptime("2023/6/14 18:00", s_format),
        datetime.strptime("2023/9/20 18:00", s_format),
        datetime.strptime("2023/11/1 18:00", s_format),
        datetime.strptime("2023/12/13 19:00", s_format),
        datetime.strptime("2024/1/31 19:00", s_format),
        datetime.strptime("2024/3/20 18:00", s_format),
        datetime.strptime("2024/5/1 18:00", s_format),
        datetime.strptime("2024/6/12 18:00", s_format),
        datetime.strptime("2024/7/31 18:00", s_format),
    ],
    "日銀政策金利発表": [
        datetime.strptime("2023/4/28 2:00", s_format),
        datetime.strptime("2023/6/16 2:00", s_format),
        datetime.strptime("2023/7/28 2:00", s_format),
        datetime.strptime("2023/9/22 2:00", s_format),
        datetime.strptime("2023/10/31 2:00", s_format),
        datetime.strptime("2023/12/19 2:00", s_format),
        datetime.strptime("2024/1/23 2:00", s_format),
        datetime.strptime("2024/3/19 2:00", s_format),
        datetime.strptime("2024/4/26 2:00", s_format),
        datetime.strptime("2024/6/14 2:00", s_format),
        datetime.strptime("2024/7/31 2:00", s_format),
    ],
    "日銀記者会見": [
        datetime.strptime("2023/4/28 6:30", s_format),
        datetime.strptime("2023/6/16 6:30", s_format),
        datetime.strptime("2023/7/28 6:30", s_format),
        datetime.strptime("2023/9/22 6:30", s_format),
        datetime.strptime("2023/10/31 6:30", s_format),
        datetime.strptime("2023/12/19 6:30", s_format),
        datetime.strptime("2024/1/23 6:30", s_format),
        datetime.strptime("2024/3/19 6:30", s_format),
        datetime.strptime("2024/4/26 6:30", s_format),
        datetime.strptime("2024/6/14 6:30", s_format),
        datetime.strptime("2024/7/31 6:30", s_format),
    ],
}

#index_set:除外する指標のリスト, range:除外する指標発表時間前後の秒数
class ImportantIndex():
    def __init__(self, index_set = [], range=0):
        self.index_dict = {}

        #指標発表のYMDごとにチェックすべき日時リストを作成する
        for idx in index_set:
            tmp_list = indexs.get(idx)

            if tmp_list == None:
                print("INDEX IS INCORRECT", idx)
                exit(1)

            for dt in tmp_list:
                tmp_score = int(time.mktime(dt.timetuple()))
                start_score = tmp_score - (60 * 2)
                end_score = tmp_score + range
                if idx == "FOMC金利発表":
                    #FOMC金利発表の場合は30分後まで除外
                    end_score = tmp_score + (60 * 30)
                elif idx == "日銀政策金利発表":
                    # 後120分除外
                    end_score = tmp_score + (60 * 120)
                elif idx == "日銀記者会見":
                    # 後60分除外
                    end_score = tmp_score + (60 * 60)
                ymd = self.get_ymd(dt)
                if ymd in self.index_dict.keys():
                    self.index_dict[ymd].append([start_score, end_score])
                else:
                    self.index_dict[ymd] = [[start_score, end_score]]

    def get_ymd(self, dt):
        return str(dt.year) + str(dt.month) + str(dt.day)

    #除外すべきならTrueを返す
    def is_except(self, timestamp):
        flg = False

        if len(self.index_dict) == 0:
            #何も指標がなかったら
            return flg

        dt = datetime.fromtimestamp(timestamp)
        ymd = self.get_ymd(dt)
        if ymd in self.index_dict.keys():
            tmp_list = self.index_dict[ymd]
            for idx in tmp_list:
                if idx[0] <= timestamp and timestamp <= idx[1]:
                    flg = True
                    break

        return flg

    def get_index_list(self):
        return self.index_list


"""
c = ImportantIndex(["GDP", "CPI"], 10)
d = int(time.mktime(datetime.strptime("2023/4/27 12:30", s_format).timetuple()))
print(c.is_except(d))
"""


