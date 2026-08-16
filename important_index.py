import json
from datetime import datetime,timedelta ,date
import time

import redis

s_format = '%Y/%m/%d %H:%M'
DB_HOST = "win2"
DB_NO = 2
DB_KEY = "IMPORTANT_INDEX"

index_list = [
    {
        'time': "2024/11/5 00:00",
        'event': "大統領選挙",
        'country': "アメリカ",
        'importance': "importance_high",
    },
    {
        'time': "2024/11/6 00:00",
        'event': "大統領選挙",
        'country': "アメリカ",
        'importance': "importance_high",
    },
    ]

#index_set:除外する指標のリスト, range:除外する指標発表時間前後の秒数
class ImportantIndex():
    def __init__(self, importance, range=None, startDt=None, endDt=None, nichiginOnly=False):

        self.index_dict = {}
        self.index_list = []
        regist_cnt = 0

        if range != None:
            redis_db = redis.Redis(host=DB_HOST, port=6379, db=DB_NO, decode_responses=True)

            if startDt == None and endDt == None:
                    # result_data = redis_db.zrangebyscore(DB_KEY, 0, 1756476000,withscores=True)
                    result_data = redis_db.zrange(DB_KEY, 0, -1, withscores=False)  # 全件取得
            elif "startDt" != None and endDt != None:
                    start_score = startDt.timestamp()
                    end_score = (endDt + timedelta(days=1)).timestamp()
                    result_data = redis_db.zrangebyscore(DB_KEY, start_score, end_score, withscores=False)
            else:
                    print("startDt or endDt is incorrect")

            #print("result_data length:", len(result_data))

            #index_listをDB結果に追加
            index_list_json = []
            for i in index_list:
                #score追加しjsonに変換
                i['score'] = datetime.strptime(i.get("time"), s_format).timestamp()
                index_list_json.append(json.dumps(i))

            result_data.extend(index_list_json)

            #print("result_data + index_list length:", len(result_data))

            for body in result_data:
                tmps = json.loads(body)
                tmp_importance = tmps.get("importance")

                if importance == "importances_high" and tmp_importance != "importances_high":
                    continue
                elif importance == "importances_mid" and tmp_importance == "importances_low":
                    continue

                tmp_score = int(tmps.get("score"))
                tmp_event = tmps.get("event")
                tmp_country = tmps.get("country")
                tmp_dt = datetime.fromtimestamp(tmp_score)

                start_score = tmp_score - (60 * 2)
                end_score = tmp_score + range
                if "政策金利発表" in tmp_event and "アメリカ" == tmp_country:
                    #FOMC金利発表の場合は記者会見もあるので60分後まで除外
                    end_score = tmp_score + (60 * 60)
                elif "政策金利発表" in tmp_event and "日本" == tmp_country:
                    # 後270分除外
                    end_score = tmp_score + (60 * 270)

                elif "記者会見" in tmp_event and "日本" == tmp_country:
                    # 後120分除外
                    end_score = tmp_score + (60 * 120)

                elif "発言" in tmp_event:
                    # 後60分除外
                    end_score = tmp_score + (60 * 60)
                elif "大統領選挙" in tmp_event and "アメリカ" == tmp_country:
                    # 後60分 * 24(1日)除外
                    end_score = tmp_score + (60 * 60 * 24)

                ymd = self.get_ymd(tmp_dt)

                if nichiginOnly:
                    if "政策金利発表" in tmp_event and "日本" == tmp_country:
                        self.index_list.append([tmp_event, tmp_dt, ymd, datetime.fromtimestamp(start_score),datetime.fromtimestamp(end_score), tmp_importance])
                        if ymd in self.index_dict.keys():
                            self.index_dict[ymd].append([start_score, end_score])
                        else:
                            self.index_dict[ymd] = [[start_score, end_score]]
                    elif "記者会見" in tmp_event and "日本" == tmp_country:
                        self.index_list.append([tmp_event, tmp_dt, ymd, datetime.fromtimestamp(start_score),datetime.fromtimestamp(end_score), tmp_importance])
                        if ymd in self.index_dict.keys():
                            self.index_dict[ymd].append([start_score, end_score])
                        else:
                            self.index_dict[ymd] = [[start_score, end_score]]

                else:
                    self.index_list.append([tmp_event, tmp_dt, ymd, datetime.fromtimestamp(start_score), datetime.fromtimestamp(end_score),tmp_importance])
                    if ymd in self.index_dict.keys():
                        self.index_dict[ymd].append([start_score, end_score])
                    else:
                        self.index_dict[ymd] = [[start_score, end_score]]

                regist_cnt += 1
        #for l in self.index_list:
        #    print(l)

        print("important index date_length:", regist_cnt)

    def get_ymd(self, dt):
        return str(dt.year) + f'{int(dt.month):02}' + f'{int(dt.day):02}'

    def print_index(self):
        print(self.index_list)

    def get_index(self):
        return self.index_list

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


if __name__ == '__main__':
    start_t = time.perf_counter()
    ImportantIndex(importance="importances_high", range=300, startDt=datetime(2010, 1, 1), endDt=datetime(2025, 8, 1))
    print("process time:", time.perf_counter() - start_t)