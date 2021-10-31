from tensorflow.keras.utils import Sequence
from pathlib import Path
import numpy as np
from datetime import datetime
import time
from decimal import Decimal
from scipy.ndimage.interpolation import shift
import redis
import json
import random
from readConf2 import *
import scipy.stats
import gc
import math
from subprocess import Popen, PIPE
import psutil
import sys

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

class DataSequence2(Sequence):

    def __init__(self, rec_num, startDt, endDt, test_flg, eval_flg):
        #コンストラクタ
        self.db_fake = []
        self.db_fake_score = {}
        self.db_fake_score_list = [] #単純にscoreのリスト
        self.db1 = [] #divide値を保持
        self.db1_score = {} #scoreとdb1リストのインデックスを保持
        self.db1_score_list = [] #データ確認用
        self.db2 = []
        self.db2_score = {}
        self.db2_score_list = [] #単純にscoreのリスト
        self.db3 = []
        self.db3_score = {}
        self.db3_score_list = []
        self.db4 = []
        self.db4_score = {}
        self.db4_score_list = []
        self.db5 = []
        self.db5_score = {}
        self.db5_score_list = []
        self.db6 = []
        self.db6_score = {}
        self.db6_score_list = []
        self.db7 = []
        self.db7_score = {}
        self.db7_score_list = []

        self.rec_num = rec_num
        # 学習対象のみの各DBのインデックスと,DB内のインデックスおよび正解ラベルが入った子リストを保持する
        # このインデックスを元に配列から指定期間分のデータと正解ラベルを取得する
        # ex:
        # [
        # [ [0,1,0], [100], [101], [45] ], これで学習データ1つ分 左から正解ラベル, DB1内のインデックス, DB2内のインデックス, DB3内のインデックス
        # [ [1,0,0], [10], [6], [200] ],
        # ]
        self.start_score = int(time.mktime(startDt.timetuple()))
        self.end_score = int(time.mktime(endDt.timetuple()))
        self.test_flg = test_flg
        self.eval_flg = eval_flg
        self.time_list = []
        #testの場合のみ正解ラベルをリターンする
        self.correct_list = [] #test用 正解ラベルを保持
        self.train_list = []
        self.train_dict = {} #test用
        self.train_dict_ex = {}  # test用 score_listと長さを合わせる
        self.pred_close_list = [] #test用 予想時のレート保持
        self.real_close_list = [] #test用 正解レートを保持
        self.score_list = [] #test用
        self.score_dict = {} #test用
        self.close_list = [] #test用
        self.train_score_list = [] #test用 予想対象のスコアを保持
        self.train_list_idx = [] #test用
        self.spread_cnt_dict = {} #test用 スプレッド毎の件数を保持
        self.spread_cnt = 0
        self.target_spread_list = [] #test用
        self.target_divide_prev_list = [] #test用
        self.target_divide_aft_list = []  # test用
        self.target_predict_list = [] #test用

        self.same_db_flg = True #すべて同じ足の長さのDBを使うかどうか 例)すべてGBPJPY_2_0のDBをつかう

        self.db_no = DB_NO
        self.real_spread_flg = REAL_SPREAD_FLG

        if self.test_flg:
            self.db_no = DB_EVAL_NO

        if self.eval_flg:
            self.db_no = DB_EVAL_NO
            self.real_spread_flg = REAL_SPREAD_EVAL_FLG

        #すべて同じ足の長さのDBを使うかどうか判定
        for i, db in enumerate(INPUT_LEN):
            if i != len(INPUT_LEN) -1:
                if DB_TERMS[i] != DB_TERMS[i + 1]:
                    self.same_db_flg = False

        print("same_db_flg:", self.same_db_flg)

        r = redis.Redis(host='localhost', port=6379, db=self.db_no, decode_responses=True)

        # Fake DB
        db_fake_index = 0
        if DB_FAKE_TERM != 0:
            # メモリ空き容量を取得
            print("before fake db ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            for i, db in enumerate(DB_FAKE_LIST):
                if test_flg == False:
                    #1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    #result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)
                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db_fake.append(tmps.get("c"))
                    else:
                        self.db_fake.append(tmps.get("d"))
                    self.db_fake_score[score] = db_fake_index
                    self.db_fake_score_list.append(score)

                    db_fake_index += 1

                del result

        if self.same_db_flg == False:

            # メモリ空き容量を取得
            print("before db2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            # 長い足のDBから先に全件取得
            db2_index = 0
            for i, db in enumerate(DB2_LIST):
                if test_flg == False:
                    #1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    #result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                #print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)
                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db2.append(tmps.get("c"))
                    else:
                        self.db2.append(tmps.get("d"))
                    self.db2_score[score] = db2_index
                    self.db2_score_list.append(score)

                    db2_index += 1

                del result
                if test_flg == False:
                    r.delete(db) #メモリ節約のため参照したDBは削除する

            # メモリ空き容量を取得
            print("before db3 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db3_index = 0
            for i, db in enumerate(DB3_LIST):
                if test_flg == False:
                    #endscoreより1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    # result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                #print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)
                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db3.append(tmps.get("c"))
                    else:
                        self.db3.append(tmps.get("d"))

                    self.db3_score[score] = db3_index
                    self.db3_score_list.append(score)

                    db3_index += 1

                del result
                if test_flg == False:
                    r.delete(db) #メモリ節約のため参照したDBは削除する

            # メモリ空き容量を取得
            print("before db4 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db4_index = 0
            for i, db in enumerate(DB4_LIST):
                if test_flg == False:
                    #endscoreより1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    # result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                #print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)

                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db4.append(tmps.get("c"))
                    else:
                        self.db4.append(tmps.get("d"))

                    self.db4_score[score] = db4_index
                    self.db4_score_list.append(score)

                    db4_index += 1

                del result
                if test_flg == False:
                    r.delete(db) #メモリ節約のため参照したDBは削除する

            # メモリ空き容量を取得
            print("before db5 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db5_index = 0
            for i, db in enumerate(DB5_LIST):
                if test_flg == False:
                    #endscoreより1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    # result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                #print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)

                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db5.append(tmps.get("c"))
                    else:
                        self.db5.append(tmps.get("d"))

                    self.db5_score[score] = db5_index
                    self.db5_score_list.append(score)

                    db5_index += 1

                del result
                if test_flg == False:
                    r.delete(db) #メモリ節約のため参照したDBは削除する

                # メモリ空き容量を取得
            print("before db6 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db6_index = 0
            for i, db in enumerate(DB6_LIST):
                if test_flg == False:
                    # endscoreより1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    # result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                # print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)

                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db6.append(tmps.get("c"))
                    else:
                        self.db6.append(tmps.get("d"))

                    self.db6_score[score] = db6_index
                    self.db6_score_list.append(score)

                    db6_index += 1

                del result
                if test_flg == False:
                    r.delete(db)  # メモリ節約のため参照したDBは削除する

                # メモリ空き容量を取得
            print("before db7 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db7_index = 0
            for i, db in enumerate(DB7_LIST):
                if test_flg == False:
                    # endscoreより1日余分に読み込む
                    result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                    result.reverse()
                    # result = r.zrange(db, 0, -1, withscores=True)
                else:
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                # print(db ,len(result))

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)

                    if DIVIDE_ALL_FLG or DIRECT_FLG:
                        self.db7.append(tmps.get("c"))
                    else:
                        self.db7.append(tmps.get("d"))

                    self.db7_score[score] = db7_index
                    self.db7_score_list.append(score)

                    db7_index += 1

                del result
                if test_flg == False:
                    r.delete(db)  # メモリ節約のため参照したDBは削除する

        # メモリ空き容量を取得
        print("before db1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        up = 0
        down = 0

        db1_index = 0
        for i, db in enumerate(DB1_LIST):

            list_idx = db1_index

            if test_flg == False:
                #lstm_generator用
                result = r.zrevrangebyscore(db, self.start_score, self.end_score, start=0, num=self.rec_num + 1, withscores=True)
                result.reverse()

            else:
                #testLstm用
                result = r.zrangebyscore(db, self.start_score, self.end_score, withscores=True)

            close_tmp, devide_tmp, score_tmp, spread_tmp = [], [], [], []
            predict_tmp = []

            prev_c = 0

            for line in result:
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                close_tmp.append(tmps.get("c"))

                if DIVIDE_ALL_FLG or DIRECT_FLG:
                    self.db1.append(tmps.get("c"))
                else:
                    self.db1.append(tmps.get("d"))

                score_tmp.append(score)

                if METHOD == "LSTM2":
                    if tmps.get("p") != None:
                        predict_tmp.append(tmps.get("p"))
                    else:
                        predict_tmp.append(None)
                else:
                    predict_tmp.append(None)

                self.db1_score[score] = db1_index
                self.db1_score_list.append(score)

                #Spreadデータを使用する場合
                if self.real_spread_flg:
                    spr = tmps.get("s")
                    spread_tmp.append(spr)
                    self.spread_cnt += 1

                    flg = False
                    for k, v in SPREAD_LIST.items():
                        if spr > v[0] and spr <= v[1]:
                            self.spread_cnt_dict[k] = self.spread_cnt_dict.get(k, 0) + 1
                            flg = True
                            break
                    if flg == False:
                        if spr < 0:
                            self.spread_cnt_dict["spread0"] = self.spread_cnt_dict.get("spread0", 0) + 1
                        else:
                            self.spread_cnt_dict["spread16Over"] = self.spread_cnt_dict.get("spread16Over", 0) + 1

                else:
                    spread_tmp.append(SPREAD -1)

                #test用にscoreをキーにレートを保持
                #レートはそのscoreのopenレートとする
                if test_flg:
                    if prev_c == 0:
                        #prev_cがない最初のレコードの場合、しょうがないので現在のレートを入れる
                        self.score_dict[score] = tmps.get("c")
                    else:
                        self.score_dict[score] = prev_c
                    prev_c = tmps.get("c")

                db1_index += 1

            del result

            print(datetime.fromtimestamp(min(self.db1_score)))
            print(datetime.fromtimestamp(max(self.db1_score)))

            list_idx = list_idx -1

            CNT_0 = 0

            for i in range(len(close_tmp)):
                list_idx += 1

                need_len = INPUT_LEN[0]
                if DIVIDE_ALL_FLG:
                    need_len = INPUT_LEN[0] + 1

                #inputデータが足りない場合スキップ
                if i < need_len:
                    self.train_dict_ex[score_tmp[i]] = None
                    continue

                try:
                    start_score = score_tmp[i - need_len]
                    end_score = score_tmp[i + PRED_TERM -1]
                    if end_score != start_score + ((need_len + PRED_TERM - 1)  * DB1_TERM):
                        #時刻がつながっていないものは除外 たとえば日付またぎなど
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                except Exception:
                    #start_score end_scoreのデータなしなのでスキップ
                    self.train_dict_ex[score_tmp[i]] = None
                    continue

                db_fake_index_tmp = -1
                db2_index_tmp = -1
                db3_index_tmp = -1
                db4_index_tmp = -1
                db5_index_tmp = -1
                db6_index_tmp = -1
                db7_index_tmp = -1

                # DB_FAKEを使う場合
                if DB_FAKE_TERM != 0:
                    need_len = DB_FAKE_INPUT_LEN
                    if DIVIDE_ALL_FLG:
                        need_len = DB_FAKE_INPUT_LEN + 1
                    try:
                        db_fake_index_tmp = self.db_fake_score[score_tmp[i]]  # scoreからインデックスを取得
                        start_score = self.db_fake_score_list[db_fake_index_tmp - need_len]
                        end_score = self.db_fake_score_list[db_fake_index_tmp]
                        if end_score != start_score + (need_len * DB_FAKE_TERM):
                            # 時刻がつながっていないものは除外 たとえば日付またぎなど
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                    except Exception:
                        # start_scoreのデータなしなのでスキップ
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                if self.same_db_flg == False:

                    #DB2を使う場合
                    if len(INPUT_LEN) > 1:
                        need_len = INPUT_LEN[1]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[1] + 1
                        try:
                            db2_index_tmp = self.db2_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db2_score_list[db2_index_tmp - need_len]
                            end_score = self.db2_score_list[db2_index_tmp]
                            if end_score != start_score + (need_len * DB2_TERM):
                                #時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            #start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                    # DB3を使う場合
                    if len(INPUT_LEN) > 2:
                        need_len = INPUT_LEN[2]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[2] + 1
                        try:
                            db3_index_tmp = self.db3_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db3_score_list[db3_index_tmp - need_len]
                            end_score = self.db3_score_list[db3_index_tmp]
                            if end_score != start_score + (need_len * DB3_TERM):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                    # DB4を使う場合
                    if len(INPUT_LEN) > 3:
                        need_len = INPUT_LEN[3]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[3] + 1
                        try:
                            db4_index_tmp = self.db4_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db4_score_list[db4_index_tmp - need_len]
                            end_score = self.db4_score_list[db4_index_tmp]
                            if end_score != start_score + (need_len * DB4_TERM):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                    # DB5を使う場合
                    if len(INPUT_LEN) > 4:
                        need_len = INPUT_LEN[4]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[4] + 1
                        try:
                            db5_index_tmp = self.db5_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db5_score_list[db5_index_tmp - need_len]
                            end_score = self.db5_score_list[db5_index_tmp]
                            if end_score != start_score + (need_len * DB5_TERM):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue
                    # DB6を使う場合
                    if len(INPUT_LEN) > 5:
                        need_len = INPUT_LEN[5]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[5] + 1
                        try:
                            db6_index_tmp = self.db6_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db6_score_list[db6_index_tmp - need_len]
                            end_score = self.db6_score_list[db6_index_tmp]
                            if end_score != start_score + (need_len * DB6_TERM):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                    # DB7を使う場合
                    if len(INPUT_LEN) > 6:
                        need_len = INPUT_LEN[6]
                        if DIVIDE_ALL_FLG:
                            need_len = INPUT_LEN[6] + 1
                        try:
                            db7_index_tmp = self.db7_score[score_tmp[i]]  # scoreからインデックスを取得
                            start_score = self.db7_score_list[db7_index_tmp - need_len]
                            end_score = self.db7_score_list[db7_index_tmp]
                            if end_score != start_score + (need_len * DB7_TERM):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                self.train_dict_ex[score_tmp[i]] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[score_tmp[i]] = None
                            continue

                #ハイローオーストラリアの取引時間外を学習対象からはずす
                if len(EXCEPT_LIST) != 0:
                    if datetime.fromtimestamp(score_tmp[i]).hour in EXCEPT_LIST:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                # -1のスプレッドは無視する
                if spread_tmp[i -1] < 0:
                    self.train_dict_ex[score_tmp[i]] = None
                    continue

                # 指定スプレッド以外のトレードは無視する
                if EXCEPT_SPREAD_FLG:
                    if not (spread_tmp[i -1] in TARGET_SPREAD_LIST):
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                # 指定した秒のトレードは無視する
                if len(EXCEPT_SEC_LIST) != 0:
                    target_sec = datetime.fromtimestamp(score_tmp[i]).second
                    if target_sec in EXCEPT_SEC_LIST:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                # 指定した分のトレードは無視する
                if len(EXCEPT_MIN_LIST) != 0:
                    target_min = datetime.fromtimestamp(score_tmp[i]).minute
                    if target_min in EXCEPT_MIN_LIST:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                if METHOD == "LSTM2":
                    if predict_tmp[i] == None:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                    """
                    #予想値をdivide / predict_tmp[i]にしている場合、0を除外。0除算エラーとなるため
                    if not(test_flg and eval_flg == False) and predict_tmp[i] == 0:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue
                    """

                # 正解をいれていく
                bef = close_tmp[i -1]
                aft = close_tmp[i -1 + PRED_TERM]

                divide_org = aft / bef
                if aft == bef:
                    divide_org = 1

                divide = 10000 * (divide_org - 1)

                if test_flg == False:
                    if DIVIDE_MAX !=0 :
                        if abs(divide) < DIVIDE_MIN or DIVIDE_MAX < abs(divide) :
                            #変化率が大きすぎる場合 外れ値とみなして除外
                            continue
                    else:
                        if abs(divide) < DIVIDE_MIN :
                            continue

                #正解までの変化率
                divide_aft = abs(divide)

                #直近の変化率
                divide_prev = close_tmp[i - 1 - PRED_TERM] / close_tmp[i -1]
                if close_tmp[i - 1 - PRED_TERM] == close_tmp[i -1]:
                    divide_prev = 1

                divide_prev = abs(10000 * (divide_prev - 1))

                tmp_label = None

                spread = SPREAD
                spread_t = spread_tmp[i -1]

                if self.real_spread_flg:
                    spread = spread_tmp[i -1] + 1

                if BORDER_DIV != 0 and not(test_flg == True and eval_flg == False):
                    if divide >= BORDER_DIV:
                        # 上がった場合
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([1, 0, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([1, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([0, 1])
                        up = up + 1
                    elif divide <= BORDER_DIV * -1:
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([0, 0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([1, 0])
                        down = down + 1
                    else:
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([0, 1, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([0, 1])
                else:
                    if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.001")) * Decimal(str(spread))):
                        # 上がった場合
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([1, 0, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([1, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([0, 1])
                        up = up + 1
                    elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.001")) * Decimal(str(spread))):
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([0, 0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([1, 0])
                        down = down + 1
                    else:
                        if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN_BOTH":
                            tmp_label = np.array([0, 1, 0])
                        elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                            tmp_label = np.array([0, 1])
                        elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                            tmp_label = np.array([0, 1])

                    if LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION":
                        if METHOD == "LSTM2":
                            if predict_tmp[i] != None:
                                tmp_label = np.array(divide - predict_tmp[i])
                            #else:
                            #    #予想はかならずあるはずなので、なければエラー
                            #    print("Error!!! predict is None! score:", score_tmp[i])
                            #    sys.exit(1)
                        else:
                            if DIRECT_FLG:
                                tmp_label = aft
                            else:
                                tmp_label = np.array(divide)

                if test_flg and eval_flg == False:
                    # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                    # 複数のDBを使用した場合に結果を時系列順にならべて確認するため
                    if len(self.train_dict) == 0:
                        print("first score", score_tmp[i])
                    self.train_dict[score_tmp[i]] = [tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp,
                                                     db5_index_tmp, db6_index_tmp, db7_index_tmp, predict_tmp[i], bef, aft, spread_t, divide_prev, divide_aft]
                else:
                    self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp, db6_index_tmp, db7_index_tmp, predict_tmp[i], bef ])

            #if test_flg == False:
            #    r.delete(db) #メモリ節約のため参照したDBは削除する

        # メモリ空き容量を取得
        #print("before db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        # メモリ節約のためredis停止
        if test_flg == False:
            #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
            sudo_password = 'Reikou0129'
            command = 'systemctl stop redis'.split()

            p = Popen(['sudo', '-S'] + command, stdin=PIPE, stderr=PIPE,
                      universal_newlines=True)
            sudo_prompt = p.communicate(sudo_password + '\n')[1]

            # メモリ空き容量を取得
            print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")


        if test_flg and eval_flg == False:
            #一番短い足のDBを複数インプットする場合用に
            #score順にならべかえてtrain_listと正解ラベルおよびレート変化幅(aft-bef)をそれぞれリスト化する
            for data in sorted(self.train_dict.items()):
                self.train_list.append(data[1][:10])
                self.correct_list.append(data[1][0])
                self.pred_close_list.append(data[1][9])
                self.real_close_list.append(data[1][10])
                self.target_spread_list.append(data[1][11])
                self.target_divide_prev_list.append(data[1][12])
                self.target_divide_aft_list.append(data[1][13])
                self.train_score_list.append(data[0])
                self.target_predict_list.append(data[1][8])

            #一番短い足のDBを複数インプットする場合用に
            #scoreのリストcloseのリストを作成 結果確認のグラフ描写で使うため
            for data in sorted(self.score_dict.items()):
                self.score_list.append(data[0])
                self.close_list.append(data[1])

            #train対象のスコアと対象外のスコアを結合
            self.train_dict_ex.update(self.train_dict)
            cnt_ex = 0
            for data in sorted(self.train_dict_ex.items()):
                if data[1] == None:
                    #train対象外のスコアは-1をいれる
                    self.train_list_idx.append(-1)
                else:
                    #train対象であればscore順のtrain_dictのインデックスをいれる
                    self.train_list_idx.append(cnt_ex)
                    cnt_ex += 1

            del self.train_dict, self.score_dict

        # メモリ解放
        del close_tmp
        del score_tmp

        #numpy化してsliceを早くする
        self.db1_np = np.array(self.db1)
        self.db2_np = np.array(self.db2)
        self.db3_np = np.array(self.db3)
        self.db4_np = np.array(self.db4)
        self.db5_np = np.array(self.db5)
        self.db6_np = np.array(self.db6)
        self.db7_np = np.array(self.db7)

        del self.db1, self.db2, self.db3, self.db4, self.db5, self.db6, self.db7

        self.data_length = len(self.train_list)
        self.cut_num = (self.data_length) % BATCH_SIZE

        print("data length:" , self.data_length)

        if self.cut_num !=0:
            # エポック毎の学習ステップ数
            # バッチサイズが1ステップで学習するデータ数
            # self.cut_num(総学習データ÷バッチサイズ)で余りがあるなら余りがある分ステップ数が1多い
            self.steps_per_epoch = int(self.data_length / BATCH_SIZE) +1
        else:
            self.steps_per_epoch = int(self.data_length / BATCH_SIZE)

        print("steps_per_epoch: ", self.steps_per_epoch)
        myLogger("UP: ",up/self.data_length)
        if self.data_length - up - down != 0:
            myLogger("SAME: ", (self.data_length - up - down) / (self.data_length))
        myLogger("DOWN: ",down/self.data_length)

        #Spreadの内訳を表示
        if self.real_spread_flg:
            print("spread total: ", self.spread_cnt)
            if self.spread_cnt != 0:
                for k, v in sorted(self.spread_cnt_dict.items()):
                    print(k, v, v / self.spread_cnt)

        #シャッフルしとく
        if test_flg == False:
            random.shuffle(self.train_list)

        """
        for i in self.train_list[0:3]:
            db1_index_tmp = i[1]

            print("db1_score:", self.db1_score_list[db1_index_tmp])
            print("db1_divide:", self.db1_np[db1_index_tmp - 5 : db1_index_tmp])

        if len(INPUT_LEN) == 3:
            for i in self.train_list[0:3]:
                db1_index_tmp = i[1]
                db2_index_tmp = i[2]
                db3_index_tmp = i[3]

                print("db1_score:", self.db1_score_list[db1_index_tmp])
                print("db1_divide:", self.db1_np[db1_index_tmp - 5 : db1_index_tmp])
                print("db2_score:", self.db2_score_list[db2_index_tmp])
                print("db2_divide:", self.db2_np[db2_index_tmp - 5 : db2_index_tmp])
                print("db3_score:", self.db3_score_list[db3_index_tmp])
                print("db3_divide:", self.db3_np[db3_index_tmp - 5 : db3_index_tmp])
        """
        def tmp_create_db1(x, start_idx):
            # 取得すべきdb1のインデックス番号を決定
            target_idx = self.train_list[start_idx + x][1]

            """
            if test_flg:
                if start_idx == 0 and x < 10:
                    print("score:", self.db1_score_list[self.train_list[start_idx + x][1]])
            """

            return self.db1_np[target_idx - INPUT_LEN[0] :target_idx]

        def tmp_create_db(x, start_idx, db_no):
            target_idx = self.train_list[start_idx + x][db_no]

            if self.same_db_flg == False:
                if db_no ==1 :
                    return self.db1_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==2 :
                    return self.db2_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==3 :
                    return self.db3_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==4 :
                    return self.db4_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==5 :
                    return self.db5_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==6 :
                    return self.db6_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
                elif db_no ==7 :
                    return self.db7_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]
            else:
                 return self.db1_np[target_idx - INPUT_LEN[db_no - 1]:target_idx]


        self.create_db = np.vectorize(tmp_create_db, otypes=[np.ndarray])

        def tmp_create_db_all(x, start_idx, db_no):
            target_idx = self.train_list[start_idx + x][db_no]

            if db_no ==1 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db1_np[target_idx -1])
                tmp_arr = self.db1_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000

            elif db_no ==2 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db2_np[target_idx -1])
                tmp_arr = self.db2_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==3 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db3_np[target_idx -1])
                tmp_arr = self.db3_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==4 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db4_np[target_idx -1])
                tmp_arr = self.db4_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==5 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db5_np[target_idx -1])
                tmp_arr = self.db5_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==6 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db6_np[target_idx -1])
                tmp_arr = self.db6_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==7 :
                base_arr = np.full(INPUT_LEN[db_no - 1], self.db7_np[target_idx -1])
                tmp_arr = self.db7_np[target_idx - INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000

        self.create_db_all = np.vectorize(tmp_create_db_all, otypes=[np.ndarray])



        def tmp_create_label(x, start_idx):
            return self.train_list[start_idx + x][0]

        self.create_label = np.vectorize(tmp_create_label, otypes=[np.ndarray])

        def tmp_create_predict(x, start_idx):
            return self.train_list[start_idx + x][8]

        self.create_predict = np.vectorize(tmp_create_predict, otypes=[np.ndarray])

        def tmp_create_now_rate(x, start_idx):
            return self.train_list[start_idx + x][9]

        self.create_now_rate = np.vectorize(tmp_create_now_rate, otypes=[np.ndarray])

    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        #start = time.time()
        # データの取得実装
        #print("idx:", idx)
        tmp_np = np.arange(BATCH_SIZE)
        # self.cut_num(総学習データ÷バッチサイズ)で余りがあり、且つ最後のステップの場合、
        # リターンする配列のサイズはバッチサイズでなく、余りの数となる
        if idx == (self.steps_per_epoch -1) and self.cut_num != 0:
            tmp_np = np.arange(self.cut_num)
        #print("tmp_np:", tmp_np)
        #tmp_np = np.arange(3)
        #tmp = self.g(tmp_np)
        #new_tmp = tmp.tolist()
        #print(np.array(new_tmp).shape)

        # start_idxからバッチサイズ分取得開始インデックスをずらす
        start_idx = idx * BATCH_SIZE

        label_data_tmp= self.create_label(tmp_np, start_idx)
        label_data_tmp = label_data_tmp.tolist()
        retY = np.array(label_data_tmp)

        retX = []

        if METHOD == "LSTM" or METHOD == "BY" or METHOD == "LSTM2" or METHOD == "SimpleRNN"  or METHOD == "RNN"  or METHOD == "GRU" :
            for i in range(len(INPUT_LEN)):
                tmpX = np.zeros((len(tmp_np), INPUT_LEN[i], 1))

                if DIVIDE_ALL_FLG:
                    # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
                    tmp_arr = self.create_db_all(tmp_np, start_idx, i + 1)
                else:
                    tmp_arr = self.create_db(tmp_np, start_idx, i + 1)
                tmp_arr = tmp_arr.tolist()
                tmp_arr = np.array(tmp_arr)
                tmpX[:, :, 0] = tmp_arr[:]

                retX.append(tmpX)

            if METHOD == "LSTM2":
                predict_data_tmp = self.create_predict(tmp_np, start_idx)
                predict_data_tmp = predict_data_tmp.tolist()
                retX.append(np.array(predict_data_tmp))

            if NOW_RATE_FLG == True:
                rate_data_tmp = self.create_now_rate(tmp_np, start_idx)
                rate_data_tmp = rate_data_tmp.tolist()
                retX.append(np.array(rate_data_tmp))

        elif METHOD == "NORMAL":
            for i in range(len(INPUT_LEN)):

                # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
                tmp_arr = self.create_db(tmp_np, start_idx, i + 1)
                tmp_arr = tmp_arr.tolist()
                tmp_arr = np.array(tmp_arr)

                retX.append(tmp_arr)

        """
        if self.test_flg:
            if idx == 0:
                print("train_list :", self.train_list[:20])
        """

        #テストの場合はテストデータのみ返す
        if self.test_flg:
            if self.eval_flg:
                return retX, retY
            else:
                return retX
        else:
            return retX, retY

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch

    def on_epoch_end(self):
        # epoch終了時の処理 リストをランダムに並べ替える
        if self.test_flg == False:
            random.shuffle(self.train_list)

    def get_data_length(self):
        return self.data_length

    def get_correct_list(self):
        retY = np.array(self.correct_list)
        return retY

    def get_pred_close_list(self):
        return self.pred_close_list

    def get_real_close_list(self):
        return self.real_close_list

    def get_score_list(self):
        return self.score_list

    def get_close_list(self):
        return self.close_list

    def get_target_spread_list(self):
        return self.target_spread_list

    def get_target_divide_prev_list(self):
        return self.target_divide_prev_list

    def get_target_divide_aft_list(self):
        return self.target_divide_aft_list

    def get_train_score_list(self):
        return self.train_score_list

    def get_train_list_index(self):
        return self.train_list_idx

    def get_target_predict_list(self):
        return self.target_predict_list