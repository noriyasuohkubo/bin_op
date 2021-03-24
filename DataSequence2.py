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

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

class DataSequence2(Sequence):

    def __init__(self, rec_num, startDt, endDt, test_flg):
        #コンストラクタ
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

        r = redis.Redis(host='localhost', port=6379, db=DB_NO, decode_responses=True)

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

            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)
                if DIVIDE_ALL_FLG:
                    self.db2.append(tmps.get("c"))
                else:
                    self.db2.append(tmps.get("d"))
                self.db2_score[score] = db2_index
                self.db2_score_list.append(score)

                db2_index += 1

            del result

        db3_index = 0
        for i, db in enumerate(DB3_LIST):
            if test_flg == False:
                #endscoreより1日余分に読み込む
                result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                result.reverse()
                # result = r.zrange(db, 0, -1, withscores=True)
            else:
                result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)
                if DIVIDE_ALL_FLG:
                    self.db3.append(tmps.get("c"))
                else:
                    self.db3.append(tmps.get("d"))

                self.db3_score[score] = db3_index
                self.db3_score_list.append(score)

                db3_index += 1

            del result

        db4_index = 0
        for i, db in enumerate(DB4_LIST):
            if test_flg == False:
                #endscoreより1日余分に読み込む
                result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                result.reverse()
                # result = r.zrange(db, 0, -1, withscores=True)
            else:
                result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                if DIVIDE_ALL_FLG:
                    self.db4.append(tmps.get("c"))
                else:
                    self.db4.append(tmps.get("d"))

                self.db4_score[score] = db4_index
                self.db4_score_list.append(score)

                db4_index += 1

            del result

        db5_index = 0
        for i, db in enumerate(DB5_LIST):
            if test_flg == False:
                #endscoreより1日余分に読み込む
                result = r.zrevrangebyscore(db, self.start_score, self.end_score - 3600 * 24, withscores=True)
                result.reverse()
                # result = r.zrange(db, 0, -1, withscores=True)
            else:
                result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                if DIVIDE_ALL_FLG:
                    self.db5.append(tmps.get("c"))
                else:
                    self.db5.append(tmps.get("d"))

                self.db5_score[score] = db5_index
                self.db5_score_list.append(score)

                db5_index += 1

            del result

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
            prev_c = 0

            for line in result:
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                close_tmp.append(tmps.get("c"))

                if DIVIDE_ALL_FLG:
                    self.db1.append(tmps.get("c"))
                else:
                    self.db1.append(tmps.get("d"))

                score_tmp.append(score)

                self.db1_score[score] = db1_index
                self.db1_score_list.append(score)

                #Spreadデータを使用する場合
                if REAL_SPREAD_FLG:
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
                    spread_tmp.append(SPREAD)

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

                db2_index_tmp = -1

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

                db3_index_tmp = -1

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

                db4_index_tmp = -1

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

                db5_index_tmp = -1

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

                #ハイローオーストラリアの取引時間外を学習対象からはずす
                if len(EXCEPT_LIST) != 0:
                    if datetime.fromtimestamp(score_tmp[i]).hour in EXCEPT_LIST:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                # 指定スプレッド以外のトレードは無視する
                if EXCEPT_SPREAD_FLG:
                    if not (spread_tmp[i -1] in TARGET_SPREAD_LIST):
                        self.train_dict_ex[score_tmp[i]] = None
                        continue

                """
                # 指定した秒のトレードは無視する
                if len(EXCEPT_SEC_LIST) != 0:
                    target_sec = datetime.fromtimestamp(score_tmp[i]).second
                    if target_sec in EXCEPT_SEC_LIST:
                        self.train_dict_ex[score_tmp[i]] = None
                        continue
                """

                # 正解をいれていく
                bef = close_tmp[i -1]
                aft = close_tmp[i -1 + PRED_TERM]

                divide = aft / bef
                if aft == bef:
                    divide = 1

                divide = 10000 * (divide - 1)

                if test_flg == False:
                    if DIVIDE_MAX !=0 and DIVIDE_MAX < abs(divide):
                        #変化率が大きすぎる場合 外れ値とみなして除外
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

                if REAL_SPREAD_FLG:
                    spread = spread_tmp[i -1] + 1

                if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.001")) * Decimal(str(spread))):
                    # 上がった場合
                    if LEARNING_TYPE == "CATEGORY":
                        tmp_label = np.array([1, 0, 0])
                    elif LEARNING_TYPE == "CATEGORY_BIN":
                        tmp_label = np.array([1, 0])
                    up = up + 1
                elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.001")) * Decimal(str(spread))):
                    if LEARNING_TYPE == "CATEGORY":
                        tmp_label = np.array([0, 0, 1])
                    elif LEARNING_TYPE == "CATEGORY_BIN":
                        tmp_label = np.array([0, 1])
                    down = down + 1
                else:
                    if LEARNING_TYPE == "CATEGORY":
                        tmp_label = np.array([0, 1, 0])
                    elif LEARNING_TYPE == "CATEGORY_BIN":
                        #2クラス分類なのでsameはなしとする
                        continue

                if LEARNING_TYPE == "REGRESSION_SIGMA" or LEARNING_TYPE == "REGRESSION":
                    tmp_label = np.array(divide)

                if test_flg:
                    # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                    # 複数のDBを使用した場合に結果を時系列順にならべて確認するため
                    if len(self.train_dict) == 0:
                        print("first score", score_tmp[i])
                    self.train_dict[score_tmp[i]] = [tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp,
                                                     db5_index_tmp, bef, aft, spread_tmp[i -1], divide_prev, divide_aft]
                else:
                    self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp])

        print("You Can Stop Redis!!!")
        # メモリ節約のためredis停止
        if test_flg == False:
            r.shutdown()

        if test_flg:
            #一番短い足のDBを複数インプットする場合用に
            #score順にならべかえてtrain_listと正解ラベルおよびレート変化幅(aft-bef)をそれぞれリスト化する
            for data in sorted(self.train_dict.items()):
                self.train_list.append(data[1][:6])
                self.correct_list.append(data[1][0])
                self.pred_close_list.append(data[1][6])
                self.real_close_list.append(data[1][7])
                self.target_spread_list.append(data[1][8])
                self.target_divide_prev_list.append(data[1][9])
                self.target_divide_aft_list.append(data[1][10])
                self.train_score_list.append(data[0])

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

        del self.db1, self.db2, self.db3, self.db4, self.db5

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
        if REAL_SPREAD_FLG:
            print("spread total: ", self.spread_cnt)
            if self.spread_cnt != 0:
                for k, v in sorted(self.spread_cnt_dict.items()):
                    print(k, v / self.spread_cnt)

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

        self.create_db_all = np.vectorize(tmp_create_db_all, otypes=[np.ndarray])



        def tmp_create_label(x, start_idx):
            return self.train_list[start_idx + x][0]

        self.create_label = np.vectorize(tmp_create_label, otypes=[np.ndarray])


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

        if METHOD == "LSTM" or METHOD == "BY":
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

        elif METHOD == "NORMAL":
            for i in range(len(INPUT_LEN)):

                # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
                tmp_arr = self.create_db(tmp_np, start_idx, i + 1)
                tmp_arr = tmp_arr.tolist()
                tmp_arr = np.array(tmp_arr)

                retX.append(tmp_arr)

        """
        if idx == 0:
            if len(INPUT_LEN) == 3:
                for i in range(3):
                    print("X1 :", retX[0][i][-5:])
                    print("X2 :", retX[1][i][-5:])
                    print("X3 :", retX[2][i][-5:])
                    print("Y :", retY[i])
        """
        #テストの場合はテストデータのみ返す
        if self.test_flg:
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
