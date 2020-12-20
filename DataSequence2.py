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
        self.change_list = [] #test用 FXでの報酬計算に使用 レートの変化幅を保持
        self.score_list = [] #test用
        self.score_dict = {} #test用
        self.close_list = [] #test用
        self.train_score_list = [] #test用 予想対象のスコアを保持

        r = redis.Redis(host='localhost', port=6379, db=DB_NO, decode_responses=True)

        # 長い足のDBから先に全件取得
        db2_index = 0
        for i, db in enumerate(DB2_LIST):
            result = r.zrange(db, 0, -1, withscores=True)
            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                self.db2.append(tmps.get("d"))
                self.db2_score[score] = db2_index
                self.db2_score_list.append(score)

                db2_index += 1

            del result

        db3_index = 0
        for i, db in enumerate(DB3_LIST):
            result = r.zrange(db, 0, -1, withscores=True)
            print(db ,len(result))

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                self.db3.append(tmps.get("d"))
                self.db3_score[score] = db3_index
                self.db3_score_list.append(score)

                db3_index += 1

            del result

        up = 0
        same = 0

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

            close_tmp, devide_tmp, score_tmp = [], [], []

            for line in result:
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)

                close_tmp.append(tmps.get("c"))
                self.db1.append(tmps.get("d"))

                score_tmp.append(score)

                self.db1_score[score] = db1_index
                self.db1_score_list.append(score)

                #test用にscoreをキーにレートを保持
                if test_flg:
                    self.score_dict[score] = tmps.get("c")

                db1_index += 1

            del result

            print(datetime.fromtimestamp(min(self.db1_score)))
            print(datetime.fromtimestamp(max(self.db1_score)))

            list_idx = list_idx -1

            for i in range(len(close_tmp)):
                list_idx += 1

                #inputデータが足りない場合スキップ
                if i < INPUT_LEN[0]:
                    continue

                try:
                    start_score = score_tmp[i - INPUT_LEN[0]]
                    end_score = score_tmp[i + PRED_TERM -1]
                    if end_score != start_score + ((INPUT_LEN[0] + PRED_TERM - 1)  * DB1_TERM):
                        #時刻がつながっていないものは除外 たとえば日付またぎなど
                        continue

                except IndexError:
                    #start_score end_scoreのデータなしなのでスキップ
                    continue

                db2_index_tmp = -1

                #DB2を使う場合
                if len(INPUT_LEN) > 1:
                    db2_index_tmp = self.db2_score[score_tmp[i]] #scoreからインデックスを取得
                    try:
                        start_score = self.db2_score_list[db2_index_tmp - INPUT_LEN[1]]
                        end_score = self.db2_score_list[db2_index_tmp]
                        if end_score != start_score + (INPUT_LEN[1] * DB2_TERM):
                            #時刻がつながっていないものは除外 たとえば日付またぎなど
                            continue

                    except IndexError:
                        #start_scoreのデータなしなのでスキップ
                        continue

                db3_index_tmp = -1

                # DB3を使う場合
                if len(INPUT_LEN) > 2:
                    db3_index_tmp = self.db3_score[score_tmp[i]] #scoreからインデックスを取得
                    try:
                        start_score = self.db3_score_list[db3_index_tmp - INPUT_LEN[2]]
                        end_score = self.db3_score_list[db3_index_tmp]
                        if end_score != start_score + (INPUT_LEN[2] * DB3_TERM):
                            # 時刻がつながっていないものは除外 たとえば日付またぎなど
                            continue

                    except IndexError:
                        # start_scoreのデータなしなのでスキップ
                        continue

                #ハイローオーストラリアの取引時間外を学習対象からはずす
                if len(EXCEPT_LIST) != 0:
                    if datetime.fromtimestamp(score_tmp[i]).hour in EXCEPT_LIST:
                        continue;

                # 正解をいれていく
                bef = close_tmp[i -1]
                aft = close_tmp[i -1 + PRED_TERM]

                change = aft - bef #変化幅を保持 test用

                divide = aft / bef
                if aft == bef:
                    divide = 1

                divide = 10000 * (divide - 1)

                if test_flg == False:
                    if DIVIDE_MAX < abs(divide):
                        #変化率が大きすぎる場合 外れ値とみなして除外
                        continue

                tmp_label = None
                if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.001")) * Decimal(str(SPREAD))):
                    # 上がった場合
                    tmp_label = np.array([1, 0, 0])
                    up = up + 1
                elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.001")) * Decimal(str(SPREAD))):
                    tmp_label = np.array([0, 0, 1])
                else:
                    tmp_label = np.array([0, 1, 0])
                    same = same + 1

                if test_flg:
                    # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                    # 複数のDBを使用した場合に結果を時系列順にならべて確認するため
                    if len(self.train_dict) == 0:
                        print("first score", score_tmp[i])
                    self.train_dict[score_tmp[i]] = [tmp_label, list_idx, db2_index_tmp, db3_index_tmp, change]
                else:
                    self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp])

        # メモリ節約のためredis停止
        #r.shutdown()

        print("You Can Stop Redis!!!")

        if test_flg:
            #一番短い足のDBを複数インプットする場合用に
            #score順にならべかえてtrain_listと正解ラベルおよびレート変化幅(aft-bef)をそれぞれリスト化する
            ######## 動作未確認 複数インプットをする場合に確認すること！！！！！ ######
            for data in sorted(self.train_dict.items()):
                self.train_list.append(data[1][:4])
                self.correct_list.append(data[1][0])
                self.change_list.append(data[1][4])
                self.train_score_list.append(data[0])

            #一番短い足のDBを複数インプットする場合用に
            #scoreのリストcloseのリストを作成 結果確認のグラフ描写で使うため
            for data in sorted(self.score_dict.items()):
                self.score_list.append(data[0])
                self.close_list.append(data[1])

            del self.train_dict, self.score_dict

        # メモリ解放
        del close_tmp
        del score_tmp

        #numpy化してsliceを早くする
        self.db1_np = np.array(self.db1)
        self.db2_np = np.array(self.db2)
        self.db3_np = np.array(self.db3)

        del self.db1, self.db2, self.db3

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
        myLogger("SAME: ", same /self.data_length)
        myLogger("DOWN: ", (self.data_length - up - same) / (self.data_length))

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

        self.create_db1 = np.vectorize(tmp_create_db1, otypes=[np.ndarray])

        def tmp_create_db2(x, start_idx):
            # 取得すべきdb2のインデックス番号を決定
            target_idx = self.train_list[start_idx + x][2]

            return self.db2_np[target_idx - INPUT_LEN[1]:target_idx]

        self.create_db2 = np.vectorize(tmp_create_db2, otypes=[np.ndarray])

        def tmp_create_db3(x, start_idx):
            # 取得すべきdb3のインデックス番号を決定
            target_idx = self.train_list[start_idx + x][3]

            return self.db3_np[target_idx - INPUT_LEN[2]:target_idx]

        self.create_db3 = np.vectorize(tmp_create_db3, otypes=[np.ndarray])

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

        if len(INPUT_LEN) == 1:
            retX = np.zeros((len(tmp_np), INPUT_LEN[0], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr = self.create_db1(tmp_np, start_idx)
            tmp_arr = tmp_arr.tolist()
            tmp_arr = np.array(tmp_arr)
            retX[:, :, 0] = tmp_arr[:]

            retX = [retX, ]

        elif len(INPUT_LEN) == 2:
            retX1 = np.zeros((len(tmp_np), INPUT_LEN[0], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr1 = self.create_db1(tmp_np, start_idx)
            tmp_arr1 = tmp_arr1.tolist()
            tmp_arr1 = np.array(tmp_arr1)
            retX1[:, :, 0] = tmp_arr1[:]

            retX2 = np.zeros((len(tmp_np), INPUT_LEN[1], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr2 = self.create_db2(tmp_np, start_idx)
            tmp_arr2 = tmp_arr2.tolist()
            tmp_arr2 = np.array(tmp_arr2)
            retX2[:, :, 0] = tmp_arr2[:]

            retX = [retX1, retX2 ]

        elif len(INPUT_LEN) == 3:
            retX1 = np.zeros((len(tmp_np), INPUT_LEN[0], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr1 = self.create_db1(tmp_np, start_idx)
            tmp_arr1 = tmp_arr1.tolist()
            tmp_arr1 = np.array(tmp_arr1)
            retX1[:, :, 0] = tmp_arr1[:]

            retX2 = np.zeros((len(tmp_np), INPUT_LEN[1], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr2 = self.create_db2(tmp_np, start_idx)
            tmp_arr2 = tmp_arr2.tolist()
            tmp_arr2 = np.array(tmp_arr2)
            retX2[:, :, 0] = tmp_arr2[:]

            retX3 = np.zeros((len(tmp_np), INPUT_LEN[2], 1))

            # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
            tmp_arr3 = self.create_db3(tmp_np, start_idx)
            tmp_arr3 = tmp_arr3.tolist()
            tmp_arr3 = np.array(tmp_arr3)
            retX3[:, :, 0] = tmp_arr3[:]

            retX = [retX1, retX2, retX3]

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
        random.shuffle(self.train_list)

    def get_data_length(self):
        return self.data_length

    def get_correct_list(self):
        retY = np.array(self.correct_list)
        return retY

    def get_change_list(self):
        return self.change_list

    def get_score_list(self):
        return self.score_list

    def get_close_list(self):
        return self.close_list

    def get_train_score_list(self):
        return self.train_score_list
