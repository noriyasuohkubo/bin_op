from keras.utils import Sequence
from pathlib import Path
import pandas
import numpy as np
from keras.utils import np_utils
from datetime import datetime
import time
from decimal import Decimal
from scipy.ndimage.interpolation import shift
import redis
import json
import random
from readConf import *
import scipy.stats
import gc
import math

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

class DataSequence(Sequence):

    def __init__(self, rec_num, startDt, endDt, test_flg):
        #コンストラクタ
        self.rec_num = rec_num
        # 学習対象のみのself.closeのインデックスと正解ラベルが入った子リストを保持する。このインデックスを元にself.closeから指定期間分のデータと正解ラベルを取得する
        self.train_list = []
        self.start_score = int(time.mktime(startDt.timetuple()))
        self.end_score = int(time.mktime(endDt.timetuple()))
        self.test_flg = test_flg
        self.features = {}
        self.features_tmp = {}
        self.time_list = []

        for i in range(len(in_features)):
            self.features["feature_" + str(i)] = []
            self.features_tmp["feature_tmp_" + str(i)] = []

        self.longers = {}
        self.longers_index = {}

        for i in range(len(in_longers)):
            self.longers["longer_" + str(i)] = []
            # 一分足のスコアがkey、リストのインデックスがvalueの辞書を中身にもつ
            #後で短い足から直前の長い足のデータをひっぱってくるために使用
            self.longers_index["longer_" + str(i)] = {}
        up = 0
        same = 0

        r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

        # 長い足を使用する場合は先にデータ取得する
        for i, longer in enumerate(in_longers):
            # take all data
            result_longer = r.zrange(in_longers_db[longer], 0, -1, withscores=True)
            print("result_longer length:" ,len(result_longer))

            for j, line_longer in enumerate(result_longer):
                body_longer = line_longer[0]
                score_longer = int(line_longer[1])
                tmps_longer = json.loads(body_longer)

                self.longers["longer_" + str(i)].append(tmps_longer.get("close_divide"))
                self.longers_index["longer_" + str(i)][score_longer] = j

            print(self.longers["longer_" + str(i)][:10])

            #試しにスコア表示
            """
            tmp_i = 0
            for score, list_ind in self.longers_index["longer_" + str(i)].items():
                print("score:", score, " index:", list_ind)
                tmp_i = tmp_i + 1
                if tmp_i > 10:
                    break
            """

        for sbl in symbols:
            list_start_idx = len(self.features["feature_0"])
            print("symbol:", sbl)
            print("list_start_idx:",list_start_idx)

            if test_flg == False:
                #lstm_generator用
                result = r.zrevrangebyscore(sbl, self.start_score, self.end_score, start=0, num=self.rec_num + 1)
                result.reverse()
                print("start_score:" + str(self.start_score))
            else:
                #testLstm用
                result = r.zrangebyscore(sbl, self.start_score, self.end_score)
            print("DataSequence:get redis data");
            close_tmp, time_tmp = [], []
            feature_1,feature_2,feature_3,feature_4= [],[],[],[]
            longers_score_tmp = {}
            for i in range(len(in_longers)):
                longers_score_tmp["longer_score_" + str(i)] = []

            for line in result:
                tmps = json.loads(line)
                #closeはask,bidの仲値がDBに入っている
                close_tmp.append(tmps.get("close"))
                #close_tmp.append(tmps.get("open"))
                time_tmp.append(tmps.get("time"))

                for i,feature in enumerate(in_features):
                    self.features_tmp["feature_tmp_" + str(i)].append(tmps.get(feature))

                for i, longer in enumerate(in_longers):
                    longers_score_tmp["longer_score_" + str(i)].append(tmps.get(longer))

            # メモリ解放
            del result
            gc.collect()
            #self.time_list.extend(time_tmp)

            for i, feature in enumerate(in_features):
                self.features["feature_" + str(i)].extend(self.features_tmp["feature_tmp_" + str(i)])

            print(close_tmp[-10:])
            print(time_tmp[0:10])
            print(time_tmp[-10:])
            for i, longer in enumerate(in_longers):
                print(longers_score_tmp["longer_score_" + str(i)][0:10])

            tmp_data_length = len(close_tmp) - close_shift - (maxlen * close_shift) - (pred_term * close_shift)
            list_start_idx = list_start_idx -1
            for i in range(tmp_data_length):
                #学習対象closeのインデックスを保持
                list_start_idx = list_start_idx + 1
                #print(list_start_idx)
                #ハイローオーストラリアの取引時間外を学習対象からはずす(lstm_generator用のみ)
                if test_flg == False:
                    if except_highlow:
                        if datetime.strptime(time_tmp[i + (maxlen * close_shift) - 1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                            continue;
                # maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
                tmp_time_bef = datetime.strptime(time_tmp[i], '%Y-%m-%d %H:%M:%S')
                tmp_time_aft = datetime.strptime(time_tmp[i + (maxlen * close_shift) - 1], '%Y-%m-%d %H:%M:%S')
                delta = tmp_time_aft - tmp_time_bef

                if delta.total_seconds() >= (maxlen * int(s)):
                    # print(tmp_time_aft)
                    continue;

                # sよりmergの方が大きい数字の場合、
                # 検証時(testLstm.py)は秒をmergで割った余りが0のデータだけを使って結果をみる、なぜならDB内データの間隔の方がトレードタイミングより短いため
                if test_flg == True:
                    if int(s) < int(merg):
                        sec = time_tmp[i][-2:]
                        if Decimal(str(sec)) % Decimal(merg) != 0:
                            continue
                # 学習時、使用するデータセットを絞る
                if test_flg == False:
                    if len(data_set) != 0:
                        use_flg = False
                        for tmp_sec in data_set:
                            sec = time_tmp[i][-2:]
                            if close_shift > 1:
                                if Decimal(str(sec)) % Decimal(s) == tmp_sec:
                                    use_flg = True
                                    break
                            else:
                                if Decimal(str(sec)) % Decimal(merg) == tmp_sec:
                                    use_flg = True
                                    break
                        if use_flg == False:
                            #データ使用しない場合
                            continue

                # 正解をいれていく
                bef = close_tmp[i + (maxlen * close_shift) -1]
                aft = close_tmp[i + (maxlen * close_shift) + (pred_term * close_shift) -1]

                tmp_label = None
                if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
                    # 上がった場合
                    tmp_label = np.array([1, 0, 0])
                    up = up + 1
                elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
                    tmp_label = np.array([0, 0, 1])
                else:
                    tmp_label = np.array([0, 1, 0])
                    same = same + 1

                #直前の長い足のスコアを入れていく
                longer_scores = []
                for j, longer in enumerate(in_longers):
                    longer_scores.append(longers_score_tmp["longer_score_" + str(j)][i + (maxlen * close_shift) -1])

                self.train_list.append([list_start_idx, tmp_label, longer_scores])

        # メモリ解放
        del close_tmp
        del time_tmp
        del longers_score_tmp
        del self.features_tmp
        gc.collect()

        for i, feature in enumerate(self.features):
            self.features["feature_" + str(i)] = np.array(self.features["feature_" + str(i)])

        self.cut_num = len(self.train_list) % batch_size
        #print("tmp train list length: " , str(len(self.train_list)))
        #print("cut_num: " , str(cut_num))

        #self.train_list = self.train_list[cut_num:]
        print("train list length: " , str(len(self.train_list)))

        self.data_length = len(self.train_list)
        if self.cut_num !=0:
            # エポック毎の学習ステップ数
            # バッチサイズが1ステップで学習するデータ数
            # self.cut_num(総学習データ÷バッチサイズ)で余りがあるなら余りがある分ステップ数が1多い
            self.steps_per_epoch = int(self.data_length / batch_size) +1
        else:
            self.steps_per_epoch = int(self.data_length / batch_size)

        print("steps_per_epoch: ", self.steps_per_epoch)
        myLogger("UP: ",up/self.data_length)
        myLogger("SAME: ", same / self.data_length)
        myLogger("DOWN: ", (self.data_length - up - same) / self.data_length)

        print(self.train_list[0:10])
        #print(self.close[self.train_list[0][0]:(self.train_list[0][0] + self.maxlen)])
        #print(self.train_list[0][1])

        #シャッフルしとく
        if test_flg == False:
            random.shuffle(self.train_list)

        def tmp_create_features(x, start_idx, maxlen,feature_num):
            # 取得すべきself.featuresのインデックス番号を決定
            target_idx = self.train_list[start_idx + x][0]
            """
            if (start_idx + x) == 0:
                tmp_dt = self.time_list[target_idx]
                print("time", self.time_list[target_idx])
                tdatetime = datetime.strptime(tmp_dt, '%Y-%m-%d %H:%M:%S')
                print("score", int(time.mktime(tdatetime.timetuple())))

                tmp_dt = self.time_list[target_idx + (maxlen * close_shift) -1]
                print("time", self.time_list[target_idx + (maxlen * close_shift) -1])
                tdatetime = datetime.strptime(tmp_dt, '%Y-%m-%d %H:%M:%S')
                print("score", int(time.mktime(tdatetime.timetuple())))
            """
            return self.features["feature_" + str(feature_num)][target_idx:(target_idx + (maxlen * close_shift)):close_shift]

        self.create_features = np.vectorize(tmp_create_features, otypes=[np.ndarray])

        def tmp_create_longers(x, start_idx, maxlen,longer_num):
            # 取得すべき長い足のインデックス番号を探すためのスコアを取得
            target_score = self.train_list[start_idx + x][2][longer_num]
            """
            if (start_idx + x) == 0:
                print("target_score:",target_score)
            """
            # 長い足のデータのインデックスを取得
            target_index =  self.longers_index["longer_" + str(longer_num)][target_score]
            return   self.longers["longer_" + str(longer_num)][(target_index + 1 - (maxlen * close_shift)):(target_index + 1):close_shift]

        self.create_longers = np.vectorize(tmp_create_longers, otypes=[np.ndarray])

        def tmp_create_close(x, start_idx, maxlen):
            # 取得すべきself.closeのインデックス番号を決定
            target_idx = self.train_list[start_idx + x][0]

            return self.close[target_idx:(target_idx + (maxlen * close_shift)):close_shift]

        self.create_close = np.vectorize(tmp_create_close, otypes=[np.ndarray])

        def tmp_create_spread(x, start_idx, maxlen):
            target_idx = self.train_list[start_idx + x][0]

            return self.spreads[target_idx:(target_idx + + (maxlen * close_shift)):close_shift]

        self.create_spread = np.vectorize(tmp_create_spread, otypes=[np.ndarray])

        def tmp_create_high(x, start_idx, maxlen):
            target_idx = self.train_list[start_idx + x][0]

            return self.high[target_idx:(target_idx + (maxlen * close_shift)):close_shift]

        self.create_high = np.vectorize(tmp_create_high, otypes=[np.ndarray])

        def tmp_create_low(x, start_idx, maxlen):
            target_idx = self.train_list[start_idx + x][0]

            return self.low[target_idx:(target_idx + (maxlen * close_shift)):close_shift]

        self.create_low = np.vectorize(tmp_create_low, otypes=[np.ndarray])

        def tmp_create_label(x, start_idx):
            return self.train_list[start_idx + x][1]

        # frompyfuncで新たな関数を作成
        self.create_label = np.vectorize(tmp_create_label, otypes=[np.ndarray])

        def f(x):
            return x * np.array([1, 1, 1, 1, 1], dtype=np.float32)

        self.g = np.vectorize(f, otypes=[np.ndarray])



    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        #start = time.time()
        # データの取得実装
        #print("idx:", idx)
        tmp_np = np.arange(batch_size)
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
        start_idx = idx * batch_size

        label_data_tmp= self.create_label(tmp_np, start_idx)
        #print(label_data_tmp[:2])
        #print(type(label_data_tmp))
        new_label_data = label_data_tmp.tolist()
        label_data = np.array(new_label_data)
        #print(label_data[:2])

        if functional_flg:
            retX_sec = np.zeros((len(tmp_np), maxlen, 1))
            for i, feature in enumerate(self.features):
                # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
                feature_tmp_arr = self.create_features(tmp_np, start_idx, maxlen, i)
                new_feature_data = feature_tmp_arr.tolist()
                feature_data = np.array(new_feature_data)
                retX_sec[:, :, 0] = feature_data[:]

            retX_min = np.zeros((len(tmp_np), maxlen_min, 1))
            for i, longer in enumerate(self.longers):
                longer_tmp_arr = self.create_longers(tmp_np, start_idx, maxlen_min, i)
                new_longer_data = longer_tmp_arr.tolist()
                longer_data = np.array(new_longer_data)
                retX_min[:, :, 0] = longer_data[:]

            retX = [retX_sec, retX_min]

        else:
            retX = np.zeros((len(tmp_np), maxlen, x_length))
            x_cnt = 0;
            for i,feature in enumerate(self.features):
                # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
                feature_tmp_arr = self.create_features(tmp_np, start_idx, maxlen,i)
                new_feature_data = feature_tmp_arr.tolist()
                feature_data = np.array(new_feature_data)
                retX[:, :, x_cnt] = feature_data[:]
                x_cnt = x_cnt + 1

            for i, longer in enumerate(self.longers):
                longer_tmp_arr = self.create_longers(tmp_np, start_idx, maxlen, i)
                new_longer_data = longer_tmp_arr.tolist()
                longer_data = np.array(new_longer_data)
                retX[:, :, x_cnt] = longer_data[:]
                x_cnt = x_cnt + 1

        retY = np.array(label_data)
        """
        if idx == 0:
            print("X SHAPE:", retX.shape)
            print("Y SHAPE:", retY.shape)
            print("X :", retX[0:1])
            print("Y :", retY[0:1])
        """
        #elapsed_time = time.time() - start
        #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
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
        #print(self.train_list[0:10])

    def get_data_length(self):

        return self.data_length