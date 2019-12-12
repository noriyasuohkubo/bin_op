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

class DataSequence(Sequence):

    def __init__(self, maxlen, pred_term, s, in_num, batch_size, symbol, spread, rec_num, symbols, startDt, endDt, test_flg):
        #コンストラクタ
        self.maxlen = maxlen
        self.pred_term = pred_term
        self.s = s
        self.rec_num = rec_num
        self.in_num = in_num
        self.batch_size = batch_size
        self.symbol = symbol
        self.spread = spread
        self.symbols = symbols
        self.close = []
        self.spreads = []
        self.train_list = []
        self.start_score = int(time.mktime(startDt.timetuple()))
        self.end_score = int(time.mktime(endDt.timetuple()))
        self.test_flg = test_flg

        up = 0
        same = 0

        r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

        for sbl in symbols:
            list_start_idx = len(self.close)
            print("symbol:", sbl)
            print("list_start_idx:",list_start_idx)

            if rec_num != 0:
                #lstm_generator用
                result = r.zrevrangebyscore(sbl, self.start_score, self.end_score, start=0, num=self.rec_num + 1)
                result.reverse()
                print(self.start_score)
            else:
                #testLstm用
                result = r.zrangebyscore(sbl, self.start_score, self.end_score)

            print("DataSequence:get redis data");
            close_tmp, time_tmp, spread_tmp = [], [], []
            for line in result:
                tmps = json.loads(line)
                #askとbid(close)の仲値にする。ハイローに合わせる
                mid = float((Decimal(str(tmps.get("close"))) + Decimal(str(tmps.get("ask")))) / Decimal("2"))
                close_tmp.append(mid)
                time_tmp.append(tmps.get("time"))

                if rec_num != 0 :
                    # lstm_generator用 スプレッドを計算し保存
                    spread_tmp.append(float(Decimal(str(tmps.get("ask"))) - Decimal(str(mid))))

            #ask: 1.50582 の形でDBに入っている
            self.close.extend(10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:])
            #self.spreads.extend(10000 * np.log(spread_tmp / shift(spread_tmp, 1, cval=np.NaN))[1:])
            if in_num != 1:
                #スプレッドを特徴量としてインプットする場合
                self.spreads.extend(scipy.stats.zscore(spread_tmp)[1:])
            time_tmp.append(tmps.get("time"))
            if rec_num != 0:
            # lstm_generator用
                print("スプレッド平均 ",np.mean(spread_tmp))
                print("スプレッド標準偏差 ", np.std(spread_tmp))

            print("close len:", len(self.close))
            print(self.close[-10:])
            print(close_tmp[-10:])
            print(time_tmp[0:10])
            print(time_tmp[-10:])

            tmp_data_length = len(close_tmp) - 1 - maxlen - pred_term -1
            list_start_idx = list_start_idx -1
            for i in range(tmp_data_length):
                list_start_idx = list_start_idx + 1
                #print(list_start_idx)
                #ハイローオーストラリアの取引時間外を学習対象からはずす(lstm_generator用のみ)
                if rec_num != 0:
                    if except_highlow:
                        if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                            continue;
                # maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
                tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
                tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen - 1], '%Y-%m-%d %H:%M:%S')
                delta = tmp_time_aft - tmp_time_bef

                if delta.total_seconds() > ((maxlen - 1) * int(s)):
                    # print(tmp_time_aft)
                    continue;

                if except_low_spread:
                    if spread_tmp[1 + i + maxlen - 1] <= limit_spread:
                        continue

                bef = close_tmp[1 + i + maxlen -1]
                aft = close_tmp[1 + i + maxlen + pred_term -1]

                tmp_label = None
                if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
                    # 上がった場合
                    tmp_label = [1, 0, 0]
                    up = up + 1
                elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
                    tmp_label = [0, 0, 1]
                else:
                    tmp_label = [0, 1, 0]
                    same = same + 1

                self.train_list.append([list_start_idx, tmp_label])

        self.cut_num = len(self.train_list) % self.batch_size
        #print("tmp train list length: " , str(len(self.train_list)))
        #print("cut_num: " , str(cut_num))

        #self.train_list = self.train_list[cut_num:]
        print("train list length: " , str(len(self.train_list)))

        self.data_length = len(self.train_list)
        if self.cut_num !=0:
            self.steps_per_epoch = int(self.data_length / self.batch_size) +1
        else:
            self.steps_per_epoch = int(self.data_length / self.batch_size)

        print("steps_per_epoch: ", self.steps_per_epoch)
        print("UP: ",up/self.data_length)
        print("SAME: ", same / self.data_length)
        print("DOWN: ", (self.data_length - up - same) / self.data_length)

        print(self.train_list[0:10])
        #print(self.close[self.train_list[0][0]:(self.train_list[0][0] + self.maxlen)])
        #print(self.train_list[0][1])

        def tmp_create_close(x, start_idx, maxlen):
            target_idx = self.train_list[start_idx + x][0]

            return self.close[target_idx:(target_idx + maxlen)]

        self.create_close = np.vectorize(tmp_create_close, otypes=[np.ndarray])

        def tmp_create_spread(x, start_idx, maxlen):
            target_idx = self.train_list[start_idx + x][0]

            return self.spreads[target_idx:(target_idx + maxlen)]

        self.create_spread = np.vectorize(tmp_create_spread, otypes=[np.ndarray])

        def tmp_create_label(x, start_idx):
            return self.train_list[start_idx + x][1]

        # frompyfuncで新たな関数を作成
        self.create_label = np.vectorize(tmp_create_label, otypes=[np.ndarray])

        def f(x):
            return x * np.array([1, 1, 1, 1, 1], dtype=np.float32)

        self.g = np.vectorize(f, otypes=[np.ndarray])

    def __getitem__(self, idx):
        #start = time.time()
        # データの取得実装
        #print("idx:", idx)
        tmp_np = np.arange(self.batch_size)
        if idx == (self.steps_per_epoch -1) and self.cut_num != 0:
            tmp_np = np.arange(self.cut_num)
        #print("tmp_np:", tmp_np)
        #tmp_np = np.arange(3)
        #tmp = self.g(tmp_np)
        #new_tmp = tmp.tolist()
        #print(np.array(new_tmp).shape)
        start_idx = idx * self.batch_size
        close_data= self.create_close(tmp_np, start_idx, self.maxlen)
        new_close_data = close_data.tolist()
        close_data = np.array(new_close_data)
        #print(new_close_data)

        label_data= self.create_label(tmp_np, start_idx)
        new_label_data = label_data.tolist()
        label_data = np.array(new_label_data)
        #print(close_data)

        """
        close_data, label_data = [], []
        #print("idx:"+ str(idx))
        start_idx = idx * self.batch_size
        for i in range(self.batch_size):
            target_idx = self.train_list[start_idx + i][0]
            label_data.append(self.train_list[start_idx + i][1])

            close_data.append(self.close[target_idx:(target_idx + self.maxlen)])


        close_np = np.array(close_data)
        """
        retX = np.zeros((len(close_data), self.maxlen, self.in_num))
        retX[:, :, 0] = close_data[:]
        if in_num == 2:
            # Spreadも入力データとする場合
            spread_data = self.create_spread(tmp_np, start_idx, self.maxlen)
            new_spread_data = spread_data.tolist()
            spread_data = np.array(new_spread_data)
            retX[:, :, 1] = spread_data[:]
        retY = np.array(label_data)
        """
        if idx == 0:
            print("X SHAPE:", retX.shape)
            print("Y SHAPE:", retY.shape)
            print("X :", retX[0:10])
            print("Y :", retY[0:10])
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