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

class DataSequence(Sequence):

    def __init__(self, maxlen, pred_term, s, in_num, batch_size, symbol, spread, rec_num, symbols, startDt, endDt, test_flg):
        #コンストラクタ
        self.maxlen = maxlen
        self.pred_term = pred_term
        self.s = s
        self.rec_num = rec_num
        self.in_num = in_num #2ならスプレッド 3ならhigh,lowを学習対象に加える
        self.batch_size = batch_size
        self.symbol = symbol
        self.spread = spread
        self.symbols = symbols
        self.close = []
        self.high = []
        self.low = []
        self.spreads = []
        # 学習対象のみのself.closeのインデックスと正解ラベルが入った子リストを保持する。このインデックスを元にself.closeから指定期間分のデータと正解ラベルを取得する
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
                print("start_score:" + str(self.start_score))
            else:
                #testLstm用
                result = r.zrangebyscore(sbl, self.start_score, self.end_score)
            print("DataSequence:get redis data");
            close_tmp, time_tmp, spread_tmp = [], [], []
            high_tmp, low_tmp = [], []
            for line in result:
                tmps = json.loads(line)
                #closeはask,bidの仲値がDBに入っている
                close_tmp.append(tmps.get("close"))
                time_tmp.append(tmps.get("time"))

                """
                if rec_num != 0:
                    # lstm_generator用 スプレッドを計算し保存
                    if in_num == 2:
                        spread_tmp.append(float(Decimal(str(tmps.get("ask"))) - Decimal(str(tmps.get("ask")))))
                """
                if in_num == 3:
                    high_tmp.append(tmps.get("high"))
                    low_tmp.append(tmps.get("low"))

            # メモリ解放
            del result
            gc.collect()

            #ask: 1.50582 の形でDBに入っている
            #close_shift分前の終値との変化率を特徴量とする
            tmp_array = []
            tmp_high_array, tmp_low_array = [], []
            for i, v in enumerate(close_tmp):
                if i < close_shift:
                    continue
                divide = close_tmp[i] / close_tmp[i - close_shift]
                if close_tmp[i] == close_tmp[i - close_shift]:
                    divide = 1
                tmp_array.append(divide)

                if in_num == 3:
                    if close_shift > 1:
                        # シフトが1より多い場合、シフトするデータの中で一番の高値と安値を求める
                        tmp_high = 0
                        tmp_low = 10000 #適当に10000
                        for j in np.arange(close_shift):
                            if tmp_high < high_tmp[i - j]:
                                tmp_high = high_tmp[i - j]


                            if tmp_low > low_tmp[i - j] :
                                tmp_low = low_tmp[i - j]

                        tmp_high_array.append(close_tmp[i] / tmp_high)
                        tmp_low_array.append(close_tmp[i] / tmp_low)
                    else:
                        tmp_high_array.append(close_tmp[i] / high_tmp[i])
                        tmp_low_array.append(close_tmp[i] / low_tmp[i])

            self.close.extend(10000 * np.log(tmp_array))
            #self.close.extend(10000 * np.log(close_tmp/shift(close_tmp, close_shift, cval=np.NaN) )[close_shift:])
            if in_num == 3:
                self.high.extend(tmp_high_array)
                self.low.extend(tmp_low_array)
            #self.spreads.extend(10000 * np.log(spread_tmp / shift(spread_tmp, 1, cval=np.NaN))[1:])
            """
            if rec_num != 0:
                if in_num == 2:
                    #スプレッドを特徴量としてインプットする場合
                    self.spreads.extend(scipy.stats.zscore(spread_tmp)[close_shift:])
            """
            #time_tmp.append(tmps.get("time"))
            """
            if rec_num != 0:
            # lstm_generator用
                if in_num == 2:
                    print("スプレッド平均 ",np.mean(spread_tmp))
                    print("スプレッド標準偏差 ", np.std(spread_tmp))
            """
            print("close len:", len(self.close))
            print(self.close[-10:])
            print(close_tmp[-10:])
            print(time_tmp[0:10])
            print(time_tmp[-10:])

            tmp_data_length = len(close_tmp) - close_shift - (maxlen * close_shift) - (pred_term * close_shift) -1
            list_start_idx = list_start_idx -1
            for i in range(tmp_data_length):
                #学習対象closeのインデックスを保持
                list_start_idx = list_start_idx + 1
                #print(list_start_idx)
                #ハイローオーストラリアの取引時間外を学習対象からはずす(lstm_generator用のみ)
                if rec_num != 0:
                    if except_highlow:
                        if datetime.strptime(time_tmp[close_shift + i + (maxlen * close_shift) -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                            continue;
                # maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
                tmp_time_bef = datetime.strptime(time_tmp[close_shift + i], '%Y-%m-%d %H:%M:%S')
                tmp_time_aft = datetime.strptime(time_tmp[close_shift + i + (maxlen * close_shift) - 1], '%Y-%m-%d %H:%M:%S')
                delta = tmp_time_aft - tmp_time_bef

                if delta.total_seconds() >= (maxlen * int(s)):
                    # print(tmp_time_aft)
                    continue;
                """
                if except_low_spread:
                    if spread_tmp[close_shift + i + (maxlen * close_shift) - 1] <= limit_spread:
                        continue
                """
                # sよりmergの方が大きい数字の場合、
                # 検証時(testLstm.py)は秒をmergで割った余りが0のデータだけを使って結果をみる、なぜならDB内データの間隔の方がトレードタイミングより短いため
                if rec_num == 0:
                    if int(s) < int(merg):
                        sec = time_tmp[close_shift + i][-2:]
                        if Decimal(str(sec)) % Decimal(merg) != 0:
                            continue
                # 学習時、使用するデータセットを絞る
                if rec_num != 0:
                    if len(data_set) != 0:
                        use_flg = False
                        for tmp_sec in data_set:
                            sec = time_tmp[close_shift + i][-2:]
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
                bef = close_tmp[close_shift + i + (maxlen * close_shift) -1]
                aft = close_tmp[close_shift + i + (maxlen * close_shift) + (pred_term * close_shift) -1]

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

                self.train_list.append([list_start_idx, tmp_label])

        # メモリ解放
        del close_tmp
        del time_tmp
        if in_num == 2:
            del spread_tmp
        if in_num == 3:
            del high_tmp
            del low_tmp
        gc.collect()

        self.close = np.array(self.close)
        if in_num == 3:
            self.high = np.array(self.high)
            self.low = np.array(self.low)
        #self.spreads = np.array(self.spreads)
        self.cut_num = len(self.train_list) % self.batch_size
        #print("tmp train list length: " , str(len(self.train_list)))
        #print("cut_num: " , str(cut_num))

        #self.train_list = self.train_list[cut_num:]
        print("train list length: " , str(len(self.train_list)))

        self.data_length = len(self.train_list)
        if self.cut_num !=0:
            # エポック毎の学習ステップ数
            # バッチサイズが1ステップで学習するデータ数
            # self.cut_num(総学習データ÷バッチサイズ)で余りがあるなら余りがある分ステップ数が1多い
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
        tmp_np = np.arange(self.batch_size)
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
        start_idx = idx * self.batch_size
        # 取得開始インデックスから返却すべきデータ数(tmp_npの長さ)を取得する
        close_data_tmp= self.create_close(tmp_np, start_idx, self.maxlen)
        #print(close_data_tmp[:2])
        new_close_data = close_data_tmp.tolist()
        close_data = np.array(new_close_data)
        #print(close_data[:2])

        label_data_tmp= self.create_label(tmp_np, start_idx)
        #print(label_data_tmp[:2])
        #print(type(label_data_tmp))
        new_label_data = label_data_tmp.tolist()
        label_data = np.array(new_label_data)
        #print(label_data[:2])

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
        if in_num == 3:
            # high,lowも入力データとする場合
            high_data = self.create_high(tmp_np, start_idx, self.maxlen)
            new_high_data = high_data.tolist()
            high_data = np.array(new_high_data)
            retX[:, :, 1] = high_data[:]

            low_data = self.create_low(tmp_np, start_idx, self.maxlen)
            new_low_data = low_data.tolist()
            low_data = np.array(new_low_data)
            retX[:, :, 2] = low_data[:]

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