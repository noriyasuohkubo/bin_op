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
import tensorflow as tf

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

class DataSequence_lgbm:

    def __init__(self, batch_len):
        #コンストラクタ
        self.xs = []
        self.ys = []
        self.cnt = 0
        self.batch_len = batch_len

        for i in range(101):
            tmp_list = (i,i+1)
            self.xs.append(tmp_list)
            self.ys.append(i)
        #print(self.xs)

        if len(self.xs) % self.batch_len == 0:
            self.steps_per_epoch = int(len(self.xs) / self.batch_len)
        else:
            self.steps_per_epoch = int(len(self.xs) / self.batch_len) + 1

    def getx(self, start, end):
        print("startx", start)
        for item1 in self.xs[start : end]:
            print(item1)

        return self.xs[start : end]

    def gety(self, start, end):
        print("starty", start)
        for item1 in self.ys[start : end]:
            print(item1)

        return self.ys[start : end]

    @tf.function
    def __getitem__(self):
        start = self.cnt * self.batch_len
        print("start", start)
        end = self.cnt + self.batch_len

        batch_len =  self.batch_len

        if end > len(self.xs):
            end = len(self.xs)
            batch_len = len(self.xs) - start

        print("end", end)

        datasetX = tf.data.Dataset.from_tensor_slices(self.getx(start,end))
        datasetY = tf.data.Dataset.from_tensor_slices(self.gety(start,end))

        datasetX = datasetX.batch(batch_len)
        datasetY = datasetY.batch(batch_len)
        #print(dataset)
        #dataset = dataset.repeat(1)
        #iterator = dataset.make_one_shot_iterator()
        #features_tensors, labels = iterator.get_next()
        #iterator = iter(dataset)
        #print(iterator.get_next())
        #print(dataset)
        #features_tensors, labels = iterator.get_next()
        #features_tensors, labels = dataset
        #print(features_tensors)
        features = {'x': datasetX}

        self.cnt = self.cnt + 1

        return features, datasetY

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch

    def on_epoch_end(self):
        # epoch終了時の処理 リストをランダムに並べ替える
        if self.test_flg == False:
            random.shuffle(self.train_list)

    def get_data_length(self):
        return self.data_length
