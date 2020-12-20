import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import huber
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow.keras.layers import BatchNormalization

import time
import os

from logging import getLogger
from datetime import datetime
from datetime import timedelta
import time
from decimal import Decimal
from DataSequence2 import DataSequence2

from readConf2 import *
from lstm_generator2 import create_model

"""
チェックポイントで保存した重みからモデルを改めて保存する
"""


if __name__ == '__main__':

    suffix = "90*14"

    SAVE_DIR = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix

    if os.path.isdir(SAVE_DIR):
        #既にモデル保存ディレクトリがある場合はLEARNING_NUMが間違っているのでエラー
        print("ERROR!! SAVE_DIR Already Exists ")
        exit(1)


    model = None

    model = create_model()
    model.load_weights(LOAD_CHK_PATH)

    # SavedModel形式で保存
    model.save(SAVE_DIR)

    print("END!!")