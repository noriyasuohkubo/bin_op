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
from lstm_generator2 import create_model_lstm, create_model_normal, create_model_by

"""
チェックポイントで保存した重みからモデルを改めて保存する
"""


if __name__ == '__main__':

    suffixs = [
        ["0001", "1"],
        ["0002", "2"],
        ["0003", "3"],
        ["0004", "4"],
        ["0005", "5"],
        ["0006", "6"],
        ["0007", "7"],
        ["0008", "8"],
        ["0009", "9"],
        ["0010", "10"],
        ["0011", "11"],
        ["0012", "12"],
        ["0013", "13"],
        ["0014", "14"],
        ["0015", "15"],
        ["0016", "16"],
        ["0017", "17"],
        ["0018", "18"],
        ["0019", "19"],
        ["0020", "20"],
        ["0021", "21"],
        ["0022", "22"],
        ["0023", "23"],
        ["0024", "24"],
        ["0025", "25"],
        ["0026", "26"],
        ["0027", "27"],
        ["0028", "28"],
        ["0029", "29"],
        ["0030", "30"],
        ["0031", "31"],
        ["0032", "32"],
        ["0033", "33"],
        ["0034", "34"],
        ["0035", "35"],
        ["0036", "36"],
        ["0037", "37"],
        ["0038", "38"],
        ["0039", "39"],

    ]

    #suffixs = [     ["0009", "130*64"],]

    for suffix in suffixs:
        chk_path = os.path.join(CHK_DIR_LOAD, suffix[0])
        save_path = "/app/model/bin_op/" + FILE_PREFIX + "-" + suffix[1]

        print(chk_path)

        if os.path.isdir(save_path):
            print("ERROR!! SAVE_DIR Already Exists ", save_path)
            exit(1)

        model = None

        if METHOD == "LSTM" or METHOD == "LSTM2":
            model = create_model_lstm()
        elif METHOD == "NORMAL":
            model = create_model_normal()
        elif METHOD == "BY":
            model = create_model_by()
        model.load_weights(chk_path)

        # SavedModel形式で保存
        model.save(save_path)

    print("END!!")
