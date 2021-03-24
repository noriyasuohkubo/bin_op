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
        ["0001", "90*1"],
        ["0002", "90*2"],
        ["0003", "90*3"],
        ["0004", "90*4"],
        ["0005", "90*5"],
        ["0006", "90*6"],
        ["0007", "90*7"],
        ["0008", "90*8"],
        ["0009", "90*9"],
        ["0010", "90*10"],
        ["0011", "90*11"],
        ["0012", "90*12"],
        ["0013", "90*13"],
        ["0014", "90*14"],
        ["0015", "90*15"],
        ["0016", "90*16"],
        ["0017", "90*17"],
        ["0018", "90*18"],
        ["0019", "90*19"],
        ["0020", "90*20"],
        ["0021", "90*21"],
        ["0022", "90*22"],
        ["0023", "90*23"],
        ["0024", "90*24"],
        ["0025", "90*25"],
        ["0026", "90*26"],
        ["0027", "90*27"],
        ["0028", "90*28"],
        ["0029", "90*29"],
        ["0030", "90*30"],
        ["0031", "90*31"],
        ["0032", "90*32"],
        ["0033", "90*33"],
        ["0034", "90*34"],
        ["0035", "90*35"],
        ["0036", "90*36"],
        ["0037", "90*37"],
        ["0038", "90*38"],
        ["0039", "90*39"],

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

        if METHOD == "LSTM":
            model = create_model_lstm()
        elif METHOD == "NORMAL":
            model = create_model_normal()
        elif METHOD == "BY":
            model = create_model_by()
        model.load_weights(chk_path)

        # SavedModel形式で保存
        model.save(save_path)

    print("END!!")
