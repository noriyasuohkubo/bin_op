import time


import datetime
import pytz
import talib
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb
from lgbm_make_data import LgbmMakeData
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

model_dir_lstm = "/app/model/bin_op/"

names = [
    "MN771-33",
    "MN773-9",
    'MN774-32',
    'USDJPY_LT1_M7_LSTM1_B2_BS2_T40_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_234-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub2_MN744-9',
    "MN798-15"
]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

for name in names:
    model_tmp = load_model(model_dir_lstm + name,
                           custom_objects={"root_mean_squared_error": root_mean_squared_error, })

    savename = "/app/model/bin_op/" + name +".h5"
    model_tmp.save(savename,save_format='h5')