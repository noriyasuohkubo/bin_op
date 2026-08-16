import copy
import csv

import numpy as np
import redis
import json
from matplotlib import pyplot as plt
from datetime import datetime
import time
import conf_class
from CustomCSVLogger import CustomCallback
from DataSequence2_from_pickle import DataSequence2_from_pickle
from DataSequence2_from_pickle_on_memory import DataSequence2_from_pickle_on_memory
from util import *
import send_mail as mail
import tensorflow_probability as tfp
from silence_tensorflow import silence_tensorflow
from tcn import TCN  # keras-tcn
from DataSequence2_copy import DataSequence2_copy
import psutil
import warnings
warnings.simplefilter('ignore')


silence_tensorflow()  # ログ抑制 import tensorflowの前におく

import tensorflow as tf
import socket
from DataSequence2 import DataSequence2
from tensorflow.keras import backend as K
from important_index import *
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow_addons.optimizers import AdamW,RectifiedAdam,LazyAdam
from DataSequence2_from_pickle_test import DataSequence2_from_pickle_test
from DataSequence2_from_pickle_test_raw import DataSequence2_from_pickle_test_raw
from DataSequence2_from_pickle_test_on_memory import DataSequence2_from_pickle_test_on_memory

tf.keras.optimizers.AdamW = AdamW
tf.keras.optimizers.RectifiedAdam = RectifiedAdam
tf.keras.optimizers.LazyAdam = LazyAdam

from tensorflow.keras.callbacks import CSVLogger

host = socket.gethostname()

"""
DBに保存した利益(answer)を参照してテストする

LEARNING_TYPE == 
"CATEGORY_BIN_UP_IFDSF","CATEGORY_BIN_DW_IFDSF"
"CATEGORY_BIN_UP_TP","CATEGORY_BIN_DW_TP"
の場合専用
"""
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

c = None

png_dir = "/app/fx/png/"


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def fx_mean_squared_error(y_true, y_pred):
    # 予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(error ** 2 + conf_class.FX_LOSS_PNALTY * error ** 2 * not_trend_match)

    return loss


def fx_mean_squared_error2(y_true, y_pred):
    # 予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(
        error ** 2 - conf_class.FX_LOSS_PNALTY * error ** 2 + conf_class.FX_LOSS_PNALTY * 2 * error ** 2 * not_trend_match)
    return loss

def fx_mean_squared_error3(y_true, y_pred):
    # 予想値のトレンドが異なる場合は罰則を強化する
    # 正解が0.0の場合は罰則なし

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    not_trend_match = not_trend_match * tf.cast(tf.math.sign(y_true) != 0.0, tf.float32)# 正解が0.0の場合は罰則なし
    loss = tf.math.reduce_mean(error ** 2 + ((conf_class.FX_LOSS_PNALTY * error) ** 2) * not_trend_match)

    return loss

def mean_squared_error_custome(y_true, y_pred):
    # 誤差の３乗を罰則とする
    error = abs(y_true - y_pred)
    loss = tf.math.reduce_mean(error ** conf_class.MSE_PENALTY)

    return loss


def fx_insensitive_error(y_true, y_pred):
    # ε-感度損失:細かい誤差は気にしない
    # 閾値以上の誤差がある場合だけ罰則
    error = abs(y_true - y_pred)
    not_trend_match = tf.cast(error >= conf_class.INSENSITIVE_BORDER, tf.float32)

    loss = tf.math.reduce_mean(error ** 2 * not_trend_match)

    return loss


def negative_log_likelihood(y_true, y_pred):
    return -1 * y_pred.log_prob(y_true)

def categorical_focal_loss(alpha, gamma=2.):
    """
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def do_eval(conf, file, model_suffix):

    print("eval start", datetime.now())

    global c
    c = conf

    if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL) != 0:
        dataSequence2 = DataSequence2_from_pickle(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL, True)
        eval_file = conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL[0]["score"]
        print("DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL:", eval_file)

    elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY) != 0:
        dataSequence2 = DataSequence2_from_pickle_on_memory(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY, True)
        eval_file = conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY[0]["score"]
        print("DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY:", eval_file)


    if "CATEGORY" in conf.LEARNING_TYPE:
        learning_type = "CATEGORY"
    else:
        learning_type = "REGRESSION"


    print("file:", file)
    r = redis.Redis(host="win2", port=6379, db=1)
    stp = file.split("MN")[1]

    makedirs(c.HISTORY_DIR_PARENT + file)

    csv_path = c.HISTORY_DIR_PARENT + file + "/eval_" + str(eval_file) + ".csv"
    print("csv_path:",csv_path)

    result_data = r.zrangebyscore("MODEL_NO", stp, stp, withscores=True)
    for res in result_data:
        body = res[0]
        score = res[1]
        tmps = json.loads(body)
        tmp_name = tmps.get("name")
        print("name:", tmp_name)

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        if learning_type == "CATEGORY":
            writer.writerow(['epoch', 'val_accuracy', 'val_loss'])
        else:
            writer.writerow(['epoch', 'val_loss'])

    for suffix in model_suffix:

        print("")
        print("suffix:", suffix)


        load_dir = "/app/model/bin_op/" + file + "-" + suffix
        if not os.path.isdir(load_dir):
            print("model not exists:" + load_dir)
            exit(1)

        model = tf.keras.models.load_model(load_dir,
                                           custom_objects={"root_mean_squared_error": root_mean_squared_error,
                                                           "fx_mean_squared_error": fx_mean_squared_error,
                                                           "fx_mean_squared_error2": fx_mean_squared_error2,
                                                           "fx_mean_squared_error3": fx_mean_squared_error3,
                                                           "mean_squared_error_custome": mean_squared_error_custome,
                                                           "fx_insensitive_error": fx_insensitive_error,
                                                           "negative_log_likelihood": negative_log_likelihood,
                                                           "categorical_focal_loss_fixed": categorical_focal_loss(alpha=[c.C_ENTROPY_FOCAL_AlPHA], gamma=c.C_ENTROPY_FOCAL_GAMMA),
                                                           })

        loss, accuracy = model.evaluate_generator(dataSequence2,
                                                  steps=None,
                                                  max_queue_size=c.MAX_QUEUE_SIZE * 1,
                                                  use_multiprocessing=False,
                                                  verbose=0,
                                                  callbacks=[CustomCallback(file_path=csv_path,model_number=suffix, learning_type="CATEGORY"),]

                                                  )

        #print(loss,accuracy)



    print("eval end", time.perf_counter() - start_time)


if __name__ == "__main__":

    start_time = time.perf_counter()
    # print("load_dir = ", "/app/model/bin_op/" + FILE_PREFIX)
    # do_predict()
    conf = conf_class.ConfClass()
    if conf.FX == False:
        print("conf.FX == False !!!")
        exit(1)


    file = 'MN1585'
    model_suffix = [str(i + 1) for i in range(200)]

    conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL = [

    ]

    conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY = [
        {
        "score": "453",
        "save_dir_path": "/nvme2/dataSequence2/USDJPY/DS2F453-0",
        }
    ]




    do_eval(conf, file=file, model_suffix=model_suffix )


    print("Processing Time(Sec)", time.perf_counter() - start_time)
    # 終わったらメールで知らせる
    mail.send_message(host, ": testLstmFX2_eval finished!!!")