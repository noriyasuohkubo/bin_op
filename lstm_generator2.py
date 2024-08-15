import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow

silence_tensorflow()  # ログ抑制 import tensorflowの前におく
import tensorflow as tf
from tensorflow.keras.losses import huber, mean_squared_error, log_cosh, mean_absolute_error, hinge, poisson, \
    squared_hinge
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Activation, LayerNormalization
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow.keras import backend as K
from datetime import datetime
import time
from DataSequence2 import DataSequence2
import os
import tensorflow_probability as tfp
import numpy as np
import random as rn
from adabound_tf import AdaBound
import logging.config
from util import *
from tensorflow.keras import initializers
import conf_class
import send_mail as mail
import socket
from qrnn import *
from tcn import TCN  # keras-tcn
import tensorflow_addons as tfa
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import layers
from matplotlib import pyplot as plt

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


# コンピュータ名を取得
host = socket.gethostname()

c = None

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

# set_memory_growthを設定しないと、LSTMだと以下のようにエラーが出てしまう(GRUで代用するしかない)
# よって、設定しておく
# https://github.com/tensorflow/tensorflow/issues/33721
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

current_dir = os.path.dirname(__file__)
logging.config.fileConfig(os.path.join(current_dir, "config", "logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)


# 範囲つき予測(不当分散モデル)のための独自損失関数
# see:
# https://aotamasaki.hatenablog.com/entry/2019/03/01/185430
def loss(y_true, y_pred):
    # flat な1次元にする
    mu = K.reshape(y_pred[:, 0], [-1])
    # 精度パラメーターβを導入
    # β = 1/σ(標準偏差)
    # beta = K.square(K.reshape(y_pred[:,1],[-1]))

    # β = logσ
    beta = K.exp(K.reshape(y_pred[:, 1], [-1]))

    y_true = K.reshape(y_true, [-1])
    dist = tfp.distributions.Normal(loc=mu, scale=beta)
    return K.mean(-1 * dist.log_prob(y_true), axis=-1)

    # return K.mean(beta * K.square(mu - y_true) - K.log(beta), axis=-1)


# モデル作成
def create_model_normal():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))
    if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW":
        activ = 'softmax'
    else:
        activ = None

    K_I = initializers.GlorotUniform(seed=c.SEED)  # RNN系の初期値
    R_I = initializers.Orthogonal(gain=1.0, seed=c.SEED)
    D_I = initializers.GlorotUniform(seed=c.SEED)  # DENSE系の初期値
    O_I = initializers.GlorotUniform(seed=c.SEED)  # OUTPUT系の初期値

    l2_D = None
    if c.L_D_RATE != 0:
        l2_D = tf.keras.regularizers.l2(c.L_D_RATE)

    if len(c.INPUT_LEN) > 1:
        inputs = []
        for i, length in enumerate(c.INPUT_LEN):
            input = keras.Input(shape=(length,))
            inputs.append(input)

        concate = keras.layers.Concatenate()(inputs)

        dense = None
        for i, unit in enumerate(c.DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], activation="relu", kernel_initializer=D_I,
                                           kernel_regularizer=l2_D, )(concate)
                if c.DROP > 0:
                    dense = keras.layers.Dropout(c.DROP)(dense)
            else:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], activation="relu", kernel_initializer=D_I,
                                           kernel_regularizer=l2_D, )(dense)
                if c.DROP > 0:
                    dense = keras.layers.Dropout(c.DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(dense)
        else:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(inputs)

        model = keras.Model(inputs=[inputs], outputs=[output])

    else:
        # inputが1種類の場合

        input = keras.Input(shape=(c.INPUT_LEN[0],))
        dense = None
        for i, unit in enumerate(c.DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], activation="relu", kernel_initializer=D_I,
                                           kernel_regularizer=l2_D, )(input)  # 正則化： L2、
                if c.DROP > 0:
                    dense = keras.layers.Dropout(c.DROP)(dense)

            else:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], activation="relu", kernel_initializer=D_I,
                                           kernel_regularizer=l2_D, )(dense)  # 正則化： L2、
                if c.DROP > 0:
                    dense = keras.layers.Dropout(c.DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(dense)
        else:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(input)

        model = keras.Model(inputs=[input], outputs=[output])

    if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_UP" or c.LEARNING_TYPE == "CATEGORY_BIN_DW":
        if c.LOSS_TYPE == "B-ENTROPY":
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])
        elif c.LOSS_TYPE == "C-ENTROPY":
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])

    elif c.LEARNING_TYPE == "REGRESSION_SIGMA":
        # 範囲つき予測
        model.compile(loss=loss, optimizer=Adam(lr=c.LEARNING_RATE))
    elif c.LEARNING_TYPE in ["REGRESSION", "REGRESSION_UP", "REGRESSION_DW", "REGRESSION_OCOPS"]:
        if c.LOSS_TYPE == "MSE":
            model.compile(loss=mean_squared_error, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "MSEC":
            model.compile(loss=mean_squared_error_custome, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "FXMSE":
            model.compile(loss=fx_mean_squared_error, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "FXMSE2":
            model.compile(loss=fx_mean_squared_error2, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "IE":
            model.compile(loss=fx_insensitive_error, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "RMSE":
            model.compile(loss=root_mean_squared_error, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "HUBER":
            model.compile(loss=huber, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "LOG_COSH":
            model.compile(loss=log_cosh, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "MAE":
            model.compile(loss=mean_absolute_error, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "HINGE":
            model.compile(loss=hinge, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "POISSON":
            model.compile(loss=poisson, optimizer=Adam(lr=c.LEARNING_RATE))
        elif c.LOSS_TYPE == "SQUARED_HINGE":
            model.compile(loss=squared_hinge, optimizer=Adam(lr=c.LEARNING_RATE))
    return model


def get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, rs):
    if lstm_type in ["LSTM" ,"Bi", "KSA-LSTM", "LSTM-KSA", "MHA-LSTM", "LSTM-MHA","LSTM-KSA-CNN","LSTM-MHA-CNN"]:

        return keras.layers.LSTM(lstm_unit,
                                     activation=config.RNN_ACTIVATION,
                                     recurrent_activation=config.RNN_REC_ACTIVATION,
                                     kernel_initializer=k_i,
                                     recurrent_initializer=r_i,
                                     kernel_regularizer=l2_k,
                                     recurrent_regularizer=l2_r,
                                     dropout=config.L_DO,
                                     recurrent_dropout=config.L_RDO,
                                     return_sequences=rs)
    elif lstm_type == "LAYERNORM":
        return keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(lstm_unit,
                                                            kernel_initializer=k_i,
                                                            recurrent_initializer = r_i,
                                                            kernel_regularizer = l2_k,
                                                            recurrent_regularizer = l2_r,
                                                            dropout=config.L_DO,
                                                            recurrent_dropout=config.L_RDO,
                                                          ),
                                     return_sequences=rs)

    elif lstm_type == "PEEPHOLE":
        return keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(lstm_unit,
                                                          kernel_initializer=k_i,
                                                          recurrent_initializer=r_i,
                                                          kernel_regularizer=l2_k,
                                                          recurrent_regularizer=l2_r,
                                                          dropout=config.L_DO,
                                                          recurrent_dropout=config.L_RDO,
                                                          ),
                                     return_sequences=rs)

    elif lstm_type == "GRU":
        return keras.layers.GRU(lstm_unit,
                                activation=config.RNN_ACTIVATION,
                                recurrent_activation=config.RNN_REC_ACTIVATION,
                                kernel_initializer=k_i,
                                recurrent_initializer=r_i,
                                kernel_regularizer=l2_k,
                                recurrent_regularizer=l2_r,
                                dropout=config.L_DO,
                                recurrent_dropout=config.L_RDO,
                                return_sequences=rs)
    elif lstm_type == "CuDNNLSTM":
        return tf.compat.v1.keras.layers.CuDNNLSTM(lstm_unit,
                                                   kernel_initializer=k_i,
                                                   recurrent_initializer=r_i,
                                                   kernel_regularizer=l2_k,
                                                   recurrent_regularizer=l2_r,
                                                   dropout=config.L_DO,
                                                   recurrent_dropout=config.L_RDO,
                                                   return_sequences=rs)

    elif lstm_type == "QRNN":
        return QRNN(lstm_unit, window_size=c.WINDOW_SIZE, return_sequences=rs)

def get_ksa(config, input):
    tmp_layer = SeqSelfAttention(units=c.KSA_UNIT_NUM)(input)
    if config.SELF_AT_NORMAL == "BATCH":
        tmp_layer = BatchNormalization()(tmp_layer)
    elif config.SELF_AT_NORMAL == "LAYER":
        tmp_layer = LayerNormalization()(tmp_layer)

    if config.SELF_AT_INPUT_PLUS:
        tmp_layer = tmp_layer + input

    return tmp_layer

def get_mha(config, input):
    tmp_layer = layers.MultiHeadAttention(
        key_dim=config.MHA_UNIT_NUM, num_heads=config.MHA_HEAD_NUM, )(input, input)

    if config.SELF_AT_NORMAL == "BATCH":
        tmp_layer = BatchNormalization()(tmp_layer)
    elif config.SELF_AT_NORMAL == "LAYER":
        tmp_layer = LayerNormalization()(tmp_layer)

    if config.SELF_AT_INPUT_PLUS:
        tmp_layer = tmp_layer + input

    return tmp_layer

def get_cnn(config, input,lstm_unit):
    tmp_layer = layers.Conv1D(filters=lstm_unit, kernel_size=1, activation="relu")(input)
    tmp_layer = layers.Conv1D(filters=input.shape[-1], kernel_size=1)(tmp_layer)
    if config.CNN_NORMAL == "BATCH":
        tmp_layer = BatchNormalization()(tmp_layer)
    elif config.CNN_NORMAL == "LAYER":
        tmp_layer = LayerNormalization()(tmp_layer)

    if config.CNN_INPUT_PLUS:
        tmp_layer = tmp_layer + input

    return tmp_layer

def get_tcn(lstm_unit, config, rs):
    return TCN(nb_filters=lstm_unit,
               kernel_size=config.TCN_KERNEL_SIZE,
               nb_stacks=config.TCN_NB_STACKS,
               return_sequences=rs)


def make_layer(lstms, input, lstm_layer_num, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, config):
    if lstm_type in ["LSTM", "GRU", "QRNN", "CuDNNLSTM", "LAYERNORM", "PEEPHOLE"]:
        if lstm_layer_num == 1:
            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False)(input)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)

            if c.LSTM_DO > 0:
                tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(input)
                elif j == lstm_layer_num -1:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False)(tmp_layer)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

                if c.LSTM_DO > 0:
                    tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

        lstms.append(tmp_layer)

    elif lstm_type in ["LSTM-KSA", "LSTM-KSA-CNN","LSTM-MHA", "LSTM-MHA-CNN"]:
        if lstm_layer_num == 1:
            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(input)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(input)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

        for l in range(config.SELF_AT_LAYER_NUM):
            if lstm_type in ["LSTM-KSA", "LSTM-KSA-CNN",]:
                tmp_layer = get_ksa(config, tmp_layer)
            elif lstm_type in ["LSTM-MHA", "LSTM-MHA-CNN", ]:
                tmp_layer = get_mha(config, tmp_layer)

            if lstm_type in["LSTM-KSA-CNN", "LSTM-MHA-CNN"]:
                tmp_layer = get_cnn(config, tmp_layer, config.CNN_UNIT_NUM)

        tmp_layer = layers.GlobalAveragePooling1D(data_format="channels_first")(tmp_layer)
        lstms.append(tmp_layer)

    elif lstm_type in ["KSA-LSTM","MHA-LSTM",]:
        for l in range(config.SELF_AT_LAYER_NUM):
            if l == 0:
                if lstm_type in ["KSA-LSTM"]:
                    tmp_layer = get_ksa(config, input)
                elif lstm_type in ["MHA-LSTM"]:
                    tmp_layer = get_mha(config, input)
            else:
                if lstm_type in ["KSA-LSTM"]:
                    tmp_layer = get_ksa(config, tmp_layer)
                elif lstm_type in ["MHA-LSTM"]:
                    tmp_layer = get_mha(config, tmp_layer)
        if lstm_layer_num == 1:

            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False)(tmp_layer)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == lstm_layer_num -1:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False)(tmp_layer)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

        lstms.append(tmp_layer)

    elif lstm_type == "Bi":
        if lstm_layer_num == 1:
            tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False))(
                    input)

            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)

            if c.LSTM_DO > 0:
                tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

            lstms.append(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True))(
                        input)
                elif j != 0 and j != (lstm_layer_num - 1):
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, True))(
                        tmp_layer)
                elif j == (lstm_layer_num - 1):
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, False))(
                            tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

                if c.LSTM_DO > 0:
                    tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

            lstms.append(tmp_layer)

    elif lstm_type == "TCN":
        if lstm_layer_num == 1:
            lstms.append(get_tcn(lstm_unit, config, config.RETURN_SEQ)(input))
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_tcn(lstm_unit, config, True)(input)
                elif j != 0 and j != (lstm_layer_num - 1):
                    tmp_layer = get_tcn(lstm_unit, config, True)(tmp_layer)
                elif j == (lstm_layer_num - 1):
                    lstms.append(get_tcn(lstm_unit, config, config.RETURN_SEQ)(tmp_layer))

    elif lstm_type in["KSA-TCN", "MHA-TCN"]:
        for l in range(config.SELF_AT_LAYER_NUM):
            if l == 0:
                if lstm_type in ["KSA-TCN"]:
                    tmp_layer = get_ksa(config, input)
                elif lstm_type in ["MHA-TCN"]:
                    tmp_layer = get_mha(config, input)
            else:
                if lstm_type in ["KSA-TCN"]:
                    tmp_layer = get_ksa(config, tmp_layer)
                elif lstm_type in ["MHA-TCN"]:
                    tmp_layer = get_mha(config, tmp_layer)

        if lstm_layer_num == 1:
            tmp_layer = get_tcn(lstm_unit, config, config.RETURN_SEQ)(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_tcn(lstm_unit, config, True)(tmp_layer)
                elif j != 0 and j != (lstm_layer_num - 1):
                    tmp_layer = get_tcn(lstm_unit, config, True)(tmp_layer)
                elif j == (lstm_layer_num - 1):
                    tmp_layer = get_tcn(lstm_unit, config, config.RETURN_SEQ)(tmp_layer)

        lstms.append(tmp_layer)

# モデル作成
def create_model_lstm(conf=None):
    global c
    if conf != None:
        c = conf
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))
    if c.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN_UP_IFD",
                           "CATEGORY_BIN_DW_IFD",
                           "CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO", "CATEGORY_BIN_UP_IFDSF",
                           "CATEGORY_BIN_DW_IFDSF",
                           "CATEGORY_BIN_UP_TP", "CATEGORY_BIN_DW_TP",
                           "CATEGORY_BIN_UP_OCO", "CATEGORY_BIN_DW_OCO", "CATEGORY_BIN_UP_OCOA", "CATEGORY_BIN_DW_OCOA",
                           "CATEGORY_OCOPS"]:
        activ = 'softmax'
    else:
        activ = None

    K_I = initializers.GlorotUniform(seed=c.SEED)  # RNN系の初期値
    R_I = initializers.Orthogonal(gain=1.0, seed=c.SEED)
    D_I = initializers.GlorotUniform(seed=c.SEED)  # DENSE系の初期値
    O_I = initializers.GlorotUniform(seed=c.SEED)  # OUTPUT系の初期値
    l2_K = None
    l2_R = None
    if c.L_K_RATE != "":
        type, val = c.L_K_RATE.split("-")
        val = float(val)
        if type == "1":
            l2_K = tf.keras.regularizers.l1(val)
        elif type == "2":
            l2_K = tf.keras.regularizers.l2(val)
        elif type == "12":
            l2_K = tf.keras.regularizers.L1L2(val, val)
        else:
            print("invalid type")
            exit(1)

    if c.L_R_RATE != "":
        type, val = c.L_R_RATE.split("-")
        val = float(val)
        if type == "1":
            l2_R = tf.keras.regularizers.l1(val)
        elif type == "2":
            l2_R = tf.keras.regularizers.l2(val)
        elif type == "12":
            l2_R = tf.keras.regularizers.L1L2(val, val)
        else:
            print("invalid type")
            exit(1)

    l2_D = None
    if c.L_D_RATE != "":
        type, val = c.L_D_RATE.split("-")
        val = float(val)
        if type == "1":
            l2_D = tf.keras.regularizers.l1(val)
        elif type == "2":
            l2_D = tf.keras.regularizers.l2(val)
        elif type == "12":
            l2_D = tf.keras.regularizers.L1L2(val, val)
        else:
            print("invalid type")
            exit(1)

    #if len(c.LSTM_UNIT) > 1:
    lstms = []
    inputs = []

    for i, unit in enumerate(c.LSTM_UNIT):
        if i == 0 and c.DB1_NOT_LEARN:
            #DB1が学習対象でないならスキップ
            continue

        ipt_data = c.INPUT_DATAS[i]
        ipt_lists = ipt_data.split("_")
        if c.INPUT_SEPARATE_FLG == False or ipt_data == "" or len(ipt_lists) == 1:
            if ipt_data == "":
                input = keras.Input(shape=(c.INPUT_LEN[i], 1))
            else:
                input = keras.Input(shape=(c.INPUT_LEN[i], len(ipt_lists)))

            if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or \
                    c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                    c.METHOD == "TCN" or c.METHOD == "TCN7":
                make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[i], K_I, R_I, l2_K, l2_R, c)

            inputs.append(input)

        else:
            for j in range(len(ipt_lists)):
                input = keras.Input(shape=(c.INPUT_LEN[i], 1))

                if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                        c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                        c.METHOD == "TCN" or c.METHOD == "TCN7":
                    make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[i], K_I, R_I, l2_K, l2_R, c)

                inputs.append(input)

    if len(c.FOOT_DBS) != 0:
        for db_tmp in c.FOOT_DBS:
            d_term,d_len,d_unit,d_x,db_name,separate_flg = db_tmp
            ipt_lists_foot = d_x.split("_")
            if separate_flg:
                for j in range(len(ipt_lists_foot)):
                    tmp_input = keras.Input(shape=(d_len, 1))
                    inputs.append(tmp_input)
                    make_layer(lstms, tmp_input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, d_unit, K_I, R_I, l2_K, l2_R, c)
            else:
                tmp_input = keras.Input(shape=(d_len, len(ipt_lists_foot)))
                inputs.append(tmp_input)
                make_layer(lstms, tmp_input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, d_unit, K_I, R_I, l2_K, l2_R, c)

    if c.METHOD == "LSTM2":
        # LSTMの予想値を入力
        predict_input = keras.Input(shape=(1,))
        inputs.append(predict_input)
        lstms.append(predict_input)

    if c.METHOD == "LSTM3" or c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
            c.METHOD == "TCN7" :
        # 秒データをone-hotで入力
        sec_input = keras.Input(shape=(c.SEC_OH_LEN,))
        inputs.append(sec_input)
        lstms.append(sec_input)

    if c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
            c.METHOD == "TCN7":
        # 分データをone-hotで入力
        min_input = keras.Input(shape=(c.MIN_OH_LEN,))
        inputs.append(min_input)
        lstms.append(min_input)

    if c.METHOD == "LSTM5" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
            c.METHOD == "TCN7":
        # 時間データをone-hotで入力
        hour_input = keras.Input(shape=(c.HOUR_OH_LEN,))
        inputs.append(hour_input)
        lstms.append(hour_input)

    if c.METHOD == "LSTM10":
        # 曜日データをone-hotで入力
        week_input = keras.Input(shape=(c.WEEK_OH_LEN,))
        inputs.append(week_input)
        lstms.append(week_input)

    if c.HOR_LEARN_ON:
        tmp_data_num = c.HOR_DATA_NUM * 2 + 1
        for n in range(tmp_data_num):
            mm_input = keras.Input(shape=(1,))
            inputs.append(mm_input)
            lstms.append(mm_input)

    if c.HIGHLOW_DB_CORE != "":
        for n in range(c.HIGHLOW_DATA_NUM):
            mm_input = keras.Input(shape=(1,))
            inputs.append(mm_input)
            lstms.append(mm_input)

    if len(c.NON_LSTM_LIST) != 0:
        for l in c.NON_LSTM_LIST:
            for m in l["inputs"]:
                mm_input = keras.Input(shape=(l["length"],))
                inputs.append(mm_input)
                lstms.append(mm_input)

    if c.OANDA_ORD_DB != "":
        i_num = int(c.OANDA_ORD_NUM * 2 + 1)
        for i in range(i_num):
            mm_input = keras.Input(shape=(1,))
            inputs.append(mm_input)
            lstms.append(mm_input)

    if c.OANDA_POS_DB != "":
        i_num = int(c.OANDA_POS_NUM * 2 + 1)
        for i in range(i_num):
            mm_input = keras.Input(shape=(1,))
            inputs.append(mm_input)
            lstms.append(mm_input)

    for mm in c.IND_FOOT_COL:
        mm_input = keras.Input(shape=(1,))
        inputs.append(mm_input)
        lstms.append(mm_input)

    if c.METHOD == "LSTM8":
        # tick数を入力
        volume_input = keras.Input(shape=(1,))
        inputs.append(volume_input)
        lstms.append(volume_input)

    if c.METHOD == "LSTM9":
        # 予想を入力
        for ipt in c.LSTM9_INPUTS:
            pred_input = keras.Input(shape=(1,))
            inputs.append(pred_input)
            lstms.append(pred_input)

            if c.LSTM9_USE_CLOSE:
                pred_close_input = keras.Input(shape=(1,))
                inputs.append(pred_close_input)
                lstms.append(pred_close_input)

    if c.DB_EXTRA_1 != "":
        tmp_input = keras.Input(shape=(c.DB_EXTRA_1_LEN, 1))
        inputs.append(tmp_input)
        lstms.append(keras.layers.LSTM(c.DB_EXTRA_1_UNIT,
                                       kernel_initializer=K_I,
                                       recurrent_initializer=R_I,
                                       kernel_regularizer=l2_K,
                                       recurrent_regularizer=l2_R,
                                       # dropout=0.,
                                       # recurrent_dropout=0.,
                                       return_sequences=False)(tmp_input))

    if c.NOW_RATE_FLG == True:
        now_rate_input = keras.Input(shape=(1,))
        inputs.append(now_rate_input)
        lstms.append(now_rate_input)

    for i in range(len(c.OPTIONS)):
        option_input = keras.Input(shape=(1,))
        inputs.append(option_input)
        lstms.append(option_input)

    if len(lstms) > 1:
        concate = keras.layers.Concatenate()(lstms)
    else:
        concate = lstms[0]

    if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM":
        concate = BatchNormalization()(concate)
    elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM":
        concate = LayerNormalization()(concate)

    dense = None
    for i, unit in enumerate(c.DENSE_UNIT):
        if i == 0:
            dense = keras.layers.Dense(c.DENSE_UNIT[i], kernel_initializer=D_I,
                                       kernel_regularizer=l2_D, )(concate)  # 正則化： L2、
        else:
            dense = keras.layers.Dense(c.DENSE_UNIT[i], kernel_initializer=D_I,
                                       kernel_regularizer=l2_D, )(dense)  # 正則化： L2、

        if c.NORMAL_TYPE == "BATCH_NORMAL" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
            dense = BatchNormalization()(dense)
        elif c.NORMAL_TYPE == "LAYER_NORMAL" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
            dense = LayerNormalization()(dense)

        if c.DENSE_ACTIVATION == 'leaky_relu':
            dense = Activation(tf.nn.leaky_relu)(dense)
        elif c.DENSE_ACTIVATION == 'relu6':
            dense = Activation(tf.nn.relu6)(dense)
        elif c.DENSE_ACTIVATION == 'crelu':
            dense = Activation(tf.nn.crelu)(dense)
        else:
            dense = Activation(c.DENSE_ACTIVATION)(dense)
        if c.DROP > 0:
            dense = keras.layers.Dropout(c.DROP)(dense)

    if dense != None:
        if c.MIXTURE_NORMAL:
            # denseのユニット数を計算
            # see https://qiita.com/pocokhc/items/be178d1d7deeeafac8c0

            params_size = tfp.layers.MixtureNormal.params_size(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))
            mn_output = keras.layers.Dense(params_size, activation=None)(dense)  # 指定のDense層を追加(activationはNone)
            output = tfp.layers.MixtureNormal(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))(mn_output)  # MixtureNormal層を最後に追加
        else:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(dense)
    else:
        if c.MIXTURE_NORMAL:
            # denseのユニット数を計算
            # see https://qiita.com/pocokhc/items/be178d1d7deeeafac8c0
            params_size = tfp.layers.MixtureNormal.params_size(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))
            mn_output = keras.layers.Dense(params_size, activation=None)(concate)  # 指定のDense層を追加(activationはNone)
            output = tfp.layers.MixtureNormal(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))(mn_output)  # MixtureNormal層を最後に追加
        else:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(concate)


    if c.MIXTURE_NORMAL:
        model = keras.Model(inputs=inputs, outputs=output)
    else:
        model = keras.Model(inputs=inputs, outputs=[output])

    """
    else:
        inputs = []
        lstms = []

        # inputが1種類の場合
        ipt_data = c.INPUT_DATAS[0]
        ipt_lists = ipt_data.split("_")
        if c.INPUT_SEPARATE_FLG == False or ipt_data == "" or len(ipt_lists) == 1:

            if ipt_data == "":
                input = keras.Input(shape=(c.INPUT_LEN[0], 1))
            else:
                input = keras.Input(shape=(c.INPUT_LEN[0], len(ipt_lists)))

            inputs.append(input)
            if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or \
                    c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                    c.METHOD == "TCN" or c.METHOD == "TCN7":
                make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[0], K_I, R_I, l2_K, l2_R, c)
        else:
            for j in range(len(ipt_lists)):
                input = keras.Input(shape=(c.INPUT_LEN[0], 1))
                inputs.append(input)
                if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or \
                        c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM10" or \
                        c.METHOD == "TCN" or c.METHOD == "TCN7":
                    make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[0], K_I, R_I, l2_K, l2_R, c)

        if len(c.FOOT_DBS) != 0:
            for db_tmp in c.FOOT_DBS:
                d_term,d_len,d_unit,d_x,db_name,separate_flg = db_tmp
                ipt_lists_foot = d_x.split("_")
                if separate_flg:
                    for j in range(len(ipt_lists_foot)):
                        tmp_input = keras.Input(shape=(d_len, 1))
                        inputs.append(tmp_input)
                        make_layer(lstms, tmp_input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, d_unit, K_I, R_I, l2_K, l2_R, c)
                else:
                    tmp_input = keras.Input(shape=(d_len, len(ipt_lists_foot)))
                    inputs.append(tmp_input)
                    make_layer(lstms, tmp_input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, d_unit, K_I, R_I, l2_K, l2_R, c)

        if c.METHOD == "LSTM2":
            # LSTMの予想値を入力
            predict_input = keras.Input(shape=(1,))
            inputs.append(predict_input)
            lstms.append(predict_input)

        if c.METHOD == "LSTM3" or c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM10" or \
                c.METHOD == "TCN7":
            # 秒データをone-hotで入力
            sec_input = keras.Input(shape=(c.SEC_OH_LEN,))
            inputs.append(sec_input)
            lstms.append(sec_input)

        if c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM10" or \
                c.METHOD == "TCN7":
            # 分データをone-hotで入力
            min_input = keras.Input(shape=(c.MIN_OH_LEN,))
            inputs.append(min_input)
            lstms.append(min_input)

        if c.METHOD == "LSTM5" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM10" or c.METHOD == "TCN7":
            # 時間データをone-hotで入力
            hour_input = keras.Input(shape=(c.HOUR_OH_LEN,))
            inputs.append(hour_input)
            lstms.append(hour_input)

        if c.METHOD == "LSTM10":
            # 曜日データをone-hotで入力
            week_input = keras.Input(shape=(c.WEEK_OH_LEN,))
            inputs.append(week_input)
            lstms.append(week_input)

        if c.HOR_LEARN_ON:
            tmp_data_num = c.HOR_DATA_NUM * 2 + 1
            for n in range(tmp_data_num):
                mm_input = keras.Input(shape=(1,))
                inputs.append(mm_input)
                lstms.append(mm_input)

        if c.HIGHLOW_DB_CORE != "":
            for n in range(c.HIGHLOW_DATA_NUM):
                mm_input = keras.Input(shape=(1,))
                inputs.append(mm_input)
                lstms.append(mm_input)

        if len(c.NON_LSTM_LIST) != 0:
            for l in c.NON_LSTM_LIST:
                for m in l["inputs"]:
                    mm_input = keras.Input(shape=(l["length"],))
                    inputs.append(mm_input)
                    lstms.append(mm_input)

        if c.OANDA_ORD_DB != "":
            i_num = int(c.OANDA_ORD_NUM * 2 + 1)
            for i in range(i_num):
                mm_input = keras.Input(shape=(1,))
                inputs.append(mm_input)
                lstms.append(mm_input)

        if c.OANDA_POS_DB != "":
            i_num = int(c.OANDA_POS_NUM * 2 + 1)
            for i in range(i_num):
                mm_input = keras.Input(shape=(1,))
                inputs.append(mm_input)
                lstms.append(mm_input)

        for mm in c.IND_FOOT_COL:
            mm_input = keras.Input(shape=(1,))
            inputs.append(mm_input)
            lstms.append(mm_input)

        if c.METHOD == "LSTM8":
            # tick数を入力
            volume_input = keras.Input(shape=(1,))
            inputs.append(volume_input)
            lstms.append(volume_input)

        if c.NOW_RATE_FLG == True:
            now_rate_input = keras.Input(shape=(1,))
            inputs.append(now_rate_input)
            lstms.append(now_rate_input)

        for i in range(len(c.OPTIONS)):
            option_input = keras.Input(shape=(1,))
            inputs.append(option_input)
            lstms.append(option_input)

        concate = lstms[0]

        if len(lstms) > 1:
            concate = keras.layers.Concatenate()(lstms)

        #concate = keras.layers.Concatenate()(lstms)
        if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM":
            concate = BatchNormalization()(concate)
        elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM":
            concate = LayerNormalization()(concate)

        dense = None
        for i, unit in enumerate(c.DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], kernel_initializer=D_I,
                                               kernel_regularizer=l2_D, )(concate)  # 正則化： L2、
            else:
                dense = keras.layers.Dense(c.DENSE_UNIT[i], kernel_initializer=D_I,
                                               kernel_regularizer=l2_D, )(dense)  # 正則化： L2、

            if c.NORMAL_TYPE == "BATCH_NORMAL" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                dense = BatchNormalization()(dense)
            elif c.NORMAL_TYPE == "LAYER_NORMAL" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                dense = LayerNormalization()(dense)

            if c.DENSE_ACTIVATION == 'leaky_relu':
                dense = Activation(tf.nn.leaky_relu)(dense)
            elif c.DENSE_ACTIVATION == 'relu6':
                dense = Activation(tf.nn.relu6)(dense)
            elif c.DENSE_ACTIVATION == 'crelu':
                dense = Activation(tf.nn.crelu)(dense)
            else:
                dense = Activation(c.DENSE_ACTIVATION)(dense)

            if c.DROP > 0:
                dense = keras.layers.Dropout(c.DROP)(dense)

        if dense != None:
            if c.MIXTURE_NORMAL:
                # denseのユニット数を計算
                # see https://qiita.com/pocokhc/items/be178d1d7deeeafac8c0
                params_size = tfp.layers.MixtureNormal.params_size(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))
                mn_output = keras.layers.Dense(params_size, activation=None)(dense)  # 指定のDense層を追加(activationはNone)
                output = tfp.layers.MixtureNormal(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))(mn_output)  # MixtureNormal層を最後に追加
            else:
                output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(dense)
        else:
            if c.MIXTURE_NORMAL:
                # denseのユニット数を計算
                # see https://qiita.com/pocokhc/items/be178d1d7deeeafac8c0
                params_size = tfp.layers.MixtureNormal.params_size(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))
                mn_output = keras.layers.Dense(params_size, activation=None)(concate)  # 指定のDense層を追加(activationはNone)
                output = tfp.layers.MixtureNormal(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))(mn_output)  # MixtureNormal層を最後に追加
            else:
                output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I)(concate)


        if c.MIXTURE_NORMAL:
            model = keras.Model(inputs=inputs, outputs=output)
        else:
            model = keras.Model(inputs=inputs, outputs=[output])
    """
    opt = Adam(lr=c.LEARNING_RATE)
    if c.OPT == "ADABOUND":
        opt = AdaBound(lr=c.LEARNING_RATE, amsbound=False, )
    elif c.OPT == "AMSBOUND":
        opt = AdaBound(lr=c.LEARNING_RATE, amsbound=True, )

    if c.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN_UP_IFD",
                           "CATEGORY_BIN_DW_IFD",
                           "CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO", "CATEGORY_BIN_UP_IFDSF",
                           "CATEGORY_BIN_DW_IFDSF", "CATEGORY_BIN_UP_TP", "CATEGORY_BIN_DW_TP",
                           "CATEGORY_BIN_UP_OCO", "CATEGORY_BIN_DW_OCO", "CATEGORY_BIN_UP_OCOA", "CATEGORY_BIN_DW_OCOA",
                           "CATEGORY_OCOPS"]:
        if c.LOSS_TYPE == "B-ENTROPY":
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        elif c.LOSS_TYPE == "C-ENTROPY":
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    elif c.LEARNING_TYPE == "REGRESSION_SIGMA":
        # 範囲つき予測
        model.compile(loss=loss, optimizer=opt)
    elif c.LEARNING_TYPE in ["REGRESSION", "REGRESSION_HIGH_LOW_DIV", "REGRESSION_HIGH_LOW", "REGRESSION_UP",
                             "REGRESSION_DW", "REGRESSION_OCOPS"]:

        if c.LOSS_TYPE == "MSE":
            model.compile(loss=mean_squared_error, optimizer=opt)
        elif c.LOSS_TYPE == "MSEC":
            model.compile(loss=mean_squared_error_custome, optimizer=opt)
        elif c.LOSS_TYPE == "FXMSE":
            model.compile(loss=fx_mean_squared_error, optimizer=opt)
        elif c.LOSS_TYPE == "FXMSE2":
            model.compile(loss=fx_mean_squared_error2, optimizer=opt)
        elif c.LOSS_TYPE == "IE":
            model.compile(loss=fx_insensitive_error, optimizer=opt)
        elif c.LOSS_TYPE == "RMSE":
            model.compile(loss=root_mean_squared_error, optimizer=opt)
        elif c.LOSS_TYPE == "HUBER":
            model.compile(loss=huber, optimizer=opt)
        elif c.LOSS_TYPE == "LOG_COSH":
            model.compile(loss=log_cosh, optimizer=opt)
        elif c.LOSS_TYPE == "MAE":
            model.compile(loss=mean_absolute_error, optimizer=opt)
        elif c.LOSS_TYPE == "HINGE":
            model.compile(loss=hinge, optimizer=opt)
        elif c.LOSS_TYPE == "POISSON":
            model.compile(loss=poisson, optimizer=opt)
        elif c.LOSS_TYPE == "SQUARED_HINGE":
            model.compile(loss=squared_hinge, optimizer=opt)
        elif c.LOSS_TYPE == "NLL":
            model.compile(loss=negative_log_likelihood, optimizer=opt)
    return model


def get_model():
    model = None
    if c.SINGLE_FLG:
        print("SINGLE_FLG=TRUE")
        # 複数GPUを使用しない CPU用
        if c.LOAD_TYPE == 1:
            if c.LOSS_TYPE == "RMSE":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD,
                                                   custom_objects={"root_mean_squared_error": root_mean_squared_error})
            elif c.LOSS_TYPE == "MSEC":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                    "mean_squared_error_custome": mean_squared_error_custome})
            elif c.LOSS_TYPE == "FXMSE":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD,
                                                   custom_objects={"fx_mean_squared_error": fx_mean_squared_error})
            elif c.LOSS_TYPE == "FXMSE2":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD,
                                                   custom_objects={"fx_mean_squared_error2": fx_mean_squared_error2})
            elif c.LOSS_TYPE == "IE":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD,
                                                   custom_objects={"fx_insensitive_error": fx_insensitive_error})
            elif c.LOSS_TYPE == "NLL":
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD,
                                                   custom_objects={"negative_log_likelihood": negative_log_likelihood})
            else:
                model = tf.keras.models.load_model(c.MODEL_DIR_LOAD)

        elif c.LOAD_TYPE == 2:
            # 重さのみロード
            if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                    c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                    c.METHOD == "TCN" or c.METHOD == "TCN7":
                model = create_model_lstm()
            elif c.METHOD == "NORMAL":
                model = create_model_normal()
            model.load_weights(c.LOAD_CHK_PATH)
        else:
            # 新規作成
            if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                    c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                    c.METHOD == "TCN" or c.METHOD == "TCN7":
                model = create_model_lstm()
            elif c.METHOD == "NORMAL":
                model = create_model_normal()

    else:
        # モデル作成
        with tf.distribute.MirroredStrategy().scope():
            # 複数GPU使用する
            # https://qiita.com/ytkj/items/18b2910c3363b938cde4
            if c.LOAD_TYPE == 1:
                if c.LOSS_TYPE == "RMSE":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "root_mean_squared_error": root_mean_squared_error})
                elif c.LOSS_TYPE == "MSEC":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "mean_squared_error_custome": mean_squared_error_custome})
                elif c.LOSS_TYPE == "FXMSE":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "fx_mean_squared_error": fx_mean_squared_error})
                elif c.LOSS_TYPE == "FXMSE2":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "fx_mean_squared_error2": fx_mean_squared_error2})
                elif c.LOSS_TYPE == "IE":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "fx_insensitive_error": fx_insensitive_error})
                elif c.LOSS_TYPE == "NLL":
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD, custom_objects={
                        "negative_log_likelihood": negative_log_likelihood})
                else:
                    model = tf.keras.models.load_model(c.MODEL_DIR_LOAD)
            elif c.LOAD_TYPE == 2:
                # 重さのみロード
                if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                        c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or c.METHOD == "TCN" or c.METHOD == "TCN7":
                    model = create_model_lstm()
                elif c.METHOD == "NORMAL":
                    model = create_model_normal()

                model.load_weights(c.LOAD_CHK_PATH)
            else:
                # 新規作成
                if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                        c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or c.METHOD == "TCN" or c.METHOD == "TCN7" :
                    model = create_model_lstm()
                elif c.METHOD == "NORMAL":
                    model = create_model_normal()

    model.summary()

    return model


# callbacks.append(CSVLogger("history.csv"))
# look
# https://qiita.com/yukiB/items/f45f0f71bc9739830002

def make_data(conf, start, end, test_flg, eval_flg, target_spread_list=[], target_spread_percent_list=[]):
    global c
    c = conf

    dataSequence2 = DataSequence2(c, start, end, test_flg, eval_flg, target_spread_list=target_spread_list, target_spread_percent_list=target_spread_percent_list)
    print("DataSequence Init End!!")

    return dataSequence2


def do_train(conf, dataSequence2, dataSequence2_eval):
    global c
    c = conf

    if c.SINGLE_FLG:
        os.environ["CUDA_VISIBLE_DEVICES"] = c.DEVICE

    # 乱数を固定して学習の再現性を保つ
    # see:http://tomo-techblog.com/tensorflowgpu/
    np.random.seed(c.SEED)
    rn.seed(c.SEED)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    tf.compat.v1.set_random_seed(c.SEED)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    c.print_load_info()

    if os.path.isdir(c.MODEL_DIR):
        # 既にモデル保存ディレクトリがある場合はLEARNING_NUMが間違っているのでエラー
        print("ERROR!! MODEL_DIR Already Exists ")
        exit(1)

    makedirs(c.HISTORY_DIR)
    makedirs(c.CHK_DIR)

    # 処理時間計測
    t1 = time.time()
    model = get_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=c.CHK_DIR + "/{epoch:04d}",
        verbose=0,
        save_weights_only=True, )

    # see: http://tech.wonderpla.net/entry/2017/10/24/110000
    # max_queue_size：データ生成処理を最大いくつキューイングしておくかという設定
    # use_multiprocessing:Trueならマルチプロセス、Falseならマルチスレッドで並列処理
    # workers:1より大きい数字を指定すると並列処理を実施

    use_multiprocessing = True if c.WORKERS != 0 else False
    hist = model.fit_generator(dataSequence2,
                               validation_data=dataSequence2_eval,
                               # validation_steps = dataSequence2_eval.__len__(),
                               steps_per_epoch=dataSequence2.__len__(),
                               epochs=c.EPOCH,
                               max_queue_size=c.MAX_QUEUE_SIZE,
                               use_multiprocessing=use_multiprocessing,
                               workers=c.WORKERS,
                               verbose=2,
                               shuffle=False,
                               # verbose=1,
                               callbacks=[tf.keras.callbacks.CSVLogger(
                                   filename=c.HISTORY_DIR + "/history.csv",
                                   append=False),
                                   cp_callback
                               ],
                               )

    # SavedModel形式で保存
    model.save(c.MODEL_DIR)

    # 全学習おわり
    print("total learning take:", time.time() - t1)

    # 学習結果（損失）のグラフを描画
    if hist is not None:
        try:
            png_dir = "/app/bin_op/png/"
            # png保存用のディレクトリ作成
            plog_save_dir = png_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
            makedirs(plog_save_dir)
            print("PNG SAVE DIR:", plog_save_dir)
            filename = plog_save_dir + "/training_history.png"
            # 損失の履歴をプロット
            fig = plt.figure()
            plt.plot(hist.history['loss'], color='r')  # red
            plt.plot(hist.history['val_loss'], color='b')  # blue
            plt.title('model loss')
            # plt.show()
            fig.savefig(filename)
        except Exception as e:
            print(tracebackPrint(e))

    # セッション終了
    K.clear_session()

    print("END")


if __name__ == '__main__':
    conf = conf_class.ConfClass()
    # conf.change_real_spread_flg(False)
    print("FILE_PREFIX", conf.FILE_PREFIX)

    start = datetime(2016, 1, 1, )
    end = datetime(2021, 1, 1)

    start_eval = datetime(2021, 1, 1, )
    end_eval = datetime(2022, 5, 1, )

    dataSequence2_eval = make_data(conf, start_eval, end_eval, True, True, conf.TARGET_SPREAD_LISTS)
    dataSequence2 = make_data(conf, start, end, False, False, conf.TARGET_SPREAD_LISTS)

    do_train(conf, dataSequence2, dataSequence2_eval)
    # 終わったらメールで知らせる
    mail.send_message(host, ": lstm_do finished!!!")
