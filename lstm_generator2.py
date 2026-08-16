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
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW,RectifiedAdam,LazyAdam

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

from keras_self_attention import SeqSelfAttention
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from testLstmFX2_eval import do_eval


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

# コンピュータ名を取得
host = socket.gethostname()

c = None

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

            if c.FRAGMENT_NUM != None:
                tmp_length = c.FRAGMENTS_INPUT_LEN
            else:
                tmp_length =length

            for a in c.ADDITIONAL_DATA_LIST:
                tmp_length += a["input_len"]

            input = keras.Input(shape=(tmp_length,))

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
        if c.FRAGMENT_NUM != None:
            tmp_length = c.FRAGMENTS_INPUT_LEN
        else:
            tmp_length = c.INPUT_LEN[0]

        for a in c.ADDITIONAL_DATA_LIST:
            tmp_length += a["input_len"]

        input = keras.Input(shape=(tmp_length,))

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
            tf.keras.losses.c
        elif c.LOSS_TYPE == "C-ENTROPY-FOCAL":
            model.compile(loss=categorical_focal_loss(alpha=[c.C_ENTROPY_FOCAL_AlPHA], gamma=c.C_ENTROPY_FOCAL_GAMMA), optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])

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
        elif c.LOSS_TYPE == "FXMSE3":
            model.compile(loss=fx_mean_squared_error3, optimizer=Adam(lr=c.LEARNING_RATE))
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


def get_activation(activation, layer):

    if activation == 'leaky_relu':
        layer = Activation(tf.nn.leaky_relu)(layer)
    elif activation == 'relu6':
        layer = Activation(tf.nn.relu6)(layer)
    elif activation == 'crelu':
        layer = Activation(tf.nn.crelu)(layer)
    else:
        layer = Activation(activation)(layer)

    return layer

def get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=False, return_state=False ):
    if lstm_type in ["LSTM" ,"Bi", "KSA-LSTM", "LSTM-KSA", "MHA-LSTM", "LSTM-MHA","LSTM-KSA-CNN","LSTM-MHA-CNN", "LSTM-ATTENTION", "LSTM-ATTENTION-LSTM","CNN-LSTM", "SimpleRNN" ]:
        return keras.layers.LSTM(lstm_unit,
                                     activation=config.RNN_ACTIVATION,
                                     recurrent_activation=config.RNN_REC_ACTIVATION,
                                     kernel_initializer=k_i,
                                     recurrent_initializer=r_i,
                                     kernel_regularizer=l2_k,
                                     recurrent_regularizer=l2_r,
                                     dropout=config.L_DO,
                                     recurrent_dropout=config.L_RDO,
                                     return_sequences=return_sequences,
                                     return_state=return_state,
                                 )

    elif lstm_type == "LAYERNORM":
        return keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(lstm_unit,
                                                            kernel_initializer=k_i,
                                                            recurrent_initializer = r_i,
                                                            kernel_regularizer = l2_k,
                                                            recurrent_regularizer = l2_r,
                                                            dropout=config.L_DO,
                                                            recurrent_dropout=config.L_RDO,
                                                          ),
                                     return_sequences=return_state)

    elif lstm_type == "PEEPHOLE":
        return keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(lstm_unit,
                                                          kernel_initializer=k_i,
                                                          recurrent_initializer=r_i,
                                                          kernel_regularizer=l2_k,
                                                          recurrent_regularizer=l2_r,
                                                          dropout=config.L_DO,
                                                          recurrent_dropout=config.L_RDO,
                                                          ),
                                     return_sequences=return_state)
    elif lstm_type == "SimpleRNN":
        return keras.layers.SimpleRNN(lstm_unit,
                                activation=config.RNN_ACTIVATION,
                                recurrent_activation=config.RNN_REC_ACTIVATION,
                                kernel_initializer=k_i,
                                recurrent_initializer=r_i,
                                kernel_regularizer=l2_k,
                                recurrent_regularizer=l2_r,
                                dropout=config.L_DO,
                                recurrent_dropout=config.L_RDO,
                                return_sequences=return_state,
                                return_state=return_state,)

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
                                return_sequences=return_state,
                                return_state=return_state,)
    elif lstm_type == "CuDNNLSTM":
        return tf.compat.v1.keras.layers.CuDNNLSTM(lstm_unit,
                                                   kernel_initializer=k_i,
                                                   recurrent_initializer=r_i,
                                                   kernel_regularizer=l2_k,
                                                   recurrent_regularizer=l2_r,
                                                   dropout=config.L_DO,
                                                   recurrent_dropout=config.L_RDO,
                                                   return_sequences=return_state)

    elif lstm_type == "QRNN":
        return QRNN(lstm_unit, window_size=c.WINDOW_SIZE, return_sequences=return_state)

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

def get_cnn(config, input,lstm_unit,):
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

class ATTENTION(tf.keras.Model):
  def __init__(self, conf):
    super(ATTENTION, self).__init__()

    self.conf = conf
    self.FEATURE_D_LIST = []
    self.STATE_D_LIST = []

    self.USE_STATE = conf.LSTM_ATTENTION_CONF["USE_STATE"]
    self.FEATURE_DENSE = conf.LSTM_ATTENTION_CONF["FEATURE_DENSE"]
    self.ATTENTION_CNT = conf.LSTM_ATTENTION_CONF["ATTENTION_CNT"]

    for d in self.FEATURE_DENSE:
      self.FEATURE_D_LIST.append(tf.keras.layers.Dense(d))

      if self.USE_STATE:
        self.STATE_D_LIST.append(tf.keras.layers.Dense(d))

    self.ATTENTION_D = tf.keras.layers.Dense(self.ATTENTION_CNT)

  def call(self, inp):
    features, hidden = inp

    for i, fd in enumerate(self.FEATURE_D_LIST):
      if i == 0:
        fd_output = fd(features)
      else:
        fd_output = fd(fd_output)
      #fd_output = BatchNormalization()(fd_output)
      fd_output =  get_activation(self.conf.DENSE_ACTIVATION, fd_output)

    if self.USE_STATE:
      hidden_expand = tf.expand_dims(hidden, 1)#shape=(BATCH_SIZE, 1, LSTM_HIDDEN)

      for i, sd in enumerate(self.STATE_D_LIST):
        if i == 0:
          sd_output = sd(hidden_expand)
        else:
          sd_output = sd(sd_output)
        #sd_output = BatchNormalization()(sd_output)
        sd_output = get_activation(self.conf.DENSE_ACTIVATION, sd_output)

      fd_output = tf.nn.tanh(fd_output + sd_output) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_UNIT)

    ad_output = self.ATTENTION_D(fd_output)
    #ad_output = BatchNormalization()(ad_output)
    ad_output = Activation(tf.nn.relu)(ad_output) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_CNT)

    attention_weights = tf.nn.softmax(ad_output, axis=1) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_CNT)

    if self.ATTENTION_CNT > 1:
        context_list = []
        for i in range(self.ATTENTION_CNT):
          tmp = tf.expand_dims(attention_weights[:, :, i], axis=2) * features

          if self.conf.LSTM_TYPE == "LSTM-ATTENTION":
              context_vector = tf.reduce_sum(tmp, axis=1)  # shape=(BATCH_SIZE, LSTM_HIDDEN)
              context_list.append(context_vector)

          elif self.conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
              context_list.append(tmp)

        if self.conf.LSTM_TYPE == "LSTM-ATTENTION":
            concate = tf.keras.layers.Concatenate(axis=1)(context_list) #shape=(BATCH_SIZE, ATTENTION_CNT * LSTM_HIDDEN)

        elif self.conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
            concate = tf.keras.layers.Concatenate(axis=2)(context_list)

        return concate

    elif self.ATTENTION_CNT == 1:

        tmp = tf.expand_dims(attention_weights[:, :, 0], axis=2) * features

        if self.conf.LSTM_TYPE == "LSTM-ATTENTION":
            context_vector = tf.reduce_sum(tmp, axis=1)  # shape=(BATCH_SIZE, LSTM_HIDDEN)
            return context_vector

        elif self.conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
            return tmp
def attention(conf, inp):
    features, hidden = inp

    FEATURE_D_LIST = []
    STATE_D_LIST = []

    USE_STATE = conf.LSTM_ATTENTION_CONF["USE_STATE"]
    FEATURE_DENSE = conf.LSTM_ATTENTION_CONF["FEATURE_DENSE"]
    ATTENTION_CNT = conf.LSTM_ATTENTION_CONF["ATTENTION_CNT"]

    for d in FEATURE_DENSE:
      FEATURE_D_LIST.append(tf.keras.layers.Dense(d))

      if USE_STATE:
        STATE_D_LIST.append(tf.keras.layers.Dense(d))

    ATTENTION_D = tf.keras.layers.Dense(ATTENTION_CNT)

    for i, fd in enumerate(FEATURE_D_LIST):
      if i == 0:
        fd_output = fd(features)
      else:
        fd_output = fd(fd_output)
      #fd_output = BatchNormalization()(fd_output)
      fd_output =  get_activation(conf.DENSE_ACTIVATION, fd_output)

    if USE_STATE:
      hidden_expand = tf.expand_dims(hidden, 1)#shape=(BATCH_SIZE, 1, LSTM_HIDDEN)

      for i, sd in enumerate(STATE_D_LIST):
        if i == 0:
          sd_output = sd(hidden_expand)
        else:
          sd_output = sd(sd_output)
        #sd_output = BatchNormalization()(sd_output)
        sd_output = get_activation(conf.DENSE_ACTIVATION, sd_output)

      fd_output = tf.nn.tanh(fd_output + sd_output) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_UNIT)

    ad_output = ATTENTION_D(fd_output)
    #ad_output = BatchNormalization()(ad_output)
    ad_output = Activation(tf.nn.relu)(ad_output) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_CNT)

    attention_weights = tf.nn.softmax(ad_output, axis=1) #shape=(BATCH_SIZE, INPUT_LEN, ATTENTION_CNT)

    if ATTENTION_CNT > 1:
        context_list = []
        for i in range(ATTENTION_CNT):
          tmp = tf.expand_dims(attention_weights[:, :, i], axis=2) * features

          if conf.LSTM_TYPE == "LSTM-ATTENTION":
              context_vector = tf.reduce_sum(tmp, axis=1)  # shape=(BATCH_SIZE, LSTM_HIDDEN)
              context_list.append(context_vector)

          elif conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
              context_list.append(tmp)

        if conf.LSTM_TYPE == "LSTM-ATTENTION":
            concate = tf.keras.layers.Concatenate(axis=1)(context_list) #shape=(BATCH_SIZE, ATTENTION_CNT * LSTM_HIDDEN)

        elif conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
            concate = tf.keras.layers.Concatenate(axis=2)(context_list)

        return concate

    elif ATTENTION_CNT == 1:

        tmp = tf.expand_dims(attention_weights[:, :, 0], axis=2) * features

        if conf.LSTM_TYPE == "LSTM-ATTENTION":
            context_vector = tf.reduce_sum(tmp, axis=1)  # shape=(BATCH_SIZE, LSTM_HIDDEN)
            return context_vector

        elif conf.LSTM_TYPE == "LSTM-ATTENTION-LSTM":
            return tmp

def make_layer(lstms, input, lstm_layer_num, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, config):
    if lstm_type in ["LSTM", "GRU", "QRNN", "CuDNNLSTM", "LAYERNORM", "PEEPHOLE", "SimpleRNN"]:
        if lstm_layer_num == 1:
            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=c.RETURN_SEQ_STR)(input)

            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)

            if c.LSTM_ACTIVATION != "":
                tmp_layer = get_activation(c.LSTM_ACTIVATION, tmp_layer)

            if c.LSTM_DO > 0:
                tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

            if c.RETURN_SEQ_STR:
                tmp_layer = keras.layers.Flatten()(tmp_layer)

        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(input)
                elif j == lstm_layer_num -1:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=c.RETURN_SEQ_STR)(tmp_layer)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

                if c.LSTM_ACTIVATION != "":
                    tmp_layer = get_activation(c.LSTM_ACTIVATION, tmp_layer)

                if c.LSTM_DO > 0:
                    tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

            if c.RETURN_SEQ_STR:
                tmp_layer = keras.layers.Flatten()(tmp_layer)

        lstms.append(tmp_layer)

    elif lstm_type in ["LSTM-ATTENTION", "LSTM-ATTENTION-LSTM"]:
        if lstm_layer_num == 1:
            feature, h, cell = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True, return_state=True)(input)

            #feature = ATTENTION(c)([feature, h])  # ここにAttentionレイヤを挟む
            feature = attention(c, [feature, h])

            if lstm_type == "LSTM-ATTENTION-LSTM":
                feature = get_lstm(config, lstm_type, c.LSTM_ATTENTION_CONF["LSTM_UNIT"], k_i, r_i, l2_k, l2_r, return_sequences=False,return_state=False)(feature)

            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                feature = BatchNormalization()(feature)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                feature = LayerNormalization()(feature)

            if c.LSTM_ACTIVATION != "":
                feature = get_activation(c.LSTM_ACTIVATION, feature)

            if c.LSTM_DO > 0:
                feature = keras.layers.Dropout(c.LSTM_DO)(feature)

        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    feature = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True, return_state=False)(input)
                elif j == lstm_layer_num -1:
                    feature, h, cell = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True, return_state=False)(feature)
                else:
                    feature = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True, return_state=True)(feature)

                #feature = ATTENTION(c)([feature, h])  # ここにAttentionレイヤを挟む
                feature = attention(c, [feature, h])

                if lstm_type == "LSTM-ATTENTION-LSTM":
                    feature = get_lstm(config, lstm_type, c.LSTM_ATTENTION_CONF["LSTM_UNIT"], k_i, r_i, l2_k, l2_r, return_sequences=False, return_state=False)(feature)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    feature = BatchNormalization()(feature)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    feature = LayerNormalization()(feature)

                if c.LSTM_ACTIVATION != "":
                    feature = get_activation(c.LSTM_ACTIVATION, feature)

                if c.LSTM_DO > 0:
                    feature = keras.layers.Dropout(c.LSTM_DO)(feature)

        lstms.append(feature)

    elif lstm_type in ["LSTM-KSA", "LSTM-KSA-CNN","LSTM-MHA", "LSTM-MHA-CNN"]:
        if lstm_layer_num == 1:
            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(input)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(input)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(tmp_layer)

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

            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=False)(tmp_layer)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == lstm_layer_num -1:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=False)(tmp_layer)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

        lstms.append(tmp_layer)

    elif lstm_type == "Bi":
        if lstm_layer_num == 1:
            tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=False))(
                    input)

            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)

            if c.LSTM_ACTIVATION != "":
                tmp_layer = get_activation(c.LSTM_ACTIVATION, tmp_layer)

            if c.LSTM_DO > 0:
                tmp_layer = keras.layers.Dropout(c.LSTM_DO)(tmp_layer)

            lstms.append(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True))(
                        input)
                elif j != 0 and j != (lstm_layer_num - 1):
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True))(
                        tmp_layer)
                elif j == (lstm_layer_num - 1):
                    tmp_layer = keras.layers.Bidirectional(get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=False))(
                            tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

                if c.LSTM_ACTIVATION != "":
                    tmp_layer = get_activation(c.LSTM_ACTIVATION, tmp_layer)

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

    elif lstm_type in ["CNN-LSTM"]:
        tmp_layer = layers.Conv1D(filters=config.CNN_UNIT_NUM, kernel_size=config.CNN_KERNEL_SIZE, activation="relu", strides=1, padding='valid')(input)
        if config.CNN_NORMAL == "BATCH":
            tmp_layer = BatchNormalization()(tmp_layer)
        elif config.CNN_NORMAL == "LAYER":
            tmp_layer = LayerNormalization()(tmp_layer)

        if lstm_layer_num == 1:
            tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=config.RETURN_SEQ)(tmp_layer)
            if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                tmp_layer = BatchNormalization()(tmp_layer)
            elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                tmp_layer = LayerNormalization()(tmp_layer)

            if c.RETURN_SEQ_STR:
                tmp_layer = keras.layers.Flatten()(tmp_layer)
        else:
            for j in range(lstm_layer_num):
                if j == 0:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(input)
                elif j == (lstm_layer_num - 1):
                    tmp_layer = get_tcn(lstm_unit, config, config.RETURN_SEQ)(tmp_layer)
                else:
                    tmp_layer = get_lstm(config, lstm_type, lstm_unit, k_i, r_i, l2_k, l2_r, return_sequences=True)(tmp_layer)

                if c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                    tmp_layer = BatchNormalization()(tmp_layer)
                elif c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                    tmp_layer = LayerNormalization()(tmp_layer)

            if c.RETURN_SEQ_STR:
                tmp_layer = keras.layers.Flatten()(tmp_layer)

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

    if len(c.TRANSFER_CONF) == 0:
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
                    if c.FRAGMENT_NUM != None:
                        tmp_length = c.FRAGMENTS_INPUT_LEN
                    else:
                        tmp_length = c.INPUT_LEN[i]

                    for a in c.ADDITIONAL_DATA_LIST:
                        tmp_length += a["input_len"]

                    input = keras.Input(shape=(tmp_length, 1))

                    inputs.append(input)

                    if c.TIME_DISTRIBUTED != 0:
                        input = TimeDistributed(keras.layers.Dense(1))(input)

                else:
                    if c.FRAGMENT_NUM != None:
                        tmp_length = c.FRAGMENTS_INPUT_LEN
                    else:
                        tmp_length = c.INPUT_LEN[i]

                    for a in c.ADDITIONAL_DATA_LIST:
                        tmp_length += a["input_len"]

                    input = keras.Input(shape=(tmp_length, len(ipt_lists)))

                    inputs.append(input)

                    if c.TIME_DISTRIBUTED != 0:
                        input = TimeDistributed(keras.layers.Dense(len(ipt_lists)))(input)

                if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or \
                        c.METHOD == "LSTM6" or c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                        c.METHOD == "TCN" or c.METHOD == "TCN7":
                    make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[i], K_I, R_I, l2_K, l2_R, c)

            else:
                for j in range(len(ipt_lists)):
                    if c.FRAGMENT_NUM != None:
                        tmp_length = c.FRAGMENTS_INPUT_LEN
                    else:
                        tmp_length = c.INPUT_LEN[i]

                    for a in c.ADDITIONAL_DATA_LIST:
                        tmp_length += a["input_len"]

                    input = keras.Input(shape=(tmp_length, 1))

                    inputs.append(input)

                    if c.TIME_DISTRIBUTED != 0:
                        input = TimeDistributed(keras.layers.Dense(1))(input)

                    if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
                            c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or \
                            c.METHOD == "TCN" or c.METHOD == "TCN7":
                        make_layer(lstms, input, c.LSTM_LAYER_NUM, c.LSTM_TYPE, c.LSTM_UNIT[i], K_I, R_I, l2_K, l2_R, c)

    else:
        #転移学習の場合
        base_load_dir = c.MODEL_DIR_PARENT + c.TRANSFER_CONF["MN"]
        delete_layers = c.TRANSFER_CONF["DELETE_LAYERS"]
        last_layer = c.TRANSFER_CONF["LAST_LAYER"]

        base_model = tf.keras.models.load_model(base_load_dir,
                                                custom_objects={"root_mean_squared_error": root_mean_squared_error,
                                                                "fx_mean_squared_error": fx_mean_squared_error,
                                                                "fx_mean_squared_error2": fx_mean_squared_error2,
                                                                "fx_mean_squared_error3": fx_mean_squared_error3,
                                                                "mean_squared_error_custome": mean_squared_error_custome,
                                                                "fx_insensitive_error": fx_insensitive_error,
                                                                "negative_log_likelihood": negative_log_likelihood,
                                                                })

        for name in delete_layers:  # 何層まで再学習不可にするか。
            base_model.get_layer(name).trainable = False

        inputs = base_model.input
        lstms = []
        lstms.append(base_model.get_layer(last_layer).output)

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

    if c.METHOD == "LSTM7" and len(c.DENSE_UNIT_LSTM7) != 0:
        # 秒データをone-hotで入力
        sec_input = keras.Input(shape=(c.SEC_OH_LEN,))
        inputs.append(sec_input)

        # 分データをone-hotで入力
        min_input = keras.Input(shape=(c.MIN_OH_LEN,))
        inputs.append(min_input)

        # 時間データをone-hotで入力
        hour_input = keras.Input(shape=(c.HOUR_OH_LEN,))
        inputs.append(hour_input)

        lstm7_concate = keras.layers.Concatenate()([sec_input, min_input, hour_input])
        for i, unit in enumerate(c.DENSE_UNIT_LSTM7):

            lstm7_concate = keras.layers.Dense(c.DENSE_UNIT_LSTM7[i], )(lstm7_concate)

            if c.NORMAL_TYPE == "BATCH_NORMAL" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                lstm7_concate = BatchNormalization()(lstm7_concate)
            elif c.NORMAL_TYPE == "LAYER_NORMAL" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                lstm7_concate = LayerNormalization()(lstm7_concate)

            lstm7_concate = get_activation(c.DENSE_ACTIVATION, lstm7_concate)

        lstms.append(lstm7_concate)

    else:

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

    if len(c.HOR_DB_CORE_LIST) != 0:
        for n in range(c.HOR_DB_CORE_LIST):
            for i in range(c.HOR_LINE_NUM):
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
                mm_unit = l["unit"]
                for i, unit in enumerate(mm_unit):
                    mm_input = keras.layers.Dense(unit, kernel_initializer=D_I, kernel_regularizer=l2_D, )(mm_input)

                    if c.NORMAL_TYPE == "BATCH_NORMAL" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM" or c.NORMAL_TYPE == "BATCH_NORMAL_LSTM2":
                        mm_input = BatchNormalization()(mm_input)
                    elif c.NORMAL_TYPE == "LAYER_NORMAL" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
                        mm_input = LayerNormalization()(mm_input)

                    if c.DENSE_ACTIVATION == 'leaky_relu':
                        mm_input = Activation(tf.nn.leaky_relu)(mm_input)
                    elif c.DENSE_ACTIVATION == 'relu6':
                        mm_input = Activation(tf.nn.relu6)(mm_input)
                    elif c.DENSE_ACTIVATION == 'crelu':
                        mm_input = Activation(tf.nn.crelu)(mm_input)
                    else:
                        mm_input = Activation(c.DENSE_ACTIVATION)(mm_input)

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


    if len(c.TRANSFER_CONF) == 0:
        if len(lstms) > 1:
            concate = keras.layers.Concatenate()(lstms)
        else:
            concate = lstms[0]
    else:
        if len(lstms) > 1:
            concate = keras.layers.Concatenate(name="concate2")(lstms)
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
            #dense = BatchNormalization()(dense)
            dense = BatchNormalization(name="batch_normalization_dense_" + str(i))(dense)
        elif c.NORMAL_TYPE == "LAYER_NORMAL" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM" or c.NORMAL_TYPE == "LAYER_NORMAL_LSTM2":
            #dense = LayerNormalization()(dense)
            dense = LayerNormalization(name="layer_normalization_dense_" + str(i))(dense)

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
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I, dtype='float32')(dense)
    else:
        if c.MIXTURE_NORMAL:
            # denseのユニット数を計算
            # see https://qiita.com/pocokhc/items/be178d1d7deeeafac8c0
            params_size = tfp.layers.MixtureNormal.params_size(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))
            mn_output = keras.layers.Dense(params_size, activation=None)(concate)  # 指定のDense層を追加(activationはNone)
            output = tfp.layers.MixtureNormal(c.MIXTURE_NORMAL_NUM, (c.OUTPUT,))(mn_output)  # MixtureNormal層を最後に追加
        else:
            output = keras.layers.Dense(c.OUTPUT, activation=activ, kernel_initializer=O_I, dtype='float32')(concate)


    if c.MIXTURE_NORMAL:
        model = keras.Model(inputs=inputs, outputs=output)
    else:
        model = keras.Model(inputs=inputs, outputs=[output])

    opt = Adam(lr=c.LEARNING_RATE)
    if c.OPT == "ADABOUND":
        opt = AdaBound(lr=c.LEARNING_RATE, amsbound=False, )
    elif c.OPT == "AMSBOUND":
        opt = AdaBound(lr=c.LEARNING_RATE, amsbound=True, )

    elif c.OPT == "AdamW":
        opt = AdamW(lr=c.LEARNING_RATE,weight_decay=c.OPT_OPT)
    elif c.OPT == "RectifiedAdam":
        opt = RectifiedAdam(lr=c.LEARNING_RATE,)
    elif c.OPT == "LazyAdam":
        opt = LazyAdam(lr=c.LEARNING_RATE,)
    elif c.OPT == "Adamax":
        opt = Adamax(lr=c.LEARNING_RATE,)
    elif c.OPT == "Nadam":
        opt = Nadam(lr=c.LEARNING_RATE,)
    elif c.OPT == "SGD":
        opt = SGD(lr=c.LEARNING_RATE,momentum=c.OPT_OPT)

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
        elif c.LOSS_TYPE == "C-ENTROPY-FOCAL":
            model.compile(loss=categorical_focal_loss(alpha=[c.C_ENTROPY_FOCAL_AlPHA], gamma=c.C_ENTROPY_FOCAL_GAMMA), optimizer=opt, metrics=['accuracy'])


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
        elif c.LOSS_TYPE == "FXMSE3":
            model.compile(loss=fx_mean_squared_error3, optimizer=opt)
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


def get_new_model():
    # 新規作成
    if c.METHOD == "LSTM" or c.METHOD == "LSTM2" or c.METHOD == "LSTM3" or c.METHOD == "LSTM4" or c.METHOD == "LSTM5" or c.METHOD == "LSTM6" or \
            c.METHOD == "LSTM7" or c.METHOD == "LSTM8" or c.METHOD == "LSTM9" or c.METHOD == "LSTM10" or c.METHOD == "TCN" or c.METHOD == "TCN7" :
        model = create_model_lstm()
    elif c.METHOD == "NORMAL":
        model = create_model_normal()

    return model

def get_load_model(MODEL_LOAD_DIR, conf):
    custom_objects = {"root_mean_squared_error": root_mean_squared_error,
                      "fx_mean_squared_error": fx_mean_squared_error,
                      "fx_mean_squared_error2": fx_mean_squared_error2,
                      "fx_mean_squared_error3": fx_mean_squared_error3,
                      "mean_squared_error_custome": mean_squared_error_custome,
                      "fx_insensitive_error": fx_insensitive_error,
                      "negative_log_likelihood": negative_log_likelihood,
                      "categorical_focal_loss_fixed": categorical_focal_loss(alpha=[c.C_ENTROPY_FOCAL_AlPHA],
                                                                             gamma=c.C_ENTROPY_FOCAL_GAMMA),
                      "AdaBound": AdaBound,
                      }

    model = tf.keras.models.load_model(MODEL_LOAD_DIR, custom_objects=custom_objects)

    return model

def make_data(conf, start, end, test_flg, eval_flg, target_spread_list=[], target_spread_percent_list=[]):
    global c
    c = conf

    dataSequence2 = DataSequence2(c, start, end, test_flg, eval_flg, target_spread_list=target_spread_list, target_spread_percent_list=target_spread_percent_list, drop_last=c.DROP_LAST)
    print("DataSequence Init End!!")

    return dataSequence2


def do_train(conf, dataSequence2, dataSequence2_eval):
    # 処理時間計測
    t1 = time.time()

    global c
    c = conf

    if c.DEVICE == 'GPU':

        # CUDAが使用するGPUを設定
        if conf.DEVICE_CNT == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif conf.DEVICE_CNT == 2:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
            exit(1)

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

    # 乱数を固定して学習の再現性を保つ
    # see:http://tomo-techblog.com/tensorflowgpu/
    np.random.seed(c.SEED)
    rn.seed(c.SEED)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    tf.compat.v1.set_random_seed(c.SEED)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    if len(c.LOAD_CONF_DICT) == 0:
        #新規作成
        LAST_EPOCH_NO = c.EPOCH

        EPOCH = c.EPOCH

        initial_epoch = 0

        LAST_EPOCH_NO = EPOCH

        MODEL_SAVE_DIR = c.MODEL_DIR_PARENT + c.MN + "-" + str(LAST_EPOCH_NO)
        HISTORY_DIR = c.HISTORY_DIR_PARENT + c.MN

        if os.path.isdir(MODEL_SAVE_DIR):
            # 既にモデル保存ディレクトリがある場合は間違っているのでエラー
            print("ERROR!! MODEL_SAVE_DIR Already Exists ")
            exit(1)

        if os.path.isdir(HISTORY_DIR):
            print("ERROR!! HISTORY_DIR Already Exists ")
            exit(1)

        print("HISTORY_DIR", HISTORY_DIR)

        makedirs(HISTORY_DIR)

    else:
        #既存モデルロード
        c.MN = "MN" + str(c.LOAD_CONF_DICT["MN"])
        LOAD_EPOCH = c.LOAD_CONF_DICT["LOAD_EPOCH"]
        EPOCH = c.LOAD_CONF_DICT["EPOCH"]

        #train_listaをエポック分シャッフルする
        for j in range(LOAD_EPOCH + 2):
            dataSequence2.rotate_train_list()

        initial_epoch = LOAD_EPOCH

        LAST_EPOCH_NO = LOAD_EPOCH + EPOCH

        MODEL_LOAD_DIR = c.MODEL_DIR_PARENT + c.MN + "-" + str(LOAD_EPOCH)

        MODEL_SAVE_DIR = c.MODEL_DIR_PARENT + c.MN + "-" + str(LAST_EPOCH_NO)
        HISTORY_DIR = c.HISTORY_DIR_PARENT + c.MN

        makedirs(HISTORY_DIR)

        if os.path.isdir(HISTORY_DIR) == False:
            print("ERROR!! HISTORY_DIR Not Exists ")
            exit(1)

    print("FILE_PREFIX:", c.FILE_PREFIX)
    print("MN:", c.MN)

    #AMP設定
    if c.MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if c.DEVICE == 'CPU' or (c.DEVICE == 'GPU' and c.DEVICE_CNT == 1):
        if len(c.LOAD_CONF_DICT) == 0:
            model = get_new_model()
        else:
            model = get_load_model(MODEL_LOAD_DIR, conf)
    else:
        # 複数GPU使用する
        # https://qiita.com/ytkj/items/18b2910c3363b938cde4
        with tf.distribute.MirroredStrategy().scope():
            if len(c.LOAD_CONF_DICT) == 0:
                model = get_new_model()

            else:
                model = get_load_model(MODEL_LOAD_DIR, conf)

    model.summary()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=c.MODEL_DIR_PARENT + c.MN + "-" + "{epoch:d}",
        verbose=0,
        save_weights_only=False, )

    # see: http://tech.wonderpla.net/entry/2017/10/24/110000
    # max_queue_size：データ生成処理を最大いくつキューイングしておくかという設定
    # use_multiprocessing:Trueならマルチプロセス、Falseならマルチスレッドで並列処理
    # workers:1より大きい数字を指定すると並列処理を実施

    use_multiprocessing = True if c.WORKERS != 0 else False
    hist = model.fit_generator(dataSequence2,
                               initial_epoch=initial_epoch,
                               validation_data=dataSequence2_eval,
                               steps_per_epoch=dataSequence2.__len__(),
                               epochs=LAST_EPOCH_NO,
                               max_queue_size=c.MAX_QUEUE_SIZE,
                               use_multiprocessing=use_multiprocessing,
                               workers=c.WORKERS,
                               verbose=2,
                               shuffle=False,
                               # verbose=1,
                               callbacks=[tf.keras.callbacks.CSVLogger(
                                   filename=HISTORY_DIR + "/history.csv",
                                   append=True),
                                   cp_callback,
                               ],
                               )

    # SavedModel形式で保存
    model.save(MODEL_SAVE_DIR)

    # 全学習おわり
    total_t = time.time() - t1
    print("total learning take:", total_t)
    print("epoch learning take:", total_t / c.EPOCH)


    if c.EVAL_FLG:
        print(datetime.now(),"eval start")

        model_suffix = [str(i + 1) for i in range(c.EPOCH)]
        do_eval(c ,file=c.MN ,model_suffix=model_suffix)

        print(datetime.now(),"eval end")

    """
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
            if c.EVAL_FLG:
                plt.plot(hist.history['val_loss'], color='b')  # blue
            plt.title('model loss')
            # plt.show()
            fig.savefig(filename)
        except Exception as e:
            print(tracebackPrint(e))
    """

    # セッション終了
    K.clear_session()

    print("END")


if __name__ == '__main__':
    conf = conf_class.ConfClass()
    # conf.change_real_spread_flg(False)
    print("FILE_PREFIX", conf.FILE_PREFIX)

    """
    start = datetime(2016, 1, 1, )
    end = datetime(2021, 1, 1)

    start_eval = datetime(2021, 1, 1, )
    end_eval = datetime(2022, 5, 1, )

    dataSequence2_eval = make_data(conf, start_eval, end_eval, True, True, conf.TARGET_SPREAD_LISTS)
    dataSequence2 = make_data(conf, start, end, False, False, conf.TARGET_SPREAD_LISTS)

    do_train(conf, dataSequence2, dataSequence2_eval)
    # 終わったらメールで知らせる
    mail.send_message(host, ": lstm_do finished!!!")
    """
