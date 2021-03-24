import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import huber, mean_squared_error
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

import time
import os

from logging import getLogger
from datetime import datetime
from datetime import timedelta
import time
from decimal import Decimal
from DataSequence2 import DataSequence2

from readConf2 import *
import tensorflow_probability as tfp

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

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)


#範囲つき予測(不当分散モデル)のための独自損失関数
#see:
#https://aotamasaki.hatenablog.com/entry/2019/03/01/185430
def loss(y_true,y_pred):
    # flat な1次元にする
    mu = K.reshape(y_pred[:,0],[-1])
    # 精度パラメーターβを導入
    # β = 1/σ(標準偏差)
    #beta = K.square(K.reshape(y_pred[:,1],[-1]))

    # β = logσ
    beta = K.exp(K.reshape(y_pred[:, 1], [-1]))

    y_true = K.reshape(y_true,[-1])

    dist = tfp.distributions.Normal(loc = mu, scale = beta)
    return K.mean(-1 * dist.log_prob(y_true) , axis=-1)

    #return K.mean(beta * K.square(mu - y_true) - K.log(beta), axis=-1)

# モデル作成
def create_model_normal():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))
    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        activ = 'softmax'
    else:
        activ = None

    l1_l2_K = None
    l1_l2_R = None
    if L_K_RATE != 0:
        l1_l2_K = tf.keras.regularizers.l1_l2(l1=L_K_RATE, l2=L_K_RATE)
    if L_R_RATE != 0:
        l1_l2_R = tf.keras.regularizers.l1_l2(l1=L_R_RATE, l2=L_R_RATE)

    init = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

    if len(INPUT_LEN) > 1:
        inputs = []
        for i, length in enumerate(INPUT_LEN):
            input = keras.Input(shape=(length, ))
            inputs.append(input)

        concate = keras.layers.Concatenate()(inputs)

        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(concate)
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(dense)
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(inputs)

        model = keras.Model(inputs=[inputs], outputs=[output])

    else:
        #inputが1種類の場合

        input = keras.Input(shape=(INPUT_LEN[0], ))
        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(input)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(dense)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(input)

        model = keras.Model(inputs=[input], outputs=[output])

    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

    elif LEARNING_TYPE == "REGRESSION_SIGMA":
        #範囲つき予測
        model.compile(loss=loss, optimizer=Adam(lr=LEARNING_RATE))
    elif LEARNING_TYPE == "REGRESSION":
        if LOSS_TYPE == "MSE":
            model.compile(loss=mean_squared_error, optimizer=Adam(lr=LEARNING_RATE))
        elif LOSS_TYPE == "HUBER":
            model.compile(loss=huber, optimizer=Adam(lr=LEARNING_RATE))

    return model

# モデル作成
def create_model_lstm():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))
    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        activ = 'softmax'
    else:
        activ = None

    l1_l2_K = None
    l1_l2_R = None
    if L_K_RATE != 0:
        l1_l2_K = tf.keras.regularizers.l1_l2(l1=L_K_RATE, l2=L_K_RATE)
    if L_R_RATE != 0:
        l1_l2_R = tf.keras.regularizers.l1_l2(l1=L_R_RATE, l2=L_R_RATE)

    init = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

    if len(LSTM_UNIT) > 1:
        lstms = []
        inputs = []
        for i, unit in enumerate(LSTM_UNIT):
            input = keras.Input(shape=(INPUT_LEN[i], 1))
            lstms.append(keras.layers.LSTM(LSTM_UNIT[i],  kernel_initializer = init,
                                           kernel_regularizer = l1_l2_K,
                                           recurrent_regularizer = l1_l2_R,
                                           #dropout=0.,
                                           #recurrent_dropout=0.,
                                           return_sequences=False)(input))
            inputs.append(input)

        concate = keras.layers.Concatenate()(lstms)

        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                              kernel_regularizer=l1_l2_K,)(concate) # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                              kernel_regularizer=l1_l2_K,)(dense) # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(concate)

        model = keras.Model(inputs=inputs, outputs=[output])

    else:
        #inputが1種類の場合

        input = keras.Input(shape=(INPUT_LEN[0], 1))
        lstm = keras.layers.LSTM(LSTM_UNIT[0], kernel_initializer = init,
                                         kernel_regularizer = l1_l2_K,
                                         recurrent_regularizer = l1_l2_R,
                                         #dropout=0.,
                                         #recurrent_dropout=0.,
                                         return_sequences=False)(input)

        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(lstm)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(dense)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(lstm)

        model = keras.Model(inputs=[input], outputs=[output])


    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

    elif LEARNING_TYPE == "REGRESSION_SIGMA":
        #範囲つき予測
        model.compile(loss=loss, optimizer=Adam(lr=LEARNING_RATE))
    elif LEARNING_TYPE == "REGRESSION":
        if LOSS_TYPE == "MSE":
            model.compile(loss=mean_squared_error, optimizer=Adam(lr=LEARNING_RATE))
        elif LOSS_TYPE == "HUBER":
            model.compile(loss=huber, optimizer=Adam(lr=LEARNING_RATE))

    return model


# モデル作成
def create_model_by():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))
    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        activ = 'softmax'
    else:
        activ = None

    l1_l2_K = None
    l1_l2_R = None
    if L_K_RATE != 0:
        l1_l2_K = tf.keras.regularizers.l1_l2(l1=L_K_RATE, l2=L_K_RATE)
    if L_R_RATE != 0:
        l1_l2_R = tf.keras.regularizers.l1_l2(l1=L_R_RATE, l2=L_R_RATE)

    init = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

    if len(LSTM_UNIT) > 1:
        lstms = []
        inputs = []
        for i, unit in enumerate(LSTM_UNIT):
            input = keras.Input(shape=(INPUT_LEN[i], 1))
            lstms.append(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNIT[i],  kernel_initializer = init,
                                           kernel_regularizer = l1_l2_K,
                                           recurrent_regularizer = l1_l2_R,
                                           #dropout=0.,
                                           #recurrent_dropout=0.,
                                           return_sequences=False))(input))
            inputs.append(input)

        concate = keras.layers.Concatenate()(lstms)

        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                              kernel_regularizer=l1_l2_K,)(concate) # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                              kernel_regularizer=l1_l2_K,)(dense) # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(concate)

        model = keras.Model(inputs=inputs, outputs=[output])

    else:
        #inputが1種類の場合

        input = keras.Input(shape=(INPUT_LEN[0], 1))
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNIT[0], kernel_initializer = init,
                                         kernel_regularizer = l1_l2_K,
                                         recurrent_regularizer = l1_l2_R,
                                         #dropout=0.,
                                         #recurrent_dropout=0.,
                                         return_sequences=False))(input)

        dense = None
        for i, unit in enumerate(DENSE_UNIT):
            if i == 0:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(lstm)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)
            else:
                dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                           kernel_regularizer=l1_l2_K, )(dense)  # 正則化： L2、
                if DROP > 0:
                    dense = keras.layers.Dropout(DROP)(dense)

        if dense != None:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(dense)
        else:
            output = keras.layers.Dense(OUTPUT, activation=activ, kernel_initializer = init)(lstm)

        model = keras.Model(inputs=[input], outputs=[output])


    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN":
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

    elif LEARNING_TYPE == "REGRESSION_SIGMA":
        #範囲つき予測
        model.compile(loss=loss, optimizer=Adam(lr=LEARNING_RATE))
    elif LEARNING_TYPE == "REGRESSION":
        if LOSS_TYPE == "MSE":
            model.compile(loss=mean_squared_error, optimizer=Adam(lr=LEARNING_RATE))
        elif LOSS_TYPE == "HUBER":
            model.compile(loss=huber, optimizer=Adam(lr=LEARNING_RATE))

    return model


def get_model(single_flg):
    model = None
    if single_flg:
        # 複数GPUを使用しない CPU用
        if LOAD_TYPE == 1:
            model = tf.keras.models.load_model(MODEL_DIR_LOAD)
        elif LOAD_TYPE == 2:
            #重さのみロード
            if METHOD == "LSTM":
                model = create_model_lstm()
            elif METHOD == "NORMAL":
                model = create_model_normal()
            elif METHOD == "BY":
                model = create_model_by()
            model.load_weights(LOAD_CHK_PATH)
        else:
            #新規作成
            if METHOD == "LSTM":
                model = create_model_lstm()
            elif METHOD == "NORMAL":
                model = create_model_normal()
            elif METHOD == "BY":
                model = create_model_by()
    else:
        # モデル作成
        with tf.distribute.MirroredStrategy().scope():
            # 複数GPU使用する
            # https://qiita.com/ytkj/items/18b2910c3363b938cde4
            if LOAD_TYPE == 1:
                model = tf.keras.models.load_model(MODEL_DIR_LOAD)
            elif LOAD_TYPE == 2:
                # 重さのみロード
                if METHOD == "LSTM":
                    model = create_model_lstm()
                elif METHOD == "NORMAL":
                    model = create_model_normal()
                elif METHOD == "BY":
                    model = create_model_by()

                model.load_weights(LOAD_CHK_PATH)
            else:
                # 新規作成
                if METHOD == "LSTM":
                    model = create_model_lstm()
                elif METHOD == "NORMAL":
                    model = create_model_normal()
                elif METHOD == "BY":
                    model = create_model_by()

    model.summary()

    return model

# callbacks.append(CSVLogger("history.csv"))
# look
# https://qiita.com/yukiB/items/f45f0f71bc9739830002

def do_train():
    # 処理時間計測
    t1 = time.time()
    model = get_model(False)

    # startからendへ戻ってrec_num分のデータを学習用とする
    rec_num = 90000000 + (INPUT_LEN[0]) + (PRED_TERM) + 1
    #rec_num = 100000 + (INPUT_LEN[0]) + (PRED_TERM) + 1

    start = datetime(2018, 1, 1)
    #start = datetime(2020, 1, 1)

    end = datetime(2009, 12, 18) #2018,1,1開始で90000000件の場合
    #end = datetime(2007, 1, 1) #2018,1,1開始で1300000000件の場合
    #end = datetime(2019, 7, 20) #2020,1,1開始で5000000件の場合
    #end = datetime(2017, 10, 6) #2020,1,1開始で25000000件の場合
    #end = datetime(2017, 4, 27) #2020,1,1開始で30000000件の場合
    #end = datetime(2011, 12, 22) #2020,1,1開始で90000000件の場合
    #end = datetime(2010, 3, 15)  #2020,1,1開始で110000000件の場合
    #end = datetime(2009, 4, 124)  #2020,1,1開始で120000000件の場合

    if REAL_SPREAD_FLG:
        #training時はREAL_SPREAD_FLGはFalseであるべき
        print("REAL_SPREAD_FLG is True! turn to False!!")
        exit(1)

    myLogger("rec_num:" + str(rec_num))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHK_DIR + "/{epoch:04d}",
        verbose=0,
        save_weights_only=True, )


    dataSequence2 = DataSequence2(rec_num, start, end, False)

    print("DataSequence Init End!!")

    # see: http://tech.wonderpla.net/entry/2017/10/24/110000
    # max_queue_size：データ生成処理を最大いくつキューイングしておくかという設定
    # use_multiprocessing:Trueならマルチプロセス、Falseならマルチスレッドで並列処理
    # workers:1より大きい数字を指定すると並列処理を実施

    hist = model.fit_generator(dataSequence2,
                                   steps_per_epoch=dataSequence2.__len__(),
                                   epochs = EPOCH,
                                   max_queue_size = PROCESS_COUNT *1 ,
                                   use_multiprocessing = True,
                                   workers= PROCESS_COUNT,
                                   verbose= 2,
                                   #verbose=1,
                                   callbacks=[tf.keras.callbacks.CSVLogger(
                                       filename = HISTORY_DIR + "/history.csv",
                                       append = False),
                                              cp_callback
                                              ],
                                   )

    # SavedModel形式で保存
    model.save(MODEL_DIR)

    # 全学習おわり
    print("total learning take:", time.time() - t1)

    # 学習結果（損失）のグラフを描画
    if hist is not None:
        # 損失の履歴をプロット
        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.show()


    print("END")



if __name__ == '__main__':
    print_load_info()

    if os.path.isdir(MODEL_DIR):
        #既にモデル保存ディレクトリがある場合はLEARNING_NUMが間違っているのでエラー
        print("ERROR!! MODEL_DIR Already Exists ")
        exit(1)

    makedirs(HISTORY_DIR)
    makedirs(CHK_DIR)

    do_train()
