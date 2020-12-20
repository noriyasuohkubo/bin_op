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


# モデル作成
def create_model():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))

    l2 = tf.keras.regularizers.l2(L2_RATE)  # 正則化： L2、 正則化率： 0.01

    if LEARNING_TYPE == "CATEGORY":
        init = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

        if len(LSTM_UNIT) > 1:
            lstms = []
            inputs = []
            for i, unit in enumerate(LSTM_UNIT):
                input = keras.Input(shape=(INPUT_LEN[i], 1))
                lstms.append(keras.layers.LSTM(LSTM_UNIT[i],  kernel_initializer = init,
                                              return_sequences=False)(input))
                inputs.append(input)

            concate = keras.layers.Concatenate()(lstms)

            dense = None
            for i, unit in enumerate(DENSE_UNIT):
                if i == 0:
                    dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                                  kernel_regularizer=l2,)(concate) # 正則化： L2、
                else:
                    dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu",  kernel_initializer = init,
                                  kernel_regularizer=l2,)(dense) # 正則化： L2、
            if dense != None:
                output = keras.layers.Dense(OUTPUT, activation='softmax', kernel_initializer = init)(dense)
            else:
                output = keras.layers.Dense(OUTPUT, activation='softmax', kernel_initializer = init)(concate)

            model = keras.Model(inputs=inputs, outputs=[output])

        else:
            #inputが1種類の場合

            input = keras.Input(shape=(INPUT_LEN[0], 1))
            lstm = keras.layers.LSTM(LSTM_UNIT[0], kernel_initializer = init,
                                           return_sequences=False)(input)

            dense = None
            for i, unit in enumerate(DENSE_UNIT):
                if i == 0:
                    dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                               kernel_regularizer=l2, )(lstm)  # 正則化： L2、
                else:
                    dense = keras.layers.Dense(DENSE_UNIT[i], activation="relu", kernel_initializer = init,
                                               kernel_regularizer=l2, )(dense)  # 正則化： L2、

            if dense != None:
                output = keras.layers.Dense(OUTPUT, activation='softmax', kernel_initializer = init)(dense)
            else:
                output = keras.layers.Dense(OUTPUT, activation='softmax', kernel_initializer = init)(lstm)

            model = keras.Model(inputs=[input], outputs=[output])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

    return model


def get_model(single_flg):
    model = None
    if single_flg:
        # 複数GPUを使用しない CPU用
        if LOAD_TYPE == 1:
            model = tf.keras.models.load_model(MODEL_DIR_LOAD)
        elif LOAD_TYPE == 2:
            #重さのみロード
            model = create_model()
            model.load_weights(LOAD_CHK_PATH)
        else:
            #新規作成
            model = create_model()
    else:
        # モデル作成
        with tf.distribute.MirroredStrategy().scope():
            # 複数GPU使用する
            # https://qiita.com/ytkj/items/18b2910c3363b938cde4
            if LOAD_TYPE == 1:
                model = tf.keras.models.load_model(MODEL_DIR_LOAD)
            elif LOAD_TYPE == 2:
                # 重さのみロード
                model = create_model()
                model.load_weights(LOAD_CHK_PATH)
            else:
                # 新規作成
                model = create_model()

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
    end = datetime(2000, 1, 1)

    myLogger("rec_num:" + str(rec_num))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHK_DIR + "/{epoch:04d}",
        verbose=0,
        save_weights_only=True, )


    dataSequence2 = DataSequence2(rec_num, start, end, False)

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

    if os.path.isdir(MODEL_DIR):
        #既にモデル保存ディレクトリがある場合はLEARNING_NUMが間違っているのでエラー
        print("ERROR!! MODEL_DIR Already Exists ")
        exit(1)

    makedirs(HISTORY_DIR)
    makedirs(CHK_DIR)

    do_train()
