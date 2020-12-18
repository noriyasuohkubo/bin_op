import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import redis
import traceback
import json
import keras.optimizers as optimizers
import time
import pandas as pd
from sklearn import preprocessing
from keras.models import load_model
from keras import backend as K
import os
from keras.callbacks import CSVLogger
import configparser
from keras.callbacks import ModelCheckpoint
from scipy.ndimage.interpolation import shift
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from logging import getLogger
from datetime import datetime
from datetime import timedelta
import time
from decimal import Decimal
from DataSequence import DataSequence
import multiprocessing
from keras.backend import tensorflow_backend
from readConf import *
from multiGPUCheckPointCallback import MultiGPUCheckpointCallback
from keras.layers import BatchNormalization
from keras.layers import Input, Concatenate
from keras.models import Model

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

"""
for k, v in os.environ.items():
    print(k + ":"+ v)
"""
#cuda default usable memory 64GB
os.environ["TF_CUDA_HOST_MEM_LIMIT_IN_MB"]="100000"
# tensorflow versin 1.14 ~ TF_GPU_HOST_MEM_LIMIT_IN_MB
# https://blog.exxactcorp.com/new-tensorflow-release-v1-14-0/

#設定ファイル
#ini_file = os.path.join(current_dir, "config", "config.ini")

train = True

np.random.seed(0)

#以下の件数を多く設定する時はデータ読み込み後(DataSequence:get redis data が標準出力された後)はsystemctl stop redisしてメモリを開ける
#GPU温度を監視するため学習時は/app/monitor/gpu_tmp_monitor.pyを実行しておくこと！

#startからendへ戻ってrec_num分のデータを学習用とする
rec_num = 90000000 + (maxlen * close_shift) + (pred_term * close_shift) + 1

learning_rate = 0.001

start = datetime(2018, 1, 1)
end = datetime(2000, 1, 1)

epochs = 5

myLogger("rec_num:" + str(rec_num))

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def mean_pred(y_true, y_pred):
    tmp = y_true * y_pred
    o = tf.constant(1, dtype=tf.float32)
    z = tf.constant(0, dtype=tf.float32)

    return K.mean(tf.map_fn(lambda x: tf.cond(tf.greater_equal(x[0], z), lambda: o, lambda: z), tmp))

def create_model(n_out=3):
    model = None

    with tf.device("/cpu:0"):#モデルの構築はOOMエラー対策のため、CPUで明示的に行う必要がある

        if functional_flg:
            sec_input = Input(shape=(maxlen, 1))
            min_input = Input(shape=(maxlen_min, 1))
            sec_x = LSTM(n_hidden[1]
                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                        ,return_sequences=False)(sec_input)
            min_x = LSTM(min_hidden
                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                        ,return_sequences=False)(min_input)
            x = Concatenate()([sec_x, min_x])
            main_output = Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None) ,activation='softmax')(x)
            model = Model(inputs=[sec_input, min_input], outputs=[main_output])
            return model

        if type == "category":
            model = Sequential()
            """
            model.add(LSTM(n_hidden[1],input_shape=(maxlen, n_in)
                           , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                           ,return_sequences = False))
            """
            if method == "by":
                if n_hidden.get(2) is None or n_hidden.get(2) == 0:
                    model_tmp = Bidirectional(LSTM(n_hidden[1]
                                                 , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                             ,return_sequences=False)
                                            , input_shape=(maxlen, x_length)
                                            )
                else:
                    model_tmp = Bidirectional(LSTM(n_hidden[1]
                                                 , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                             ,return_sequences=True)
                                            , input_shape=(maxlen, x_length)
                                            )
            elif method == "lstm":
                if n_hidden.get(2) is None or n_hidden.get(2) == 0:
                    model_tmp = LSTM(n_hidden[1]
                                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                        ,return_sequences=False, input_shape=(maxlen, x_length))


                else:
                    model_tmp = LSTM(n_hidden[1]
                                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                        ,return_sequences=True, input_shape=(maxlen, x_length))
            elif method == "gru":
                if n_hidden.get(2) is None or n_hidden.get(2) == 0:
                    model_tmp = GRU(n_hidden[1]
                                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                        ,return_sequences=False, input_shape=(maxlen, x_length))


                else:
                    model_tmp = GRU(n_hidden[1]
                                        , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                        ,return_sequences=True, input_shape=(maxlen, x_length))

            model.add(model_tmp)
            model.add(Dropout(drop))

            for k, v in sorted(n_hidden.items()):
                if k == 1:
                    continue
                if n_hidden[k] != 0:
                    if n_hidden.get(k + 1) is None or n_hidden.get(k+1) == 0:
                        if method == "by":
                            model.add(Bidirectional(LSTM(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=False)
                                                    ))
                        elif method == "lstm":
                            model.add(LSTM(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=False)
                                                    )
                        elif method == "gru":
                            model.add(GRU(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=False)
                                                    )
                    else:
                        if method == "by":
                            model.add(Bidirectional(LSTM(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=True)
                                                    ))
                        elif method == "lstm":
                            model.add(LSTM(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=True)
                                                    )
                        elif method == "gru":
                            model.add(GRU(n_hidden[k]
                                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1,
                                                                                                        seed=None)
                                                         , return_sequences=True)
                                                    )
                    model.add(Dropout(drop))
            #HIDDEN add
            for k, v in sorted(dense_hidden.items()):
                if dense_hidden[k] != 0:
                    model.add(Dense(dense_hidden[k], kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None), use_bias=False))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))

            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
            model.add(Activation('softmax'))

        elif type == 'mean':
            model = Sequential()
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           input_shape=(maxlen, len(in_features))
                           , return_sequences=True))
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
                           ))
            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
            model.add(Activation('linear'))

    return model


def get_model():
    model = None
    #print("model:" ,model_file)
    if os.path.isfile(model_file):
        model = load_model(model_file, custom_objects={"mean_pred": mean_pred})
        print("Load Model")
    else:
        print("Create Model")
        model = create_model()
    #model_gpu = model
    model_gpu = multi_gpu_model(model, gpus=gpu_count)
    if type == "category":
        #学習率
        adm = optimizers.Adam(lr=learning_rate)
        model_gpu.compile(loss='categorical_crossentropy',
                          optimizer=adm, metrics=['accuracy'])
    elif type == 'mean':
        model_gpu.compile(loss='mean_squared_error',
                          optimizer="rmsprop", metrics=[mean_pred])

    print(model.summary())

    return model, model_gpu


'''

モデル学習
'''


# callbacks.append(CSVLogger("history.csv"))
# look
# https://qiita.com/yukiB/items/f45f0f71bc9739830002

def do_train():
    # 処理時間計測
    t1 = time.time()
    model, model_gpu = get_model()
    early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)
    #checkpoint = ModelCheckpoint(filepath= model_file + '_{epoch:02d}_.hdf5', monitor='val_loss', verbose=1, save_best_only=False)

    # see: https://qiita.com/surumetic/items/6a967649b07163da054f
    checkpoint = MultiGPUCheckpointCallback(filepath= model_file + '_{epoch:02d}_.hdf5', base_model = model, monitor='val_loss', verbose=1, save_best_only=False)
    dataSequence = DataSequence(rec_num, start, end, False)

    # see: http://tech.wonderpla.net/entry/2017/10/24/110000
    # max_queue_size：データ生成処理を最大いくつキューイングしておくかという設定
    # use_multiprocessing:Trueならマルチプロセス、Falseならマルチスレッドで並列処理
    # workers:1より大きい数字を指定すると並列処理を実施
    hist = model_gpu.fit_generator(dataSequence,
                                    steps_per_epoch=dataSequence.__len__(),
                                    epochs=epochs,
                                    max_queue_size=process_count * 1,
                                    callbacks=[CSVLogger(history_file), checkpoint],
                                    use_multiprocessing=True,
                                    workers=process_count,
                                   )
    # save model, not model_gpu
    # see http://tech.wonderpla.net/entry/2018/01/09/110000
    model.save(model_file)
    print('Model saved')

    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：" + str(elapsed_time))

    if hist is not None:
        # 損失の履歴をプロット
        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.show()

    K.clear_session()

    print("END")



if __name__ == '__main__':
    do_train()
