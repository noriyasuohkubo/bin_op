import numpy as np
import psutil
from tensorflow import keras
from tensorflow.keras import layers
import configparser
import os
import redis
import traceback
import json
import logging.config
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd
import conf_class
from indices import index
from decimal import Decimal
from DataSequence2 import DataSequence2
import math
import sys
from silence_tensorflow import silence_tensorflow
silence_tensorflow() #ログ抑制 import tensorflowの前におく
import tensorflow as tf
import gc
from util import *
import socket
import send_mail as mail
from tensorflow_addons.optimizers import AdamW,RectifiedAdam,LazyAdam
from DataSequence2_from_pickle_test_on_memory import DataSequence2_from_pickle_test_on_memory

"""
あるLSTMモデルによって算出された予測値を予想時間(score)と共にDBまたはCSVに保存する
"""

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def fx_mean_squared_error(y_true, y_pred):

    #予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(error**2 + conf_class.FX_LOSS_PNALTY * error**2 * not_trend_match)

    return loss

def fx_mean_squared_error2(y_true, y_pred):

    #予想値のトレンドが異なる場合は罰則を強化する

    error = y_true - y_pred
    not_trend_match = tf.cast(tf.math.sign(y_true) != tf.math.sign(y_pred), tf.float32)
    loss = tf.math.reduce_mean(error**2 - conf_class.FX_LOSS_PNALTY * error**2 + conf_class.FX_LOSS_PNALTY * 2 * error**2 * not_trend_match)
    return loss

def mean_squared_error_custome(y_true, y_pred):
    #誤差の３乗を罰則とする
    error = abs(y_true - y_pred)
    loss = tf.math.reduce_mean(error ** conf_class.MSE_PENALTY)

    return loss

def fx_insensitive_error(y_true, y_pred):

    #ε-感度損失:細かい誤差は気にしない
    #閾値以上の誤差がある場合だけ罰則
    error = abs(y_true - y_pred)
    not_trend_match = tf.cast(error >= conf_class.INSENSITIVE_BORDER, tf.float32)

    loss = tf.math.reduce_mean(error**2 * not_trend_match)

    return loss

def negative_log_likelihood(y_true, y_pred):

    return -1 * y_pred.log_prob(y_true)

class MakePredictDb():
    def __init__(self, symbol):
        self.symbol = symbol

    def make_predict_db(self, conf, dataSequence2, start, end, db_no_new, model_param, past_terms, file_postscript):
        #conf = conf_class.ConfClass()
        past_terms_str = "_PAST-TERMS-" + list_to_str(past_terms)
        #print(past_terms_str)

        # 処理時間計測
        t1 = time.time()

        learning_type = model_param['learning_type']
        input_len = model_param['input_len']
        input_datas = model_param['input_datas']
        db_terms = model_param['db_terms']
        DB1_NOT_LEARN = model_param['DB1_NOT_LEARN']

        conf.DB1_NOT_LEARN = DB1_NOT_LEARN

        #termのチェック
        if conf.DB1_TERM != db_terms[0]:
            print(conf.DB1_TERM, db_terms[0])
            exit(1)
        if conf.DB2_TERM != db_terms[1]:
            print(conf.DB2_TERM, db_terms[1])
            exit(1)
        if conf.DB3_TERM != db_terms[2]:
            print(conf.DB3_TERM, db_terms[2])
            exit(1)
        if conf.DB4_TERM != db_terms[3]:
            print(conf.DB4_TERM, db_terms[3])
            exit(1)
        if conf.DB5_TERM != db_terms[4]:
            print(conf.DB5_TERM, db_terms[4])
            exit(1)

        #INPUT_LENチェック
        for i,j in zip(conf.INPUT_LEN, input_len):
            if i != j:
                print("conf.INPUT_LEN not input_len", conf.INPUT_LEN, input_len)
                exit(1)

        #INPUT_DATASチェック
        for i,j in zip(conf.INPUT_DATAS, input_datas):
            if i != j:
                print("conf.INPUT_DATAS not input_len", conf.INPUT_DATAS, input_datas)
                exit(1)

        if learning_type == "REGRESSION":
            type="REG" #REG or BOTH or BIN_UP or BIN_DW

        elif learning_type == "CATEGORY":
            type="BOTH" #REG or BOTH or BIN_UP or BIN_DW


        both_type = "RAW" #MAX:up,same,dwの中から最大を選ぶ, RAW:up,same,dwそれぞれの確率を記録する

        #DB名はモデル名とする
        #model_name = "MN887-39"
        #model_name = "USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.25_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub1_MN715-40"
        model_name = model_param['model_name']

        model_no = model_name.split("MN")[1]

        col_prefix = model_no

        #DB用パラメータ
        host = "127.0.0.1"
        #host = "ub1"
        #db_no_new = 3
        db_name_new = self.symbol + "_PREDICT_"
        db_name_new = db_name_new + model_no

        load_dir = "/app/model/bin_op/" + model_name
        model = tf.keras.models.load_model(load_dir, custom_objects={"root_mean_squared_error": root_mean_squared_error,
                                                                     "fx_mean_squared_error": fx_mean_squared_error,
                                                                     "fx_mean_squared_error2": fx_mean_squared_error2,
                                                                     "mean_squared_error_custome": mean_squared_error_custome,
                                                                     "fx_insensitive_error": fx_insensitive_error,
                                                                     "negative_log_likelihood": negative_log_likelihood,
                                                                     })
        #予想実施
        #conf.change_real_spread_flg(True)
        #conf.FX_TICK_DB = ""

        #conf.DB_EVAL_NO = db_eval_no #予想対象データが入っているDB

        #dataSequence2 = DataSequence2(conf, start, end, True, False,)
        #print("dataSequence2 make finished")

        # 全close値のリスト
        close_list = dataSequence2.get_close_list()
        # 全score値のリスト
        score_list = dataSequence2.get_score_list()

        # 予想時のレート(FX用)
        pred_close_list = np.array(dataSequence2.get_pred_close_list())
        # 予想対象のscore値のリスト
        target_score_list = np.array(dataSequence2.get_train_score_list())

        print("predict start")
        predict_list = model.predict(dataSequence2,
                                               steps=None,
                                               max_queue_size=conf.MAX_QUEUE_SIZE * 1,
                                               use_multiprocessing=True,
                                               verbose=0)
        K.clear_session()
        print("predict finished")

        print("predict length:", len(predict_list))
        print("close_list length:", len(close_list))
        print("score_list length:", len(score_list))

        if len(predict_list) != len(target_score_list):
            print("length of predict_list and length of target_score_list are not same:", len(predict_list),len(target_score_list))
            exit(1)
        if len(predict_list) != len(pred_close_list):
            print("length of predict_list and length of pred_close_list are not same:", len(predict_list),len(pred_close_list))
            exit(1)

        # 予想結果と予想時のスコアを辞書で保持
        score_pred_dict = dict(zip(target_score_list, predict_list))
        # 予想時のレートと予想時のスコアを辞書で保持
        score_close_dict = dict(zip(target_score_list, pred_close_list))

        print("dict make finished")

        # CSVに書き込む為に保持するデータ
        csv_dict = {}

        redis_db = redis.Redis(host=host, port=6379, db=db_no_new, decode_responses=True)
        cnt = 0

        for i, (s, close, ) in enumerate(zip(score_list, close_list,  )):
            #print("s",s)
            try:
                pred = score_pred_dict[s]  # scoreをもとに予想を取得
            except Exception:
                #予想がない
                continue

            child = {}
            if type == "REG":
                tmp_pred = float(pred[0])

                child[col_prefix + "-REG"] = tmp_pred

            elif type == "BOTH":
                if both_type == "MAX":
                    child[col_prefix] = np.argmax(pred)
                else:
                    child[col_prefix + "-UP"] = float(pred[0])  # upの確率を格納
                    child[col_prefix + "-SAME"] = float(pred[1])  # sameの確率を格納
                    child[col_prefix + "-DW"] = float(pred[2])  # dwの確率を格納

            elif type == "BIN_UP":
                child[col_prefix + "-UP"] = float(pred[0])  # upの確率を格納
                child[col_prefix + "-DW"] = float(pred[1])  # same,dwの確率を格納

            elif type == "BIN_DW":
                child[col_prefix + "-DW"] = float(pred[0])  # dwの確率を格納
                child[col_prefix + "-UP"] = float(pred[1])  # same,upの確率を格納


            break_flg = False
            for past_term in past_terms:
                try:
                    pred_past = score_pred_dict[get_decimal_sub(s, past_term)]  # scoreをもとに過去予想を取得
                    past_close = score_close_dict[get_decimal_sub(s, past_term)]  # scoreをもとに過去予想時レートを取得
                except Exception:
                    # 予想がないのでスキップ
                    break_flg =True
                    break
                if type == "REG":
                    child[col_prefix + "-REG" + "-" + str(past_term)] = float(pred_past[0])

                elif type == "BOTH":
                    if both_type == "MAX":
                        child[col_prefix + "-" + str(past_term)] = np.argmax(pred_past)
                    else:
                        child[col_prefix + "-UP" + "-" + str(past_term)] = float(pred_past[0])  # upの確率を格納
                        child[col_prefix + "-SAME" + "-" + str(past_term)] = float(pred_past[1])  # sameの確率を格納
                        child[col_prefix + "-DW" + "-" + str(past_term)] = float(pred_past[2])  # dwの確率を格納

                elif type == "BIN_UP":
                    child[col_prefix + "-UP" + "-" + str(past_term)] = float(pred_past[0])  # upの確率を格納
                    child[col_prefix + "-DW" + "-" + str(past_term)] = float(pred_past[1])  # same,dwの確率を格納

                elif type == "BIN_DW":
                    child[col_prefix + "-DW" + "-" + str(past_term)] = float(pred_past[0])  # dwの確率を格納
                    child[col_prefix + "-UP" + "-" + str(past_term)] = float(pred_past[1])  # same,upの確率を格納

                rate_change = float(Decimal(str(close)) - Decimal(str(past_close)))

            if break_flg:
                continue

            child["score"] = float(s)
            #child["o"] = close
            #print("child", child)

            if sava_type == "db":

                old_data = redis_db.zrangebyscore(db_name_new, s, s, withscores=True)
                #すでにデータある場合,削除あとに追加
                if len(old_data) != 0:
                    rm_cnt = redis_db.zremrangebyscore(db_name_new,s,s) #削除した件数取得
                    if rm_cnt != 1:
                        #削除できなかったらおかしいのでエラーとする
                        print("cannot remove!!!", s)
                        exit()

                redis_db.zadd(db_name_new, json.dumps(child), s)

            elif sava_type == "csv":
                for tmp_col in child.keys():
                    if tmp_col in csv_dict.keys():
                        csv_dict[tmp_col].append(child[tmp_col])
                    else:
                        csv_dict[tmp_col] = [child[tmp_col]]

            cnt += 1

            if cnt % 1000000 == 0:
                dt_now = datetime.now()
                print(dt_now, " ", cnt/len(predict_list), "% 終了" )

        #csv保存する
        if sava_type == "csv":
            # CSVに書き込む際に左側にマージするスコアとオープンレートを保持するpandasデータ作成用辞書
            # lgbm_make_data.pyではスコアデータが続いていないと除外されるのでスコアをマージする
            parent_dict = {}
            parent_dict['score'] = score_list
            parent_dict['o'] = close_list
            df_parent = pd.DataFrame(parent_dict)
            tmp_dict = {'score':'float64', 'o':'float32'}
            df_parent = df_parent.astype(tmp_dict)


            df_dict = pd.DataFrame(csv_dict)
            tmp_dict = {'score':'float64',}
            df_dict = df_dict.astype(tmp_dict)

            pd.options.display.float_format = '{:.6f}'.format
            print(df_dict[:100])
            print(df_dict[-100:])

            #マージ実施
            df_parent = pd.merge(df_parent, df_dict, on="score", how='left', copy=False)

            del csv_dict, parent_dict
            gc.collect()

            csv_regist_cols = df_parent.columns.tolist()
            csv_regist_cols.sort()  # カラムを名前順にする

            df_parent.sort_values('score', ascending=True,inplace=True)  # scoreの昇順　古い順にする
            print(df_dict[:100])
            print(df_parent[-100:])
            end_tmp = end + timedelta(days=-1)

            input_name = list_to_str(csv_regist_cols, ",")
            tmp_file_name = self.symbol + "_IN-" + input_name + "_" + date_to_str(start,format='%Y%m%d') + "-" + date_to_str(end_tmp, format='%Y%m%d') + "_" + conf.DTYPE + past_terms_str + "_" + socket.gethostname() + file_postscript
            db_name_file = "PREDICT_FILE_NO_" + self.symbol

            # win2のDBを参照してモデルのナンバリングを行う
            r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
            result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
            if len(result) == 0:
                print("CANNOT GET PREDICT_FILE_NO")
                exit(1)
            else:
                newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

                """
                for line in result:
                    body = line[0]
                    score = int(line[1])
                    tmps = json.loads(body)
                    tmp_name = tmps.get("input_name")
                    if tmp_name == tmp_file_name:
                        # 同じファイルがないが確認
                        print("The File Already Exists!!!")
                    exit(1)
                """
                # DBにモデルを登録
                child = {
                    'input_name': tmp_file_name,
                    'no': newest_no
                }
                r.zadd(db_name_file, json.dumps(child), newest_no)

            csv_path = "/db2/lgbm/" + self.symbol + "/predict_file/" + "PF" + str(newest_no)
            #csv_file_name = csv_path + ".csv"
            pickle_file_name = csv_path + ".pickle"

            print("predict_file", "PF" + str(newest_no))
            print("input_name", tmp_file_name)

            tmp_dict = {}
            for col in csv_regist_cols:
                if col != 'score' and ("RC" in col) == False:
                    if type == "BOTH" and both_type == "MAX":
                        tmp_dict[col] = 'int8'
                    else:
                        tmp_dict[col] = conf.DTYPE

            df_parent = df_parent.astype(tmp_dict)

            # pickleで保存
            df_parent.to_pickle(pickle_file_name)

        del dataSequence2, df_parent, score_list, close_list, pred_close_list, target_score_list, predict_list, score_pred_dict, score_close_dict

        print("END!!!")

        t2 = time.time()
        elapsed_time = t2-t1
        print("経過時間：" + str(elapsed_time))

if __name__ == "__main__":
    """
    model_params = [
        #USDJPY_FLOAT32_GPU2_AMP_DLTrue_LT1_M1_LSTM1_B1_BS1_T30.0_I3_IL200_LU30_DU10-10-10-10-10_DROP0.1_BNL2_BDIV1.0_LRT0.0001_LT1_Adam_DA4_RA8_RRA9_d1_1_OT-d_OD-c_BS30720_SD0_SHU1_EL20-21-22_EDS300-MIN10_DSFPC-USDJPY-475_20160101-20241201_RETY:CATEGORY-d-1.0:CATEGORY-d-0.3:CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-_ub2_DUKAS_MN1750
         {
            'learning_type': 'CATEGORY',
            'model_name': 'MN1750-883',
            'db_terms':[3,0,0,0,0],
            'input_len':[200,],
            'input_datas':["d1",],
            'DB1_NOT_LEARN':False,
         },

        # USDJPY_FLOAT32_GPU2_AMP_DLTrue_LT1_M1_LSTM1_B1_BS1_T30.0_I3_IL200_LU30_DU10-10-10-10-10_DROP0.1_BNL2_BDIV1.0_LRT0.0001_LT1_Adam_DA4_RA8_RRA9_d1_1_OT-d_OD-c_BS30720_SD0_SHU1_EL20-21-22_EDS300-MIN10_DSFPC-USDJPY-510_20040101-20241201_RETY:CATEGORY-d-1.0:CATEGORY-d-0.3:CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-_ub2_DUKAS_MN1835
        {
            'learning_type': 'CATEGORY',
            'model_name': 'MN1835-67',
            'db_terms': [3, 0, 0, 0, 0],
            'input_len': [200, ],
            'input_datas': ["d1", ],
            'DB1_NOT_LEARN': False,
        },


    ]
    """
    model_params = [
        #USDJPY_FLOAT32_GPU2_AMP_DLTrue_LT1_M1_LSTM1_B1_BS1_T30.0_I3_IL300_LU30_DU_BNL2_BDIV0.5_LRT0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_OT-d_OD-c_BS30720_SD0_SHU1_EL20-21-22_EDS300-MIN5_DSFPC-USDJPY-441_20160101-20241201_RETY:CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-_ub2_DUKAS_MN1500
         {
            'learning_type': 'CATEGORY',
            'model_name': 'MN1500-196',
            'db_terms':[3,0,0,0,0],
            'input_len':[300,],
            'input_datas':["d1",],
            'DB1_NOT_LEARN':False,
         },
        #USDJPY_FLOAT32_GPU2_AMP_DLTrue_LT1_M1_LSTM1_B1_BS1_T30.0_I3_IL300_LU30_DU30-30-30_DROP0.1_BNL2_BDIV0.5_LRT0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_OT-d_OD-c_BS30720_SD0_SHU1_EL20-21-22_EDS300-MIN5_DSFPC-USDJPY-441_20160101-20241201_RETY:CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-_ub2_DUKAS_MN1504
         {
            'learning_type': 'CATEGORY',
            'model_name': 'MN1504-196',
            'db_terms':[3,0,0,0,0],
            'input_len':[300,],
            'input_datas':["d1",],
            'DB1_NOT_LEARN':False,
         },
        #USDJPY_FLOAT32_GPU2_AMP_DLTrue_LT1_M1_LSTM1_B1_BS1_T30.0_I3_IL300_LU30_DU50-50-50-50-50_DROP0.1_BNL2_BDIV0.5_LRT0.0005_LT1_Adam_DA4_RA8_RRA9_d1_1_OT-d_OD-c_BS30720_SD0_SHU1_EL20-21-22_EDS300-MIN5_DSFPC-USDJPY-441_20160101-20241201_RETY:CATEGORY-d-0.5:CATEGORY-d-0.3:CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-_ub3_DUKAS_MN1633
        {
            'learning_type': 'CATEGORY',
            'model_name': 'MN1633-79',
            'db_terms': [3, 0, 0, 0, 0],
            'input_len': [300, ],
            'input_datas': ["d1", ],
            'DB1_NOT_LEARN': False,
        },
    ]

    dates = [
        [datetime(2024, 12, 1, ), datetime(2026, 6, 27, ), ],


    ]

    file_postscript = ""

    # 過去の予想と、現時点での過去予想のレートの変化を求めるための過去秒
    past_terms = [i for i in range(1, 3, 1)]
    #past_terms = [i for i in range(4, 13, 4)]
    #past_terms = [i for i in range(30, 31, 30)]
    #past_terms = []

    symbol = "USDJPY"

    sava_type = "csv"  # 保存形式:csv or db

    db_eval_no = 1
    db_no_new = 1 #sava_typeがdbの場合に登録するDB_NO

    redis_stop = False

    try:
        mpd = MakePredictDb(symbol)
        conf = conf_class.ConfClass()
        conf.change_real_spread_flg(True)
        conf.FX_TICK_DB = ""

        conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY = {
            "score": "640",
            "save_dir_path": "/nvme2/dataSequence2/USDJPY/DS2F640-0",
        }

        conf.DB_EVAL_NO = db_eval_no #予想対象データが入っているDB

        #特徴量が欲しいだけなので,Yに関係ないLOSS_TYPEとLEARNING_TYPEはCATEGORYにしておく
        conf.LOSS_TYPE = "C-ENTROPY"
        conf.LEARNING_TYPE = "CATEGORY"

        for dt in dates:
            start, end = dt

            print(datetime.now(), "dataSequence2 make start")
            if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY) != 0:
                dataSequence2 = DataSequence2_from_pickle_test_on_memory(conf, )
                dataSequence2.load_ds2()
                print("DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY:", conf.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_ON_MEMORY["score"])
            else:
                dataSequence2 = DataSequence2(conf, start, end, test_flg=True, eval_flg=False, return_all=False, redis_stop=redis_stop)

            print(datetime.now(), "dataSequence2 make finished")

            for model_param in model_params:
                print("model:", model_param['model_name'])

                mpd.make_predict_db(conf, dataSequence2, start, end , db_no_new,model_param, past_terms, file_postscript)
                print(datetime.now(), "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
                gc.collect()
                print(datetime.now(), "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    except Exception as e:
        print(tracebackPrint(e))
        mail.send_message(socket.gethostname(), ": make_predict_db.py error occured")

    mail.send_message(socket.gethostname(), ": make_predict_db.py finished!!!")
    print("FINISH")