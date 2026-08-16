import numpy as np
import tensorflow.keras.models
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
import logging.config
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from datetime import timedelta
import time
from indices import index
from decimal import Decimal
from flask import Flask, request
import subprocess
import send_mail as m
import datetime
from datetime import date
from tensorflow.keras import initializers
from util import *
import pandas as pd
import lightgbm as lgb
from app_usdjpy_fx_predict4_lgbm_907_conf import *

import concurrent.futures
from concurrent.futures import as_completed
from multiprocessing import Process
import multiprocessing

import ray

# Start Ray.
ray.init()

"""
nginxとflaskを使ってhttpによりAiの予想を呼び出す方式
systemctl start nginxでwebサーバを起動後、以下のコマンドによりuwsgiを起動し、localhost:80へアクセス
cat_binタイプのモデルを使用し、過去のモデルの予想結果とその過去からのレート変化を参考にベットするか決定する
"""
models = {}


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

@ray.remote
def do_wait(sleep_time, ):
    time.sleep(sleep_time)
    return sleep_time

@ray.remote
def do_predict_single(model, x,):
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K
    #global models

    start = time.time()
    print("start")
    model_tmp = load_model(model_dir_ht + model["name-h5"],custom_objects={"root_mean_squared_error": root_mean_squared_error, })
    print(model_tmp)
    #model_tmp = models[model["name"]]
    predict = model_tmp.predict_on_batch(x)

    print(time.time() - start)

    return model, predict

class AppUsdjpyFxPredict4Lgbm907Class():

    def __init__(self):
        global models

        # ubuntuではGPU使わない
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # lgbmモデル
        self.bst = lgb.Booster(model_file=model_dir_lgbm + lgbm_model_file)

        self.PAST_TERM = int(get_decimal_divide(PAST_TERM_SEC, LOOP_TERM))  # 必要とするlstmモデルの過去分予想のデータ間隔

        self.PAST_LENGTH_LIST = PAST_LENGTH *self. PAST_TERM

        self.prev_score = 0

        # モデルをロードしておく
        #self.models = {}

        for base_model in base_models:
            #model_tmp = load_model(model_dir_lstm + base_model["name"],
            #                       custom_objects={"root_mean_squared_error": self.root_mean_squared_error, })
            model_tmp = load_model(model_dir_ht + base_model["name-h5"],
                                   custom_objects={"root_mean_squared_error": self.root_mean_squared_error, })
            # 最初に一度推論させてグラフ作成し二回目以降の推論を早くする
            tmp_key_index = 1680775116  # 適当なunixタイム
            res = model_tmp.predict_on_batch(
                self.get_x(True, tmp_key_index, base_model["data_length"], base_model["input_datas"],
                           base_model["input_separate_flg"], base_model["method"], closes=None))

            print(base_model["no"], res)
            models[base_model["name"]] = model_tmp

        # lstm予想を、モデルNoをキー、予想をリストにして保持
        self.predict_dict = {}

        for model in base_models:
            if model["type"] == "CATEGORY":
                self.predict_dict[model["no"] + "-UP"] = []
                self.predict_dict[model["no"] + "-SAME"] = []
                self.predict_dict[model["no"] + "-DW"] = []
            elif model["type"] == "REGRESSION":
                self.predict_dict[model["no"] + "-REG"] = []

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def get_wma(self,value):
        # 加重移動平均
        weight = np.arange(len(value)) + 1
        wma = np.sum(weight * value) / weight.sum()

        return wma

    def get_max(self,value):
        return np.max(value)

    def get_min(self,value):
        return np.min(value)

    def get_x(self,init_flg, score, data_length, input_datas, input_separate_flg, method, closes=None):
        dt = datetime.datetime.fromtimestamp(score)
        now_sec = dt.second
        sec_oh_arr = [int(Decimal(str(now_sec)) / Decimal(str(AI_MODEL_TERM)))]  # 2秒間隔データなら０から29に変換しなければならないのでAI_MODEL_TERMで割る
        min_oh_arr = [dt.minute]
        hour_oh_arr = [dt.hour]

        SEC_OH_LEN = int(Decimal("60") / Decimal(str(AI_MODEL_TERM)))
        MIN_OH_LEN = 60
        HOUR_OH_LEN = 24

        retX = []
        for i, dl in enumerate(data_length):
            s, len = dl

            if input_datas[i] == "d1":
                X = np.zeros((1, len, 1))
                if init_flg:
                    X[:, :, 0] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM))) ))

                    aft = close_n[(-1 * len) -1:, -1]
                    bef = np.roll(aft, 1) #1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target
                retX.append(X)

            elif input_datas[i] == "wmd1-1":
                X = np.zeros((1, len, 1))
                if init_flg:
                    X[:, :, 0] = np.ones(len)
                else:
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM))) ))
                    aft = np.apply_along_axis(self.get_wma, 1, close_n)[(-1 * len) -1:]
                    bef = np.roll(aft, 1) #1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target
                retX.append(X)

            elif input_datas[i] == "ehd1-1_eld1-1":
                if input_separate_flg == True:
                    X1 = np.zeros((1, len, 1))
                    X2 = np.zeros((1, len, 1))
                    if init_flg:
                        X1[:, :, 0] = np.ones(len)
                        X2[:, :, 0] = np.ones(len)
                    else:
                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_max, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X1[:, :, 0] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_min, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X2[:, :, 0] = target

                    retX.append(X1)
                    retX.append(X2)
                else:
                    X = np.zeros((1, len, 2))
                    if init_flg:
                        X[:, :, 0] = np.ones(len)
                        X[:, :, 1] = np.ones(len)
                    else:
                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_max, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X[:, :, 0] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_min, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X[:, :, 1] = target

                    retX.append(X)

            elif input_datas[i] == "d1_ehd1-1_eld1-1":
                if input_separate_flg == True:
                    X1 = np.zeros((1, len, 1))
                    X2 = np.zeros((1, len, 1))
                    X3 = np.zeros((1, len, 1))
                    if init_flg:
                        X1[:, :, 0] = np.ones(len)
                        X2[:, :, 0] = np.ones(len)
                        X3[:, :, 0] = np.ones(len)
                    else:
                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))

                        aft = close_n[(-1 * len) - 1:, -1]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X1[:, :, 0] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_max, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X2[:, :, 0] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_min, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X3[:, :, 0] = target

                    retX.append(X1)
                    retX.append(X2)
                    retX.append(X3)
                else:
                    X = np.zeros((1, len, 3))

                    if init_flg:
                        X[:, :, 0] = np.ones(len)
                        X[:, :, 1] = np.ones(len)
                        X[:, :, 2] = np.ones(len)
                    else:
                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))

                        aft = close_n[(-1 * len) - 1:, -1]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X[:, :, 0] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_max, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X[:, :, 1] = target

                        close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(AI_MODEL_TERM)))))
                        aft = np.apply_along_axis(self.get_min, 1, close_n)[(-1 * len) - 1:]
                        bef = np.roll(aft, 1)  # 1つ前にずらす
                        target = get_divide_arr(bef, aft)[1:]
                        X[:, :, 2] = target

                    retX.append(X)

        if method == "LSTM7":
            retX.append(np.identity(SEC_OH_LEN)[sec_oh_arr])
            retX.append(np.identity(MIN_OH_LEN)[min_oh_arr])
            retX.append(np.identity(HOUR_OH_LEN)[hour_oh_arr])

        return retX

    def do_wait(self, ):
        time.sleep(1)

        return "name"

    def do_predict(self,score, closes, ):
        res = "0_100_0"
        # start = time.time()
        if get_decimal_sub(score, self.prev_score) > get_decimal_multi(LOOP_TERM, 2):
            # もし最後のリクエストからLOOP_TERM * 2以上経過していたら、予想が続かなくなるので、今までの予想結果を削除
            for model in base_models:
                if model["type"] == "CATEGORY":
                    self.predict_dict[model["no"] + "-UP"] = []
                    self.predict_dict[model["no"] + "-SAME"] = []
                    self.predict_dict[model["no"] + "-DW"] = []
                elif model["type"] == "REGRESSION":
                    self.predict_dict[model["no"] + "-REG"] = []
            print("delete predict_dict. lastscore:", self.prev_score)

        self.prev_score = score


        """
        # lstmのモデルによる予想
        for model in base_models:
            start = time.time()
            model_tmp = self.models[model["name"]]
            x = self.get_x(False, score, model["data_length"], model["input_datas"], model["input_separate_flg"],
                      model["method"], closes=closes)

            predict = model_tmp.predict_on_batch(x)

            if model["type"] == "CATEGORY":
                self.predict_dict[model["no"] + "-UP"].append(predict[0][0])
                self.predict_dict[model["no"] + "-SAME"].append(predict[0][1])
                self.predict_dict[model["no"] + "-DW"].append(predict[0][2])
            elif model["type"] == "REGRESSION":
                self.predict_dict[model["no"] + "-REG"].append(predict[0][0])
            end = time.time()
            print('シングルプロセス: TIME {:.4f}\n'.format(end - start))

        """
        # マルチプロセス ProcessPoolExecutor

        start = time.time()
        result_ids = []
        #for i in range(4):
        #    result_ids.append(do_wait.remote(i+1))
        for model in base_models:
            tmp_x = self.get_x(False, score, model["data_length"], model["input_datas"], model["input_separate_flg"],
                               model["method"], closes=closes)
            result_ids.append(do_predict_single.remote(model, tmp_x))

        results = ray.get(result_ids)
        print(results)
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, model in enumerate(base_models):
                futures.append(executor.submit(do_wait,i + 1,))
            for future in as_completed(futures):
                print(future.result())
                results.append(future.result())
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for model in base_models:
                tmp_x = self.get_x(False, score, model["data_length"], model["input_datas"], model["input_separate_flg"], model["method"], closes=closes)
                futures.append(executor.submit(do_predict_single, (model, tmp_x)))

            for future in as_completed(futures):
                print(future.result())
                results.append(future.result())
        
        
        for model, predict in results:
            if model["type"] == "CATEGORY":
                self.predict_dict[model["no"] + "-UP"].append(predict[0][0])
                self.predict_dict[model["no"] + "-SAME"].append(predict[0][1])
                self.predict_dict[model["no"] + "-DW"].append(predict[0][2])
            elif model["type"] == "REGRESSION":
                self.predict_dict[model["no"] + "-REG"].append(predict[0][0])
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as exe:
            future0 = exe.submit(do_wait,)
            future1 = exe.submit(do_wait,)
            future2 = exe.submit(do_wait,)
            future3 = exe.submit(do_wait,)
            print(future0.result(),future1.result(),future2.result(),future3.result())
        """
        end = time.time()
        print('CPUの数:', os.cpu_count())
        print('マルチプロセス: TIME {:.4f}\n'.format(end - start))


        # lgbmモデルの予想に必要なlstmモデルの過去分予想のリスト確認
        predict_len_ok_flg = True
        for no, predict_list in self.predict_dict.items():
            if len(predict_list) == self.PAST_LENGTH_LIST + 2:
                del predict_list[0]  # 1つ多いので最初を削除

            elif len(predict_list) > self.PAST_LENGTH_LIST + 2:
                print("error!!! predict_list length not correct:", len(predict_list))
                exit(1)

            elif len(predict_list) == self.PAST_LENGTH_LIST + 1:
                continue

            elif len(predict_list) < self.PAST_LENGTH_LIST + 1:
                # 長さが足りない場合
                predict_len_ok_flg = False
                break

        # lgbmモデルの予想に必要なlstmモデルの過去分予想が溜まっていたらlgbm予想実施
        if predict_len_ok_flg == True:
            # lgbmモデルに渡す特徴量(pandas)を作成する元の列
            base_col_dict = {}
            for no, predict_list in self.predict_dict.items():
                predict_list_rev = predict_list[::-1]  # リストを逆順にする
                for i, p in enumerate(predict_list_rev):
                    if i % self.PAST_TERM == 0:
                        if i == 0:
                            base_col_dict[no] = [p]
                        else:
                            base_col_dict[no + "-" + str(int(get_decimal_multi(i, LOOP_TERM)))] = [p]
            # print(base_col_dict)
            if len(lgbm_ds) != 0:
                # d1をlgbmモデルの特徴量とする場合
                for i, ds in enumerate(lgbm_ds):
                    data_length = ds["data_length"]
                    data_idx = ds["data_idx"]

                    close_n = np.reshape(closes, (-1, int(Decimal(str(data_length)) / Decimal(str(AI_MODEL_TERM)))))

                    for idx in data_idx:
                        aft = close_n[-1]
                        bef = close_n[-1 - idx]
                        base_col_dict[str(data_length) + "-d-" + str(idx)] = get_divide_arr(bef, aft)

            tmp_dt = datetime.datetime.fromtimestamp(score)
            if "hour" in INPUT_DATA:
                base_col_dict["hour"] = tmp_dt.hour
            if "min" in INPUT_DATA:
                base_col_dict["min"] = tmp_dt.minute
            if "sec" in INPUT_DATA:
                base_col_dict["sec"] = tmp_dt.second
            if "week" in INPUT_DATA:
                base_col_dict["week"] = tmp_dt.weekday()

            if "weeknum" in INPUT_DATA:
                base_col_dict["weeknum"] = get_weeknum(tmp_dt.weekday(), tmp_dt.day)

            x_df = pd.DataFrame(base_col_dict, index=pd.Index([score]))
            csv_regist_cols = x_df.columns.tolist()
            csv_regist_cols.sort()  # カラムを名前順にするx_df_dixt = x_df.to_dict(orient='index')

            tmp_dict = {}
            for col in csv_regist_cols:
                if col == "hour" or col == "min" or col == "sec" or col == "week" or col == "weeknum":
                    tmp_dict[col] = 'int8'
                else:
                    tmp_dict[col] = 'float32'

            # 型変換
            x_df = x_df.astype(tmp_dict)

            # 訓練時と同じ列名の順番で特徴量を取得する
            # pandasデータをxとしてモデルに渡すが、ヘッダーは見ていないため、訓練時と同じ特徴量の順番にする必要がある(INPUT_DATAを指定することによりそのとおりの順番で抽出してくれる)
            x_df = x_df.loc[[score], INPUT_DATA]
            # print(x_df.info())

            predict_lgbm = self.bst.predict(x_df, num_iteration=int(lgbm_model_file_suffix))
            predict_lgbm = predict_lgbm[0]
            #print(predict_lgbm)
            probe_up = predict_lgbm[0]
            probe_same = predict_lgbm[1]
            probe_dw = predict_lgbm[2]

            res = str(probe_up) + "_" + str(probe_same) + "_" + str(probe_dw)

        return res


    def get_predict(self, score, vals):
        start_predict = time.perf_counter()

        closes = vals[:]

        if len(vals) != MAX_CLOSE_LEN:
            print("error!!! data length not correct:", len(vals), MAX_CLOSE_LEN)
            exit(1)

        res = self.do_predict(score, closes)
        print("score", score, datetime.datetime.fromtimestamp(score), "predict_take",time.perf_counter() - start_predict, "res", res)

        return res



