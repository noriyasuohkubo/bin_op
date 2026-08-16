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
from app_usdjpy_fx_predict30_lgbm_2009_conf import *
from util_predict import *

"""
nginxとflaskを使ってhttpによりAiの予想を呼び出す方式
systemctl start nginxでwebサーバを起動後、以下のコマンドによりuwsgiを起動し、localhost:80へアクセス
cat_binタイプのモデルを使用し、過去のモデルの予想結果とその過去からのレート変化を参考にベットするか決定する
"""

class AppUsdjpyFxPredict30Lgbm2009Class():

    def __init__(self, mode="honban"):
        self.mode = mode #テスト時はtestにする

        # ubuntuではGPU使わない
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # lgbmモデル
        self.bst = lgb.Booster(model_file=model_dir_lgbm + lgbm_model_file)

        self.PAST_TERM = int(get_decimal_divide(PAST_TERM_SEC, LOOP_TERM))  # 必要とするlstmモデルの過去分予想のデータ間隔

        self.PAST_LENGTH_LIST = PAST_LENGTH *self. PAST_TERM

        self.prev_score = 0

        # モデルをロードしておく
        self.models = {}

        #lstm予想結果にアクセスするredisコネクションを保存しておく
        self.model_dbs = {}
        for base_model in base_models:

            if mode == "test":
                print("test mode. loading modeals!!!")
                model_tmp = load_model(model_dir_lstm + base_model["name"],
                                       custom_objects={"root_mean_squared_error": root_mean_squared_error, })
                # 最初に一度推論させてグラフ作成し二回目以降の推論を早くする
                tmp_key_index = 1680775116  # 適当なunixタイム
                res = model_tmp.predict_on_batch(
                    get_x(AI_MODEL_TERM,True, tmp_key_index, base_model["data_length"], base_model["input_datas"],
                               base_model["input_separate_flg"], base_model["method"], closes=None))


                print(base_model["no"], res)
                self.models[base_model["no"]] = model_tmp

            self.model_dbs[base_model["no"]] = redis.Redis(host=base_model["db_host"], port=6379, db=base_model["db_no"], decode_responses=True, socket_keepalive=True)

        # lstm予想を、モデルNoをキー、予想をリストにして保持
        self.predict_dict = {}

        for model in base_models:
            if model["type"] == "CATEGORY":
                self.predict_dict[model["no"] + "-UP"] = []
                self.predict_dict[model["no"] + "-SAME"] = []
                self.predict_dict[model["no"] + "-DW"] = []
            elif model["type"] == "REGRESSION":
                self.predict_dict[model["no"] + "-REG"] = []

    #lstmモデルの予想結果を取得する
    def get_predict_lstm_self(self, closes, model, score):
        predict_class = self.models[model["no"]]

        x = get_x(AI_MODEL_TERM, False, score, model["data_length"], model["input_datas"],
                               model["input_separate_flg"], model["method"], closes=closes)

        predict = predict_class.predict_on_batch(x)

        if model["type"] == "CATEGORY":
            response_text = str(predict[0][0]) + "_" + str(predict[0][1]) + "_" + str(predict[0][2])
        elif model["type"] == "REGRESSION":
            response_text = str(predict[0][0])

        return response_text

    #lstmモデルの予想結果を取得する
    def get_predict_lstm(self, score, model):

        redis_predict_db = self.model_dbs[model["no"]]
        cnt = 0

        while True:
            cnt += 1
            if cnt > 150:
                break

            result = redis_predict_db.zrangebyscore(model["db_name"], score, score,withscores=True)
            # conf.LOGGER(result)
            if len(result) == 1:
                line = result[0]
                body = line[0]

                tmps = json.loads(body)
                response = tmps["response"]

                return response

            time.sleep(0.005)

        raise Exception("cannot get predict", model["no"])

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

        # lstmのモデルによる予想
        for model in base_models:
            if self.mode == "test":
                response = self.get_predict_lstm_self(closes, model, score)
            else:
                response = self.get_predict_lstm(score, model)

            if model["type"] == "CATEGORY":
                up,same,dw = response.split("_")
                self.predict_dict[model["no"] + "-UP"].append(float(up))
                self.predict_dict[model["no"] + "-SAME"].append(float(same))
                self.predict_dict[model["no"] + "-DW"].append(float(dw))
            elif model["type"] == "REGRESSION":
                self.predict_dict[model["no"] + "-REG"].append(float(response))


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



