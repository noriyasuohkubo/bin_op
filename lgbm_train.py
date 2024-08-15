import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from decimal import Decimal
import psutil
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import pickle
from lgbm_make_data import LgbmMakeData
import conf_class_lgbm
import random as rn
import socket
from util import *
import optuna
import optuna.integration.lightgbm as lgbm_opt
import redis
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error
import pdb
import json
from util import *
from math import sqrt
import subprocess
import send_mail as mail
host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + "-lgbm.txt"
output = output_log(output_log_name)

'''
モデル学習
'''
class Objective:
    def __init__(self, conf, params, train_data, eval_data, x_eval, y_eval, cat_list, verbose_eval):
        self.conf = conf
        self.params = params
        self.train_data = train_data
        self.eval_data = eval_data
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.cat_list = cat_list
        self.verbose_eval = verbose_eval

    def __call__(self, trial):
        #optuna用ハイパーパラメータの設定
        for k, v in self.conf.LGBM_OPTUNA_PARAM_DICT.items():
            if isinstance(v[0], int):
                #intの場合
                self.params[k] = trial.suggest_int(k, v[0], v[1]) #suggest_int(name, low, high, step=1, log=False)
            else:
                #floatの場合
                self.params[k] = trial.suggest_float(k, v[0], v[1]) #suggest_float(name：str、low：float、high：float、*、step：Optional [float] = None)

        #パラメータ探索用学習
        model = lgbm.train(
                           params=self.params,
                           train_set=self.train_data,
                           valid_sets=[self.train_data, self.eval_data],
                           valid_names=['train', 'eval'],
                           #evals_result=evaluation_results,
                           #verbose_eval=1,  # イテレーション毎に学習結果出力
                           #early_stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'],
                           num_boost_round=conf.OTHER_PARAM_DICT['num_boost_round'],
                           categorical_feature=self.cat_list,
                           callbacks=[lgbm.early_stopping(stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'],verbose=True),  # early_stopping用コールバック関数
                                       lgbm.log_evaluation(self.verbose_eval)]  # コマンドライン出力用コールバック関数
        )

        #予想取得
        y_predict = model.predict(data=self.x_eval, num_iteration=model.best_iteration)
        #スコア取得
        if conf.METRIC == "rmse":
            score = sqrt(mean_squared_error(self.y_eval, y_predict))
        else:
            score = log_loss(y_true=self.y_eval, y_pred=y_predict)

        #output(score)
        return score

class LgbmTrain():

    def __init__(self, conf, test_lmd, train_lmd, train_start, train_end, test_start, test_end, ):
        self.conf = conf

        self.x_eval, self.y_eval, self.x_train, self.y_train = self.get_xy(conf, test_lmd, train_lmd, train_start, train_end, test_start, test_end, )

        # columns = train_lmd.get_columns()
        output("INPUT_DATA:", conf.INPUT_DATA)
        self.train_data = lgbm.Dataset(
            data=self.x_train,
            label=self.y_train,
            feature_name=conf.INPUT_DATA,
        )

        self.eval_data = lgbm.Dataset(
            data=self.x_eval,
            label=self.y_eval,
            feature_name=conf.INPUT_DATA,
            reference=self.train_data,
        )

        # print(train_data.data)
        # print(train_data.label)

        self.cat_list = conf.CATEGORY_INPUT
        if conf.USE_H:
            self.cat_list.extend(["hour"])
        if conf.USE_M:
            self.cat_list.extend(["min"])
        if conf.USE_S:
            self.cat_list.extend(["sec"])
        if conf.USE_W:
            self.cat_list.extend(["week"])
        if conf.USE_WN:
            self.cat_list.extend(["weeknum"])

        self.verbose_eval = 1  # この数字を1にすると学習時のスコア推移がコマンドライン表示される

        #共通パラメータセット
        #see:https://lightgbm.readthedocs.io/en/latest/Parameters.html
        self.params = conf.LGBM_PARAM_DICT
        self.params['metric'] = conf.METRIC
        self.params['objective'] = conf.OBJECTIVE
        self.params['task'] = 'train'
        #self.params['verbosity'] = 1  # 1：Info, 0：Error(Warning), -1：Fatal
        self.params['verbose_eval'] = self.verbose_eval
        self.params['verbose'] = -1  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される


        if conf.LEARNING_TYPE == "CATEGORY":
            self.params["num_classes"] = 3

        if conf.LGBM_PARAM_DICT['device_type'] == "gpu":
            self.params["gpu_device_id"] = conf.GPU_DEVICE_ID

    #see:https://blog.amedama.jp/entry/lightgbm-custom-metric
    def accuracy(self, preds, data):
        """精度 (Accuracy) を計算する関数"""
        # 正解ラベル
        y_true = data.get_label()
        # 推論の結果が 1 次元の配列になっているので直す
        N_LABELS = 3  # ラベルの数
        reshaped_preds = preds.reshape(N_LABELS, len(preds) // N_LABELS)
        # 最尤と判断したクラスを選ぶ　
        y_pred = np.argmax(reshaped_preds, axis=0)
        # メトリックを計算する
        acc = np.mean(y_true == y_pred)
        # name, result, is_higher_better
        return 'accuracy', acc, True

    def show_result(self, model, evaluation_results):
        conf = self.conf
        if ("REGRESSION" in conf.LEARNING_TYPE) == False and conf.LEARNING_TYPE != "CATEGORY":
            #パフォーマンスを可視化
            #参考:http://knknkn.hatenablog.com/entry/2019/02/02/164829
            #Accuracy score: 正解率。1のものは1として分類(予測)し、0のものは0として分類した割合
            #Precision score: 精度。1に分類したものが実際に1だった割合
            #Recall score: 検出率。1のものを1として分類(予測)した割合
            #F1 score: PrecisionとRecallを複合した0~1のスコア。数字が大きいほど良い評価。
            preds = model.predict(self.x_eval, num_iteration=model.best_iteration)
            preds_round = np.round(preds)
            output('Accuracy score = \t {}'.format(accuracy_score(self.y_eval, preds_round)))
            output('Precision score = \t {}'.format(precision_score(self.y_eval, preds_round)))
            output('Recall score =   \t {}'.format(recall_score(self.y_eval, preds_round)))
            output('F1 score =      \t {}'.format(f1_score(self.y_eval, preds_round)))

            # 混同行列（Confusion Matrix）の表示
            cm = confusion_matrix(self.y_eval, preds_round)
            print(cm)

        # Plot the log loss during training
        eval_loss_list = evaluation_results['eval'][conf.METRIC]
        train_loss_list = evaluation_results['train'][conf.METRIC]

        output("best iteration:", eval_loss_list.index(min(eval_loss_list)) + 1, min(eval_loss_list))

        for i,(t,e) in enumerate(zip(train_loss_list,eval_loss_list)):
            if i % 1 == 0:
                output(i+1, ",", t, ",",e)

        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importance': model.feature_importance()}).sort_values('importance', ascending=False)
        for row in importances.itertuples():
            output(row.features, ",", row.importance)

        fig, axs = plt.subplots(1, 2, figsize=[15, 4])

        # Plot the log loss during training
        axs[0].plot(evaluation_results['train'][conf.METRIC], label='train')
        axs[0].plot(evaluation_results['eval'][conf.METRIC], label='eval')
        axs[0].set_ylabel('Log loss')
        axs[0].set_xlabel('Boosting round')
        axs[0].set_title('Training performance')
        axs[0].legend()

        # Plot feature importance

        axs[1].bar(x=np.arange(len(importances)), height=importances['importance'])
        axs[1].set_xticks(np.arange(len(importances)))
        axs[1].set_xticklabels(importances['features'])
        axs[1].set_ylabel('Feature importance (# times used to split)')
        axs[1].set_title('Feature importance')

        plt.show()

    def get_xy(self, conf, test_lmd, train_lmd, train_start, train_end, test_start, test_end, ):
        #evalデータ作成
        x_eval = test_lmd.get_x()
        x_eval = x_eval.loc[:, conf.INPUT_DATA]

        if conf.LEARNING_TYPE == "CATEGORY":
            y_eval = test_lmd.get_y(conf)
        elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP":
            y_eval = test_lmd.get_y_up(conf)
        elif conf.LEARNING_TYPE == "CATEGORY_BIN_DW":
            y_eval = test_lmd.get_y_dw(conf)
        elif conf.LEARNING_TYPE == "REGRESSION":
            y_eval = test_lmd.get_y_r()

        if test_start != None and test_end != None:
            #開始終了期間をしぼる
            start_score = int(time.mktime(test_start.timetuple()))
            end_score = int(time.mktime(test_end.timetuple()))
            y_eval.index = x_eval.index
            #x_evalにy_eval列追加
            x_eval['y_eval'] = y_eval
            #期間をしぼる
            x_eval.query('@start_score <= score < @end_score', inplace=True)
            #y_evalのみ改めて抽出
            y_eval = x_eval.loc[:, 'y_eval']
            # 不要データ削除
            x_eval.drop(['y_eval'], axis=1, inplace=True)

            if len(x_eval) != len(y_eval):
                print("x_eval,y_evalのデータ長が異なる:", len(x_eval), len(y_eval))
                exit(1)

        #trainデータ作成
        x_train = train_lmd.get_x()
        x_train = x_train.loc[:, conf.INPUT_DATA]

        if conf.LEARNING_TYPE == "CATEGORY":
            y_train = train_lmd.get_y(conf)
        elif conf.LEARNING_TYPE == "CATEGORY_BIN_UP":
            y_train = train_lmd.get_y_up(conf)
        elif conf.LEARNING_TYPE == "CATEGORY_BIN_DW":
            y_train = train_lmd.get_y_dw(conf)
        elif conf.LEARNING_TYPE == "REGRESSION":
            y_train = train_lmd.get_y_r()

        if train_start != None and train_end != None :
            #開始終了期間をしぼる
            start_score = int(time.mktime(train_start.timetuple()))
            end_score = int(time.mktime(train_end.timetuple()))
            y_train.index = x_train.index
            x_train['y_train'] = y_train
            #期間をしぼる
            x_train.query('@start_score <= score < @end_score', inplace=True)
            y_train = x_train.loc[:, 'y_train']
            x_train.drop(['y_train'], axis=1, inplace=True)

            if len(x_train) != len(y_train):
                print("x_train,y_trainのデータ長が異なる:", len(x_train), len(y_train))
                exit(1)

        print("x_eval info")
        print(x_eval.info())

        print("x_train info")
        print(x_train.info())

        return x_eval,y_eval, x_train, y_train

    #optuna使用
    def do_train_optuna(self):
        obj = Objective(conf, self.params, self.train_data, self.eval_data, self.x_eval, self.y_eval, self.cat_list, self.verbose_eval)
        study = optuna.create_study(direction='minimize')  #directionのdefault:'minimize'
        study.optimize(
                        func = obj,
                        n_trials=conf.OTHER_PARAM_DICT['n_trials'], #探索回数
                        n_jobs=1, #-1なら全CPU使用
                        show_progress_bar=True,
        )
        output("trials:")
        for trial in study.trials:
            print(trial)
        output("")
        output("Best trial:")
        trial = study.best_trial

        output("  Value: {}".format(trial.value))

        output("  Params: ")
        for key, value in trial.params.items():
            output("    {}: {}".format(key, value))

        output('BEST PARAMS:', study.best_params)

        evaluation_results = {}  # 評価結果格納用

        for k,v in study.best_params.items():
            self.params[k] = v

        #一番良いパラメータでモデル作成
        model = lgbm.train(
            params= self.params,
            train_set=self.train_data,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=['train', 'eval'],
            #evals_result=evaluation_results,
            #verbose_eval=1,  # イテレーション毎に学習結果出力
            #early_stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'],
            num_boost_round=conf.OTHER_PARAM_DICT['num_boost_round'],
            categorical_feature=self.cat_list,
            callbacks=[
                lgbm.early_stopping(stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'], verbose=True),
                # early_stopping用コールバック関数
                lgbm.log_evaluation(self.verbose_eval),
                lgbm.record_evaluation(evaluation_results)]  # コマンドライン出力用コールバック関数
        )

        return model, evaluation_results


    def do_train(self):
        conf = self.conf

        evaluation_results = {}  # 評価結果格納用
        model = lgbm.train(
            params=self.params,
            train_set=self.train_data,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=['train', 'eval'],
            #evals_result=evaluation_results,
            #verbose_eval=1,  # イテレーション毎に学習結果出力
            #early_stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'],
            num_boost_round=conf.OTHER_PARAM_DICT['num_boost_round'],
            categorical_feature= self.cat_list,
            callbacks=[
                lgbm.early_stopping(stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'], verbose=True),
                # early_stopping用コールバック関数
                lgbm.log_evaluation(self.verbose_eval),
                lgbm.record_evaluation(evaluation_results)],  # コマンドライン出力用コールバック関数

        )

        return model, evaluation_results


    def do_train_tuner(self):
        conf = self.conf

        self.params['deterministic'] = True, #再現性確保用のパラメータ
        self.params['force_row_wise'] = True  #再現性確保用のパラメータ

        """
        LightGBMTunerのデフォルトの探索範囲
            param = {
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
        """

        tuner = lgbm_opt.LightGBMTuner(
            params=self.params,
            train_set=self.train_data,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=['train', 'eval'],
            verbose_eval=100,  # イテレーション毎に学習結果出力
            early_stopping_rounds= conf.OTHER_PARAM_DICT['early_stopping_rounds'],
            num_boost_round = conf.OTHER_PARAM_DICT['num_boost_round'],
            categorical_feature=self.cat_list,
            optuna_seed=conf.LGBM_PARAM_DICT['seed'],  # 再現性確保用のパラメータ
        )
        #パラメータ探索
        tuner.run()

        output('BEST PARAMS:', tuner.best_params,)

        for k,v in tuner.best_params.items():
            self.params[k] = v

        evaluation_results = {}  # 評価結果格納用

        #一番良いパラメータでモデル作成
        model = lgbm.train(
            params=self.params,
            train_set=self.train_data,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=['train', 'eval'],
            #evals_result=evaluation_results,
            #verbose_eval=1,  # イテレーション毎に学習結果出力
            #early_stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'],
            num_boost_round=conf.OTHER_PARAM_DICT['num_boost_round'],
            categorical_feature=self.cat_list,
            callbacks=[
                lgbm.early_stopping(stopping_rounds=conf.OTHER_PARAM_DICT['early_stopping_rounds'], verbose=True),
                # early_stopping用コールバック関数
                lgbm.log_evaluation(self.verbose_eval),
                lgbm.record_evaluation(evaluation_results)]  # コマンドライン出力用コールバック関数
        )

        return model, evaluation_results


if __name__ == '__main__':
    conf = conf_class_lgbm.ConfClassLgbm()

    np.random.seed(conf.LGBM_PARAM_DICT['seed'])
    rn.seed(conf.LGBM_PARAM_DICT['seed'])

    output("FILE_PREFIX_DB", conf.FILE_PREFIX_DB)

    #すでにlgbm_make_data.pyでデータが作成されている場合 なければ空文字
    #train_data_load_path = "/db2/lgbm/" + conf.SYMBOL + "/train_file/TRAF17.pickle"
    #test_data_load_path = "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF17.pickle"
    train_data_load_path = ""
    test_data_load_path = ""


    print("test_data_load_path:",test_data_load_path)
    print("train_data_load_path:",train_data_load_path)

    test_start = conf.EVAL.split("_")[1]
    test_end = conf.EVAL.split("_")[2]

    tmp_start_year = int(test_start[0:4])
    tmp_start_month = int(test_start[4:6])
    tmp_start_day = int(test_start[6:])

    tmp_end_year = int(test_end[0:4])
    tmp_end_month = int(test_end[4:6])
    tmp_end_day = int(test_end[6:])

    test_start = datetime(tmp_start_year, tmp_start_month, tmp_start_day)
    test_end = datetime(tmp_end_year, tmp_end_month, tmp_end_day) + timedelta(days=1)

    if test_data_load_path != "":
        with open(test_data_load_path, 'rb') as f:
            test_lmd = pickle.load(f)
    else:
        # evalデータ作成
        test_lmd = LgbmMakeData()

        org_file_name = "MF203"
        org_file_path = "/db2/lgbm/" + conf.SYMBOL + get_lgbm_file_type(org_file_name) + org_file_name + ".pickle"
        print("test org_file_path:",org_file_path)

        target_spread_list = []
        test_lmd.make_data(conf, org_file_path, test_start, test_end, True, target_spread_list=target_spread_list)

        test_lmd.save_data(conf, test_lmd, org_file_name, test_start, test_end, True, target_spread_list)

        #学習時にさらに開始終了を絞り込まないようにNoneにする
        test_start = None
        test_end = None
        """
        # メモリ節約のためredis停止
        #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
        sudo_password = 'Reikou0129'
        command = 'systemctl stop redis'.split()
        p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
        sudo_prompt = p.communicate(sudo_password + '\n')[1]
        # メモリ空き容量を取得
        print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        """

    train_start = conf.SUFFIX.split("_")[1]
    train_end = conf.SUFFIX.split("_")[2]

    tmp_start_year = int(train_start[0:4])
    tmp_start_month = int(train_start[4:6])
    tmp_start_day = int(train_start[6:])

    tmp_end_year = int(train_end[0:4])
    tmp_end_month = int(train_end[4:6])
    tmp_end_day = int(train_end[6:])

    train_start = datetime(tmp_start_year, tmp_start_month, tmp_start_day)
    train_end = datetime(tmp_end_year, tmp_end_month, tmp_end_day) + timedelta(days=1)

    if train_data_load_path != "":
        with open(train_data_load_path, 'rb') as f:
            train_lmd = pickle.load(f)
    else:
        #trainデータ作成
        train_lmd = LgbmMakeData()

        org_file_name = "MF191"
        org_file_path = "/db2/lgbm/" + conf.SYMBOL + get_lgbm_file_type(org_file_name) + org_file_name + ".pickle"
        print("train org_file_path:",org_file_path)

        target_spread_list = []
        train_lmd.make_data(conf, org_file_path, train_start, train_end, False, target_spread_list=target_spread_list)

        train_lmd.save_data(conf, train_lmd, org_file_name, train_start, train_end, False, target_spread_list)

        #学習時にさらに開始終了を絞り込まないようにNoneにする
        train_start = None
        train_end = None

    # 処理時間計測
    start = time.time()

    # モデル番号付与
    conf.numbering()

    lgbm_train = LgbmTrain(conf, test_lmd, train_lmd, train_start, train_end, test_start, test_end)

    #学習
    start = time.time()

    output("TUNER_TYPE:", conf.TUNER_TYPE)

    if conf.TUNER_TYPE == 'NORMAL':
        model, evaluation_results = lgbm_train.do_train()
    elif conf.TUNER_TYPE == 'OPTUNA':
        model, evaluation_results = lgbm_train.do_train_optuna()
    elif conf.TUNER_TYPE == 'TUNER':
        model, evaluation_results = lgbm_train.do_train_tuner()

    output("train process time:", (time.time() - start) / 60, "分")
    output('Saving model...')

    # 保存
    model.save_model(filename=conf.MODEL_DIR + conf.FILE_PREFIX, num_iteration=-1)
    output("FILE_PREFIX: " + conf.FILE_PREFIX)

    # 終わったらメールで知らせる
    mail.send_message(host, ": lgbm train finished!!!")

    lgbm_train.show_result(model, evaluation_results)

    process_time = time.time() - start
    output("process_time:", process_time / 60, "分")