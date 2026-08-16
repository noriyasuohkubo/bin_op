import time

import conf_class_svm
import send_mail as mail
import socket
import psutil
from datetime import datetime
from DataSequence2 import DataSequence2
import numpy as np
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import redis
import json
from datetime import datetime, timedelta
import pickle
from util import *
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import cuml

# コンピュータ名を取得
host = socket.gethostname()

svc = SVC()

#cuML SVC
#cuMLのSVCがnumpyを返すようにする
#see:https://qiita.com/shoji18/items/829d6e6703a62625449f
class Cuml_SVC(cuml.svm.SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def predict(self, X):
        y_pred = super().predict(X).to_array().astype(int)
        return y_pred

class Cuml_SVR(cuml.svm.SVR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def predict(self, X):
        y_pred = super().predict(X).to_array().astype(float)
        return y_pred

#学習用、テスト用それぞれのデータをpickleで保存
def save_data(conf, file, startDt, endDt, test_flg, target_spread_list, file_postscript = "", ):
    end_tmp = endDt + timedelta(days=-1)

    target_spread_list_str = "_TSL" + list_to_str(target_spread_list) if len(target_spread_list) != 0 else ""

    if test_flg == False:
        except_list_str = "_EL" + list_to_str(conf.EXCEPT_LIST, spl="-") if len(conf.EXCEPT_LIST) != 0 else ""
    else:
        except_list_str = ""

    tmp_file_name = conf.SYMBOL + conf.LEARNING_TYPE_STR + conf.METHOD_STR + "_B" + str(conf.BET_TERM) + "_BS" + str(conf.BET_SHIFT) + "_T" + str(conf.TERM) +  \
                       conf.DB_SYMBOLS_STR + conf.DB1_NOT_LEARN_STR + "_I" + conf.DB_TERM_STR + \
                       "_IL" + conf.INPUT_LEN_STR + conf.BORDER_STR + conf.INPUT_DATA_STR + conf.OUTPUT_TYPE_STR + conf.OUTPUT_DATA_STR + \
                       target_spread_list_str + \
                       "_" + date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,format='%Y%m%d') + except_list_str + "_" + socket.gethostname() + file_postscript

    if test_flg:
        db_name_file = "TEST_FILE_NO_" + conf.SYMBOL
    else:
        db_name_file = "TRAIN_FILE_NO_" + conf.SYMBOL

    # win2のDBを参照してモデルのナンバリングを行う
    r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
    result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
    if len(result) == 0:
        print("CANNOT GET FILE_NO", db_name_file)
        exit(1)
    else:
        newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

        for line in result:
            body = line[0]
            score = float(line[1])
            tmps = json.loads(body)
            tmp_name = tmps.get("input_name")
            if tmp_name == tmp_file_name:
                # 同じファイルがないが確認
                print("The File Already Exists!!!", tmp_file_name)
                exit(1)

        # DBにモデルを登録
        child = {
            'input_name': tmp_file_name,
            'no': newest_no
        }
        r.zadd(db_name_file, json.dumps(child), newest_no)

    makedirs("/db2/svm/" + conf.SYMBOL + "/train_file/TESF") # dirなければつくる
    makedirs("/db2/svm/" + conf.SYMBOL + "/test_file/TRAF")

    if test_flg:
        data_save_path = "/db2/svm/" + conf.SYMBOL + "/test_file/TESF" + str(newest_no) + ".pickle"
        conf_save_path = "/db2/svm/" + conf.SYMBOL + "/test_file/TESF" + str(newest_no) + "-conf.pickle"
    else:
        data_save_path = "/db2/svm/" + conf.SYMBOL + "/train_file/TRAF" + str(newest_no) + ".pickle"
        conf_save_path = "/db2/svm/" + conf.SYMBOL + "/train_file/TRAF" + str(newest_no) + "-conf.pickle"

    print("newest_no", newest_no)
    print("input_name", tmp_file_name)
    # データをpickleで保存する
    with open(data_save_path, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
        pickle.dump(file, f)
        print("data_save_path:",data_save_path)

    with open(conf_save_path, 'wb') as f:  # confを保存しておく
        pickle.dump(conf, f)

def make_data(dataSequence2):
    newX = None
    newY = None
    for idx in range(dataSequence2.steps_per_epoch):
    #for idx in range(1):
        retX, retY = dataSequence2.__getitem__(idx)
        # retXはlistで中がnumpy
        # retYはnumpy

        # SVM用に特徴量を変形させる
        tmpX = []
        for x in retX:
            # 特徴量は1つの前提なのでshapeを変形する
            x = x.reshape(x.shape[0], x.shape[1])
            tmpX.append(x)

        if idx == 0:
            newX = np.concatenate(tmpX, 1)
            newY = retY
        else:
            newX = np.concatenate([newX, np.concatenate(tmpX, 1)])
            newY = np.concatenate([newY, retY])

    return [newX, newY]

def get_train_data(train_file_path):

    #学習用データ作成
    if train_file_path == "":
        print(datetime.now()," START MAKE TRAIN DATA")

        # 学習開始時期がファイル名設定と合致しているかチェック
        print(conf.SUFFIX.split("_")[1][:4], str(start.year))
        if conf.SUFFIX.split("_")[1][:4] != str(start.year):
            exit(1)

        print(datetime.now(), "dataSequence2 make start")
        conf.change_real_spread_flg(False)
        conf.change_fx_real_spread_flg(False)
        dataSequence2 = DataSequence2(conf, start, end, False, False, )
        print(datetime.now(), "dataSequence2 make end")

        trainData= make_data(dataSequence2)

        #save data
        save_data(conf, trainData, start, end, False, conf.TARGET_SPREAD_LISTS, file_postscript="")
        trainX = trainData[0]
        trainY = trainData[1]

        print(trainX.shape, trainY.shape)
        print(trainX[0])
        print(trainY[:100])

    else:
        with open(train_file_path, 'rb') as f:
            trainData = pickle.load(f)
            trainX = trainData[0]
            trainY = trainData[1]

    return trainX, trainY


def get_test_data(start_eval, end_eval):
    #テスト用データ作成
    print(datetime.now(), " START MAKE TEST DATA")

    print(datetime.now(), "dataSequence2 eval make start")
    conf.change_real_spread_flg(False)
    conf.change_fx_real_spread_flg(False)
    dataSequence2_eval = DataSequence2(conf, start_eval, end_eval, True, True, )
    print(datetime.now(), "dataSequence2 eval make end")

    testData= make_data(dataSequence2_eval)

    testX = testData[0]
    testY = testData[1]

    print(testX.shape, testY.shape)
    print(testX[0])
    print(testY[:100])

    return testX, testY, dataSequence2_eval

if __name__ == '__main__':
    # 設定ファイル
    conf = conf_class_svm.ConfClass()
    print("FILE_PREFIX", conf.FILE_PREFIX)

    conf.numbering()  # モデル番号付与
    print("MODEL_DIR:", conf.MODEL_DIR)

    if os.path.isfile(conf.MODEL_DIR):
        print("ERROR!! MODEL_DIR Already Exists ")
        exit(1)

    start = datetime(2023, 3, 1, )
    end = datetime(2023, 4, 1)

    # 学習用ファイル
    train_file_path = "/db2/svm/USDJPY/train_file/TRAF282.pickle"
    #train_file_path = ""

    trainX, trainY = get_train_data(train_file_path)

    # 学習開始
    print(datetime.now(), " START TRAIN")
    makedirs("/app/model_svm/bin_op/")

    if conf.SVM_GS_FLG:
        # GridSearchCVのインスタンスを作成&学習&スコア記録
        if conf.METHOD == "SVC":
            gscv = GridSearchCV(cuml.svm.SVC(), conf.SVM_PARAMS, n_jobs=conf.SVM_GS_NJOBS, cv= conf.SVM_GS_CV, scoring=conf.SVC_SCORING, verbose=3)
        elif conf.METHOD == "SVR":
            gscv = GridSearchCV(cuml.svm.SVR(), conf.SVM_PARAMS, n_jobs=conf.SVM_GS_NJOBS, cv= conf.SVM_GS_CV, scoring=conf.SVR_SCORING, verbose=3)

        gscv.fit(trainX, trainY)

        print(gscv.cv_results_)

        # 最適なパラメータの組み合わせ
        print("best_params:", gscv.best_params_)
        # BEST SCORE
        print("best_score:", gscv.best_score_)

        # modelをpickleで保存する
        with open(conf.MODEL_DIR, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
            pickle.dump(gscv, f)

        # 最適なパラメータを用いたモデル
        testX = np.array([[1]])
        start_pred = time.perf_counter()
        pred = gscv.best_estimator_.predict(testX)
        print(pred)
        print(time.perf_counter() - start_pred)
    else:

        if conf.METHOD == "SVC":
            model = cuml.svm.SVC(**conf.SVM_PARAMS, verbose=3)
        elif conf.METHOD == "SVR":
            model = cuml.svm.SVR(**conf.SVM_PARAMS, verbose=3)

        model.fit(trainX, trainY)

        # modelをpickleで保存する
        with open(conf.MODEL_DIR, 'wb') as f:  # 新規作成、存在していれば上書き b:バイナリ
            pickle.dump(model, f)

        testX = np.array([[1]])
        start_pred = time.perf_counter()
        pred = model.predict(testX)
        print(pred)
        print(time.perf_counter() - start_pred)

    # 終わったらメールで知らせる
    mail.send_message(host, ": svm_train finished!!!")