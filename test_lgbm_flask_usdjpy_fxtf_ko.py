from pathlib import Path
import numpy as np
from datetime import datetime
import time
from decimal import Decimal
import redis
import json
import random
import scipy.stats
import gc
import math
from subprocess import Popen, PIPE
import pandas as pd
import lightgbm as lgb
from tensorflow.keras.models import load_model
from util import *
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
lstmモデルの予想結果を特徴量にしたlgbmモデルの予想結果を返すflaskのテスト用

lgbmのテスト用pandasからランダムに抽出した予想日時とlstmの予想結果および、lgbmモデルの予想結果が
作成しようとしているflaskのものとそれぞれ一致するかテストする
"""

SYMBOL = "USDJPY"
BASE_TERM = 1

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_wma(value):
    # 加重移動平均
    weight = np.arange(len(value)) + 1
    wma = np.sum(weight * value) / weight.sum()

    return wma

def get_max(value):
    return np.max(value)

def get_min(value):
    return np.min(value)


def get_x(init_flg, score, data_length, input_datas, input_separate_flg, method, closes=None):
    dt = datetime.fromtimestamp(score)
    now_sec = dt.second
    sec_oh_arr = [int(Decimal(str(now_sec)) / Decimal(str(BASE_TERM)))]  # 2秒間隔データなら０から29に変換しなければならないのでBASE_TERMで割る
    min_oh_arr = [dt.minute]
    hour_oh_arr = [dt.hour]

    SEC_OH_LEN = int(Decimal("60") / Decimal(str(BASE_TERM)))
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
                close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM))) ))

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
                close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM))) ))
                aft = np.apply_along_axis(get_wma, 1, close_n)[(-1 * len) -1:]
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
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X1[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
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
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
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
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))

                    aft = close_n[(-1 * len) - 1:, -1]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X1[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X2[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
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
                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))

                    aft = close_n[(-1 * len) - 1:, -1]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 0] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_max, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 1] = target

                    close_n = np.reshape(closes, (-1, int(Decimal(str(s)) / Decimal(str(BASE_TERM)))))
                    aft = np.apply_along_axis(get_min, 1, close_n)[(-1 * len) - 1:]
                    bef = np.roll(aft, 1)  # 1つ前にずらす
                    target = get_divide_arr(bef, aft)[1:]
                    X[:, :, 2] = target

                retX.append(X)


    if method == "LSTM7":
        retX.append(np.identity(SEC_OH_LEN)[sec_oh_arr])
        retX.append(np.identity(MIN_OH_LEN)[min_oh_arr])
        retX.append(np.identity(HOUR_OH_LEN)[hour_oh_arr])

    return retX

#テスト用pandasからランダムに1件抽出

# ファイル読み込み
file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF218.pickle"
test_lmd = pd.read_pickle(file_path)
df = test_lmd.get_x()

print(df.info())

index_list = list(df.index)

key_index = 1682519033 + 4

if key_index == None:
    while True:
        scores = []
        tmp_key_index = index_list[random.randint(0, len(index_list))]#randint(a, b)はa <= n <= bのランダムな整数intを返す。
        scores.append(tmp_key_index)
        past_terms = [i for i in range(4, 21, 4)]
        ok_flg = True
        for i in past_terms:
            scores.append(tmp_key_index - i)
            #過去予想時のスコアの存在チェック
            if ((tmp_key_index - i) in index_list) == False:
                ok_flg = False
                break
        if ok_flg == True:
            break
    key_index = tmp_key_index - 4 #lgbmの予想を二回するテストに備えて一つ余分にしておく

#LGBMモデル設定
model_file = "MN733"
suffix = 85
bst = lgb.Booster(model_file="/app/model_lgbm/bin_op/" + model_file)
tmp_input = '704-4-REG@704-4-REG-12@704-4-REG-4@704-4-REG-8@712-36-DW@712-36-DW-12@712-36-DW-4@712-36-DW-8@712-36-SAME@712-36-SAME-12@712-36-SAME-4@712-36-SAME-8@712-36-UP@712-36-UP-12@712-36-UP-4@712-36-UP-8@714-36-DW@714-36-DW-12@714-36-DW-4@714-36-DW-8@714-36-SAME@714-36-SAME-12@714-36-SAME-4@714-36-SAME-8@714-36-UP@714-36-UP-12@714-36-UP-4@714-36-UP-8@715-40-DW@715-40-DW-12@715-40-DW-4@715-40-DW-8@715-40-SAME@715-40-SAME-12@715-40-SAME-4@715-40-SAME-8@715-40-UP@715-40-UP-12@715-40-UP-4@715-40-UP-8@hour@min@sec@weeknum'.split("@")

#d1をlgbmの特徴量とする場合
"""
lgbm_ds =[
    {
        "data_length": 2,
        "data_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 101, 5)],
    }
]
"""

lgbm_ds = [] #d1をlgbmの特徴量としない場合

x = df.loc[[key_index], tmp_input]
print(x.info())

x_dixt = x.to_dict(orient='index')
col1 = sorted(x_dixt[key_index].items())
# 正解予想取得
predict_list1 = bst.predict(x, num_iteration=int(suffix))


redis_db = redis.Redis(host='127.0.0.1', port=6379, db=2, decode_responses=True)
db_name = 'USDJPY_1_0'

base_model_dir = "/app/model/bin_op/"
base_models =[
    {
        "name":'USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.25_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub1_MN715-40',
        "no":"715-40",
        "type":"CATEGORY",
        "data_length":[[1,300],[5,300],[30,240],],
        "input_datas":["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg":True,
        "method":"LSTM7",
    },
    {
        "name": 'USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.1_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub2_MN712-36',
        "no": "712-36",
        "type": "CATEGORY",
        "data_length":[[1,300],[5,300],[30,240],],
        "input_datas":["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'USDJPY_LT8_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_201701_202303_L-RATE0.0005_LT5_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub1_MN704-4',
        "no": "704-4",
        "type": "REGRESSION",
        "data_length":[[1,300],[5,300],[30,240],],
        "input_datas":["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.5_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub3_MN714-36',
        "no": "714-36",
        "type": "CATEGORY",
        "data_length":[[1,300],[5,300],[30,240],],
        "input_datas":["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },


]

#モデルをロードしておく
models = {}

for base_model in base_models:
    model_tmp = load_model(base_model_dir + base_model["name"],custom_objects={"root_mean_squared_error": root_mean_squared_error,})
    #最初に一度推論させてグラフ作成し二回目以降の推論を早くする
    model_tmp.predict_on_batch(get_x(True, key_index, base_model["data_length"], base_model["input_datas"], base_model["input_separate_flg"], base_model["method"], closes=None))
    models[base_model["name"]] = model_tmp

#予想を、モデルNoをキー、予想をリストにして保持
predict_dict= {}
for model in base_models:
    if model["type"] == "CATEGORY":
        predict_dict[model["no"] + "-UP"] = []
        predict_dict[model["no"] + "-SAME"] = []
        predict_dict[model["no"] + "-DW"] = []
    elif model["type"] == "REGRESSION":
        predict_dict[model["no"] + "-REG"] = []

#過去の予想
past_terms = [i for i in range(4, 13, 4)]
past_terms = past_terms[::-1] #リストを逆順にする

for past_term in past_terms:
    result_data = redis_db.zrangebyscore(db_name, key_index - past_term - (7230 * 1), key_index - past_term - 1, withscores=True)
    # print(len(result_data))
    closes = []

    for i, line in enumerate(result_data):
        body = line[0]
        score = int(line[1])
        tmps = json.loads(body)
        closes.append(tmps["c"])

    for model in base_models:
        # model_name = model["name"]
        # model_tmp = load_model(base_model_dir + model_name,custom_objects={"root_mean_squared_error": root_mean_squared_error,})
        model_tmp = models[model["name"]]
        x = get_x(False, key_index - past_term, model["data_length"], model["input_datas"], model["input_separate_flg"], model["method"], closes=closes)
        # print(x)
        t1 = time.time()
        predict = model_tmp.predict_on_batch(x)
        #print("経過時間：" + str(time.time() - t1))
        #print(predict)
        if model["type"] == "CATEGORY":
            predict_dict[model["no"] + "-UP"].append(predict[0][0])
            predict_dict[model["no"] + "-SAME"].append(predict[0][1])
            predict_dict[model["no"] + "-DW"].append(predict[0][2])
        elif model["type"] == "REGRESSION":
            predict_dict[model["no"] + "-REG"].append(predict[0][0])

#最新の予想
result_data = redis_db.zrangebyscore(db_name, key_index - (7230 * 1), key_index -1, withscores=True)
#print(len(result_data))
closes = []

for i, line in enumerate(result_data):
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)
    closes.append(tmps["c"])

for model in base_models:
    #model_name = model["name"]
    #model_tmp = load_model(base_model_dir + model_name,custom_objects={"root_mean_squared_error": root_mean_squared_error,})

    model_tmp = models[model["name"]]
    #res = model_tmp.predict(get_x_old(True), verbose=0, batch_size=1)
    #print(res)
    x = get_x(False, key_index, model["data_length"], model["input_datas"], model["input_separate_flg"], model["method"], closes=closes)
    #print(x)
    predict = model_tmp.predict_on_batch(x)
    #print(predict[0])
    if model["type"] == "CATEGORY":
        predict_dict[model["no"] + "-UP"].append(predict[0][0])
        predict_dict[model["no"] + "-SAME"].append(predict[0][1])
        predict_dict[model["no"] + "-DW"].append(predict[0][2])
    elif model["type"] == "REGRESSION":
        predict_dict[model["no"] + "-REG"].append(predict[0][0])

print(predict_dict)

past_terms = [i for i in range(0, 13, 4)]
past_terms = past_terms[::-1] #リストを逆順にする

#lgbmモデルに渡す特徴量(pandas)を作成する元の列
base_col_dict = {}

for no,predict_list in predict_dict.items():

    for pt,p in zip(past_terms, predict_list):
        if pt == 0:
            base_col_dict[no] = [p]
        else:
            base_col_dict[no + "-" + str(pt)] = [p]

if len(lgbm_ds) != 0:
    #d1をlgbmモデルの特徴量とする場合
    for i, ds in enumerate(lgbm_ds):
        data_length = ds["data_length"]
        data_idx = ds["data_idx"]

        close_n = np.reshape(closes, (-1, int(Decimal(str(data_length)) / Decimal(str(BASE_TERM))) ))

        for idx in data_idx:
            aft = close_n[-1]
            bef = close_n[-1 - idx]
            base_col_dict[str(data_length) + "-d-" + str(idx)] = get_divide_arr(bef, aft)


tmp_dt = datetime.fromtimestamp(key_index)
if "hour" in tmp_input:
    base_col_dict["hour"] = tmp_dt.hour
if "min" in tmp_input:
    base_col_dict["min"] = tmp_dt.minute
if "sec" in tmp_input:
    base_col_dict["sec"] = tmp_dt.second
if "week" in tmp_input:
    base_col_dict["week"] = tmp_dt.weekday()

if "weeknum" in tmp_input:
    base_col_dict["weeknum"] = get_weeknum(tmp_dt.weekday(), tmp_dt.day)


print(base_col_dict)

x_df = pd.DataFrame(base_col_dict, index=pd.Index([key_index]))
csv_regist_cols = x_df.columns.tolist()
csv_regist_cols.sort()  # カラムを名前順にするx_df_dixt = x_df.to_dict(orient='index')

tmp_dict = {}
for col in csv_regist_cols:
    if col == "hour" or col == "min" or col == "sec" or col == "week" or col == "weeknum":
        tmp_dict[col] = 'int8'
    else:
        tmp_dict[col] = 'float32'

#型変換
x_df = x_df.astype(tmp_dict)

#訓練時と同じ列名の順番で特徴量を取得する
#pandasデータをxとしてモデルに渡すが、ヘッダーは見ていないため、訓練時と同じ特徴量の順番にする必要がある(tmp_inputを指定することによりそのとおりの順番で抽出してくれる)
x_df = x_df.loc[[key_index], tmp_input]
print(x_df.info())

x_df_dixt = x_df.to_dict(orient='index')
col2 = sorted(x_df_dixt[key_index].items())
print("インプット",col1)
print("インプット",col2)

t1 = time.time()
predict_list = bst.predict(x_df, num_iteration=int(suffix))
print("経過時間：" + str(time.time() - t1))

print("ランダムな予想時スコア:",key_index)
print(predict_list1) #[[0.46517532 0.04326665 0.49155803]]
print(predict_list)