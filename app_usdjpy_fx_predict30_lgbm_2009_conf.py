"""
以下からインポートされる設定ファイル

app_usdjpy_fx_predict30_lgbm_2009_class.py
"""

###共通

SYMBOL = "USDJPY"
BET_TYPE = "CATEGORY"
PREDICT_REQUEST_HOST = "192.168.1.15" #win5
PREDICT_REQUEST_DB_NO = 8
PREDICT_REQUEST_KEY = "USDJPY_PREDICT30_2009"

FX_DATA_MACHINE = "192.168.1.114" #win2
FX_DB_NO = 0
DB_FX_DATA_KEY = "VANTAGE_" + SYMBOL + ".p_S1"

DIV_SEC = 300 #指定秒数前と現在レートのdivを求める為に設定
SUB_SEC = 300 #指定秒数前と現在レートのsubを求める為に設定

AI_MODEL_TERM = 1  #AIモデルの最小データ間隔秒(closeデータの間隔)

LOOP_TERM = 1 #flaskにリクエストする間隔秒

PAST_TERM_SEC = 1 #必要とするlstmモデルの過去分予想の間隔秒
# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 2

MAX_CLOSE_LEN = 903 #渡されるcloseの長さ

#LGBMモデル設定
lgbm_model_file = "MN2009"
lgbm_model_file_suffix = 14

lgbm_model_file_ext = "MN2009"
lgbm_model_file_suffix_ext = 14

INPUT_DATA = '1500-196-DW@1500-196-DW-1@1500-196-DW-2@1500-196-SAME@1500-196-SAME-1@1500-196-SAME-2@1500-196-UP@1500-196-UP-1@1500-196-UP-2@1504-196-DW@1504-196-DW-1@1504-196-DW-2@1504-196-SAME@1504-196-SAME-1@1504-196-SAME-2@1504-196-UP@1504-196-UP-1@1504-196-UP-2@1633-79-DW@1633-79-DW-1@1633-79-DW-2@1633-79-SAME@1633-79-SAME-1@1633-79-SAME-2@1633-79-UP@1633-79-UP-1@1633-79-UP-2@1750-883-DW@1750-883-DW-1@1750-883-DW-2@1750-883-SAME@1750-883-SAME-1@1750-883-SAME-2@1750-883-UP@1750-883-UP-1@1750-883-UP-2@1835-67-DW@1835-67-DW-1@1835-67-DW-2@1835-67-SAME@1835-67-SAME-1@1835-67-SAME-2@1835-67-UP@1835-67-UP-1@1835-67-UP-2'.split("@")

"""
#d1をlgbmの特徴量とする場合
lgbm_ds =[
    {
        "data_length": 2,
        "data_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 501, 5)],
    },
]
"""

#d1をlgbmの特徴量としない場合
lgbm_ds = []

model_dir_lstm = "/app/model/bin_op/"
model_dir_lgbm = "/app/model_lgbm/bin_op/"

"""
Model:1500-196
Model:1504-196
Model:1633-79
Model:1750-883
Model:1835-67
"""
base_models =[
    {
        "name": 'MN1500-196',
        "no": "1500-196",
        "type": "CATEGORY",
        "data_length": [[3, 300],],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM1",
        "db_host": "localhost",
        "db_no":8,
        "db_name": "USDJPY_predict30_1500-196_lstm",
    },
    {
        "name": 'MN1504-196',
        "no": "1504-196",
        "type": "CATEGORY",
        "data_length": [[3, 300], ],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM1",
        "db_host": "localhost",
        "db_no": 8,
        "db_name": "USDJPY_predict30_1504-196_lstm",
    },
    {
        "name": 'MN1633-79',
        "no": "1633-79",
        "type": "CATEGORY",
        "data_length": [[3, 300], ],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM1",
        "db_host": "localhost",
        "db_no": 8,
        "db_name": "USDJPY_predict30_1633-79_lstm",
    },
    {
        "name": 'MN1750-883',
        "no": "1750-883",
        "type": "CATEGORY",
        "data_length": [[3, 200], ],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM1",
        "db_host": "localhost",
        "db_no": 8,
        "db_name": "USDJPY_predict30_1750-883_lstm",
    },
    {
        "name": 'MN1835-67',
        "no": "1835-67",
        "type": "CATEGORY",
        "data_length": [[3, 200], ],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM1",
        "db_host": "localhost",
        "db_no": 8,
        "db_name": "USDJPY_predict30_1835-67_lstm",
    },
]


###test_lgbm_flask_usdjpy_thinkm用
test_file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF381.pickle"
db_name = 'VANTAGE_USDJPY.p_S1'