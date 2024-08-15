"""
以下からインポートされる設定ファイル

test_lgbm_flask_usdjpy_thinkm.py
app_usdjpy_fx_thinkm_lgbm.py
test_lgbm_flask_http_usdjpy_thinkm.py
"""

###共通

SYMBOL = "USDJPY"
PREDICT_REQUEST_HOST = "192.168.1.14"
PREDICT_REQUEST_DB_NO = 8
PREDICT_REQUEST_KEY = "USDJPY_PREDICT30"

PORT = 8030

AI_MODEL_TERM = 1  #AIモデルの最小データ間隔秒(closeデータの間隔)

LOOP_TERM = 1 #flaskにリクエストする間隔秒

PAST_TERM_SEC = 4 #必要とするlstmモデルの過去分予想の間隔秒
# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 3

MAX_CLOSE_LEN = 7230 #渡されるcloseの長さ

#LGBMモデル設定
lgbm_model_file = "MN915"
lgbm_model_file_suffix = 97

lgbm_model_file_ext = "MN915"
lgbm_model_file_suffix_ext = 97

INPUT_DATA = '908-38-DW@908-38-DW-12@908-38-DW-4@908-38-DW-8@908-38-SAME@908-38-SAME-12@908-38-SAME-4@908-38-SAME-8@908-38-UP@908-38-UP-12@908-38-UP-4@908-38-UP-8@910-20-DW@910-20-DW-12@910-20-DW-4@910-20-DW-8@910-20-SAME@910-20-SAME-12@910-20-SAME-4@910-20-SAME-8@910-20-UP@910-20-UP-12@910-20-UP-4@910-20-UP-8@911-7-DW@911-7-DW-12@911-7-DW-4@911-7-DW-8@911-7-SAME@911-7-SAME-12@911-7-SAME-4@911-7-SAME-8@911-7-UP@911-7-UP-12@911-7-UP-4@911-7-UP-8@912-16-DW@912-16-DW-12@912-16-DW-4@912-16-DW-8@912-16-SAME@912-16-SAME-12@912-16-SAME-4@912-16-SAME-8@912-16-UP@912-16-UP-12@912-16-UP-4@912-16-UP-8@913-16-REG@913-16-REG-12@913-16-REG-4@913-16-REG-8@hour@min@sec@weeknum'.split("@")

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

base_models =[
    {
        "name": 'MN908-38',
        "no": "908-38",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN911-7',
        "no": "911-7",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN910-20',
        "no": "910-20",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN912-16',
        "no": "912-16",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN913-16',
        "no": "913-16",
        "type": "REGRESSION",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },

]


###test_lgbm_flask_usdjpy_thinkm用
test_file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF303.pickle"
db_name = 'USDJPY_1_0'