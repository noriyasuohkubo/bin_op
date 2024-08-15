"""
以下からインポートされる設定ファイル

test_lgbm_flask_usdjpy_thinkm.py
app_usdjpy_fx_thinkm_lgbm.py
test_lgbm_flask_http_usdjpy_thinkm.py
"""

###共通

SYMBOL = "EURUSD"
PREDICT_REQUEST_HOST = "192.168.1.14" #win3
PREDICT_REQUEST_DB_NO = 8
PREDICT_REQUEST_KEY = "EURUSD_PREDICT40"

AI_MODEL_TERM = 2  #AIモデルの最小データ間隔秒(closeデータの間隔)

LOOP_TERM = 2 #flaskにリクエストする間隔秒

PAST_TERM_SEC = 4 #必要とするlstmモデルの過去分予想の間隔秒
# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 3

MAX_CLOSE_LEN = 7350 #渡されるcloseの長さ 最大が300秒データを49個(high lowあるため)なので300*49/2(秒間隔)

#LGBMモデル設定
lgbm_model_file = "MN923"
lgbm_model_file_suffix = 141

lgbm_model_file_ext = "MN923"
lgbm_model_file_suffix_ext = 141

INPUT_DATA = '831-18-DW@831-18-DW-12@831-18-DW-4@831-18-DW-8@831-18-SAME@831-18-SAME-12@831-18-SAME-4@831-18-SAME-8@831-18-UP@831-18-UP-12@831-18-UP-4@831-18-UP-8@839-40-DW@839-40-DW-12@839-40-DW-4@839-40-DW-8@839-40-SAME@839-40-SAME-12@839-40-SAME-4@839-40-SAME-8@839-40-UP@839-40-UP-12@839-40-UP-4@839-40-UP-8@849-5-DW@849-5-DW-12@849-5-DW-4@849-5-DW-8@849-5-SAME@849-5-SAME-12@849-5-SAME-4@849-5-SAME-8@849-5-UP@849-5-UP-12@849-5-UP-4@849-5-UP-8@853-9-DW@853-9-DW-12@853-9-DW-4@853-9-DW-8@853-9-SAME@853-9-SAME-12@853-9-SAME-4@853-9-SAME-8@853-9-UP@853-9-UP-12@853-9-UP-4@853-9-UP-8@hour@min@sec@weeknum'.split("@")

"""
#d1をlgbmの特徴量とする場合
lgbm_ds =[
    {
        "data_length": 2,
        "data_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 151, 10)]
    },
]
"""

#d1をlgbmの特徴量としない場合
lgbm_ds = []

model_dir_lstm = "/app/model/bin_op/"
model_dir_lgbm = "/app/model_lgbm/bin_op/"

base_models =[
    {
        "name": 'MN839-40',
        "no": "839-40",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN831-18',
        "no": "831-18",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN849-5',
        "no": "849-5",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN853-9',
        "no": "853-9",
        "type": "CATEGORY",
        "data_length": [[2, 600], ],
        "input_datas": ["d1", ],
        "input_separate_flg": True,
        "method": "LSTM",
    },
]


###test_lgbm_flask_eurusd_thinkm用
test_file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF17.pickle"
db_name = 'EURUSD_2_0'