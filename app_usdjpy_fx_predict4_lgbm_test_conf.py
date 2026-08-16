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
PREDICT_REQUEST_KEY = "USDJPY_PREDICT4"

AI_MODEL_TERM = 1  #AIモデルの最小データ間隔秒(closeデータの間隔)

LOOP_TERM = 1 #flaskにリクエストする間隔秒

PAST_TERM_SEC = 4 #必要とするlstmモデルの過去分予想の間隔秒
# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 3

MAX_CLOSE_LEN = 7230 #渡されるcloseの長さ

#LGBMモデル設定
lgbm_model_file = "MN920"
lgbm_model_file_suffix = 3697

lgbm_model_file_ext = "MN920"
lgbm_model_file_suffix_ext = 3697

INPUT_DATA = '714-36-DW@714-36-DW-12@714-36-DW-4@714-36-DW-8@714-36-SAME@714-36-SAME-12@714-36-SAME-4@714-36-SAME-8@714-36-UP@714-36-UP-12@714-36-UP-4@714-36-UP-8@715-40-DW@715-40-DW-12@715-40-DW-4@715-40-DW-8@715-40-SAME@715-40-SAME-12@715-40-SAME-4@715-40-SAME-8@715-40-UP@715-40-UP-12@715-40-UP-4@715-40-UP-8@885-6-REG@885-6-REG-12@885-6-REG-4@885-6-REG-8@887-39-DW@887-39-DW-12@887-39-DW-4@887-39-DW-8@887-39-SAME@887-39-SAME-12@887-39-SAME-4@887-39-SAME-8@887-39-UP@887-39-UP-12@887-39-UP-4@887-39-UP-8'.split("@")

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
        "name": 'MN887-39',
        "no": "887-39",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN885-6',
        "no": "885-6",
        "type": "REGRESSION",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.25_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub1_MN715-40',
        "no": "715-40",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.5_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub3_MN714-36',
        "no": "714-36",
        "type": "CATEGORY",
        "data_length": [[1, 300], [5, 300], [30, 240], ],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
]


###test_lgbm_flask_usdjpy_thinkm用
test_file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF299.pickle"
db_name = 'USDJPY_1_0'