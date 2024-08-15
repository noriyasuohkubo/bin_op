"""
以下からインポートされる設定ファイル

test_lgbm_flask_usdjpy_thinkm.py
app_usdjpy_fx_thinkm_lgbm.py
test_lgbm_flask_http_usdjpy_thinkm.py
"""

###共通

SYMBOL = "USDJPY"
AI_MODEL_TERM = 2  #AIモデルの最小データ間隔秒(closeデータの間隔)

LOOP_TERM = 4 #flaskにリクエストする間隔秒

PAST_TERM_SEC = 4 #必要とするlstmモデルの過去分予想の間隔秒
# 必要とするlstmモデルの過去分予想の数
# 現在の予想のみ使用する場合は0にする
PAST_LENGTH = 1

MAX_CLOSE_LEN = 7350 #渡されるcloseの長さ 最大が300秒データを49個(high lowあるため)なので300*49/2(秒間隔)

#LGBMモデル設定
lgbm_model_file = "MN808"
lgbm_model_file_suffix = 352

lgbm_model_file_ext = "MN808"
lgbm_model_file_suffix_ext = 352

INPUT_DATA = '2-d-1@2-d-10@2-d-100@2-d-110@2-d-120@2-d-130@2-d-140@2-d-150@2-d-160@2-d-170@2-d-180@2-d-190@2-d-2@2-d-20@2-d-200@2-d-210@2-d-220@2-d-230@2-d-240@2-d-250@2-d-260@2-d-270@2-d-280@2-d-290@2-d-3@2-d-30@2-d-300@2-d-310@2-d-320@2-d-330@2-d-340@2-d-350@2-d-360@2-d-370@2-d-380@2-d-390@2-d-4@2-d-40@2-d-400@2-d-410@2-d-420@2-d-430@2-d-440@2-d-450@2-d-460@2-d-470@2-d-480@2-d-490@2-d-5@2-d-50@2-d-500@2-d-6@2-d-60@2-d-7@2-d-70@2-d-8@2-d-80@2-d-9@2-d-90@744-9-DW@744-9-DW-4@744-9-SAME@744-9-SAME-4@744-9-UP@744-9-UP-4@771-33-DW@771-33-DW-4@771-33-SAME@771-33-SAME-4@771-33-UP@771-33-UP-4@773-9-DW@773-9-DW-4@773-9-SAME@773-9-SAME-4@773-9-UP@773-9-UP-4@774-32-DW@774-32-DW-4@774-32-SAME@774-32-SAME-4@774-32-UP@774-32-UP-4@798-15-DW@798-15-DW-4@798-15-SAME@798-15-SAME-4@798-15-UP@798-15-UP-4@hour@min@sec@weeknum'.split("@")

#d1をlgbmの特徴量とする場合
lgbm_ds =[
    {
        "data_length": 2,
        "data_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, ] + [i for i in range(10, 501, 5)],
    },
]

#d1をlgbmの特徴量としない場合
#lgbm_ds = []

model_dir_lstm = "/app/model/bin_op/"
model_dir_lgbm = "/app/model_lgbm/bin_op/"

base_models =[
    {
        "name":'MN771-33',
        "no":"771-33",
        "type":"CATEGORY",
        "data_length":[[2,600],],
        "input_datas":["d1",  ],
        "input_separate_flg":False,
        "method":"LSTM",
    },
    {
        "name": 'MN773-9',
        "no": "773-9",
        "type": "CATEGORY",
        "data_length": [[10, 600], ],
        "input_datas": ["d1_ehd1-1_eld1-1", ],
        "input_separate_flg": False,
        "method": "LSTM",
    },
    {
        "name": 'MN774-32',
        "no": "774-32",
        "type": "CATEGORY",
        "data_length": [[60, 240], ],
        "input_datas": ["d1_ehd1-1_eld1-1", ],
        "input_separate_flg": False,
        "method": "LSTM",
    },

    {
        "name": 'USDJPY_LT1_M7_LSTM1_B2_BS2_T40_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_234-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub2_MN744-9',
        "no": "744-9",
        "type": "CATEGORY",
        "data_length":[[2,300],[10,300],[60,240],[300,48],],
        "input_datas":["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1", ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
    {
        "name": 'MN798-15',
        "no": "798-15",
        "type": "CATEGORY",
        "data_length": [[2, 300], [10, 300], [60, 240],],
        "input_datas": ["d1", "d1_ehd1-1_eld1-1", "d1_ehd1-1_eld1-1",  ],
        "input_separate_flg": True,
        "method": "LSTM7",
    },
]


###test_lgbm_flask_usdjpy_thinkm用
test_file_path = "/db2/lgbm/" + SYMBOL + "/test_file/TESF266.pickle"
db_name = 'USDJPY_2_0'