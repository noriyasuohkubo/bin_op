# coding: utf-8
import subprocess
import send_mail as m
from datetime import datetime
from datetime import date
import time
import send_mail as m
import os
import logging.config


"""
作成したモデルを各マシンへSCPする
"""


machines = ["localhost" ]

dirFrom = "/app/model/bin_op/"
#dirFrom = "/app/model_lgbm/bin_op/"
dirTo = "/app/model/tmp/"

"""
model_suffix = {
    1:[
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-31",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-20"
    ] ,
    2: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-21",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37"
    ] ,
    3: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-16",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD2_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-17"
    ] ,
    4: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-6",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37"
    ],
    5: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-27",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD4_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-37"
    ],
    6: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD5_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-19",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-12"
    ],
    7: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD3_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-17",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD6_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-12"
    ],
    8: [
        "GBPJPY_CATEGORY_BIN_UP_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD7_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-25",
        "GBPJPY_CATEGORY_BIN_DW_LSTM7_BET2_TERM30_INPUT2-10-30-90-300_INPUT_LEN300-300-240-80-24_L-UNIT30-30-24-8-4_D-UNIT16-8-4_DROP0.0_L-K0_L-R0_DIVIDEMAX0_SPREAD1_UB1_202101_90_L-RATE0.001_LOSS-C-ENTROPY_GU-GU-GU-1"
    ],
    }
"""
models =[
    "MN908-38",
    'MN911-7',
    'MN910-20',
    'MN912-16',
    'MN913-16',



]



"""
model_suffix = {
    1:[
        "USDJPY_LT4_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN196-38",
    ] ,
    }
"""
"""
#for bin_both_four
for spr in model_suffix:
    for group in model_suffix[spr]:
        for finename in group:
            for machine in machines:
                cmd = "scp -r " + dir + finename + " reicou@" + machine + ":" + dir
                print(cmd)
                result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
"""

"""
#現在から1日前までのファイルを抽出してコピー
def res_cmd_lfeed(cmd):
  return subprocess.Popen(
      cmd, stdout=subprocess.PIPE,
      shell=True).stdout.readlines()

cmd = "find " + dirFrom +" -maxdepth 1 -mtime -2 | grep USDJPY"
results = res_cmd_lfeed(cmd)

#print(results)
#for result in results:
#    print("a",result.decode().replace('\n', ''))
for machine in machines:
    for result in results:
        cmd = "scp -r " + result.decode().replace('\n', '') + " reicou@" + machine + ":" + dirTo
        #cmd = "cp -r " + dirFrom + finename + " " + dirTo
        print(cmd)
        result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()

"""
"""
#for bin_both
for spr in model_suffix:
    for finename in model_suffix[spr]:
        for machine in machines:

            #cmd = "scp -r " + dirFrom + finename + " reicou@" + machine + ":" + dirTo
            cmd = "cp -r " + dirFrom + finename + " " + dirTo
            print(cmd)
            result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
"""
#for bin_both
for model in models:
    #cmd = "scp -r " + dirFrom + finename + " reicou@" + machine + ":" + dirTo
    cmd = "cp -r " + dirFrom + model + " " + dirTo
    print(cmd)
    result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
