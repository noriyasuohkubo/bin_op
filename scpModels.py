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
Model:1500-196
Model:1504-196
Model:1633-79
Model:1750-883
Model:1835-67
"""
models =[
    "MN2057-194",

]


#for bin_both
for model in models:
    #cmd = "scp -r " + dirFrom + finename + " reicou@" + machine + ":" + dirTo
    cmd = "cp -r " + dirFrom + model + " " + dirTo
    print(cmd)
    result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
