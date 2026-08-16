import json

import conf_class
import lstm_generator2
import chk_to_mdl
import testLstmFX2_rgr_sum
import send_mail as mail
import socket
import testLstm2
import testLstmFX2_answer
import psutil
from datetime import datetime
import subprocess
from DataSequence2_from_pickle import DataSequence2_from_pickle
from DataSequence2_from_pickle_raw import DataSequence2_from_pickle_raw
from DataSequence2_from_pickle_test import DataSequence2_from_pickle_test
from DataSequence2_from_pickle_on_memory import DataSequence2_from_pickle_on_memory
from DataSequence2_from_pickle_test_on_memory import DataSequence2_from_pickle_test_on_memory
from util import *
import redis

# コンピュータ名を取得
host = socket.gethostname()

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

conf = conf_class.ConfClass()

if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF) != 0:
    #DataSequence2_make_pickle.pyで既に作成してあるpickleデータを使用する場合
    dataSequence2 = DataSequence2_from_pickle(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF, False)

    dataSequence2_eval = None

elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_ON_MEMORY) != 0:
    dataSequence2 = DataSequence2_from_pickle_on_memory(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF_ON_MEMORY, False)

    dataSequence2_eval = None

elif len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_RAW) != 0:
    #学習時のデータを既にDataSequence2_make_pickle.pyで作成してあるDataSequence2で取得する場合の設定
    ds2_c = DataSequence2_from_pickle_raw(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF_RAW,)
    dataSequence2 = ds2_c.get_ds2()

    dataSequence2_eval = None

else:
    #DataSequence2.pyを使ってDB参照しデータ作成する

    train_start_dt = datetime.strptime(conf.TRAIN_START_DT, '%Y%m%d')
    train_end_dt = datetime.strptime(conf.TRAIN_END_DT, '%Y%m%d')

    output(datetime.now(), "lstm_do start!!")
    conf.change_real_spread_flg(False)
    conf.change_fx_real_spread_flg(False)
    dataSequence2 = lstm_generator2.make_data(conf, train_start_dt, train_end_dt, False, False)
    output(datetime.now(), "dataSequence2 maked!!")
    dataSequence2_eval = None


if len(conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY) != 0:
    dataSequence2 = DataSequence2_from_pickle_on_memory(conf, conf.DATA_SEQUENCE_FROM_PICKLE_CONF_EVAL_ON_MEMORY, False)

"""
if host != 'ub3' or conf.BET_TERM < 1:
    # メモリ節約のためredis停止
    #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
    sudo_password = 'Reikou0129'
    command = 'systemctl stop redis'.split()
    p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]
    # メモリ空き容量を取得
    output("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
"""
if len(conf.LOAD_CONF_DICT) == 0:
    conf.numbering()#モデル番号付与

fragments_list = []

if conf.FRAGMENT_NUM != None:
    # 参照するレートのインデックスを指定する場合
    r = redis.Redis(host="win2", port=6379, db=1, decode_responses=True)
    result = r.zrangebyscore("FRAGMENTS", conf.FRAGMENT_NUM, conf.FRAGMENT_NUM, withscores=True)
    for j, line in enumerate(result):
        body = line[0]
        tmps = json.loads(body)
        list_str = tmps.get("list_str")

        for k in list_str.split(","):
            fragments_list.append(int(k))

output("fragments:",fragments_list)

lstm_generator2.do_train(conf, dataSequence2, dataSequence2_eval)
#chk_to_mdl.chk(conf)

#testLstm2.do_predict(conf, dataSequence2_test1, conf.TARGET_SPREAD_LISTS_TEST, False)

#testLstmFX2_answer.do_predict(conf, dataSequence2_test2,)

#終わったらメールで知らせる
mail.send_message(host, ": lstm_do finished!!!")

"""
if host != 'ub3' or conf.BET_TERM < 1:
    sudo_password = 'Reikou0129'
    command = 'systemctl restart redis'.split()
    p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]
"""