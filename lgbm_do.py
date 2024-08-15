import conf_class
import lstm_generator2
import chk_to_mdl
import testLstmFX2_rgr_sum
import send_mail as mail
import socket
import testLstm2
import psutil
import pandas as pd
[]
# コンピュータ名を取得
host = socket.gethostname()

conf = conf_class.ConfClass()
print(conf.FILE_PREFIX_DB)
conf.change_real_spread_flg(False)

dataSequence2, dataSequence2_eval = lstm_generator2.make_data(conf)

conf.change_real_spread_flg(False)
dataSequence2_test = testLstm2.make_data(conf, [])

"""
# メモリ節約のためredis停止
#r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
sudo_password = 'Reikou0129'
command = 'systemctl stop redis'.split()
p = psutil.Popen(['sudo', '-S'] + command, stdin=psutil.PIPE, stderr=psutil.PIPE,universal_newlines=True)
sudo_prompt = p.communicate(sudo_password + '\n')[1]
# メモリ空き容量を取得
print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
"""

learning_rates = [ 0.001,0.002,0.004,0.006, 0.008, ]

for lr in learning_rates:

    conf.change_learning_rate(lr)

    lstm_generator2.do_train(conf, dataSequence2, dataSequence2_eval)
    chk_to_mdl.chk(conf)

    testLstm2.do_predict(conf, dataSequence2_test, [])
    #testLstmFX2_rgr_sum.do_predict(conf)

#終わったらメールで知らせる
mail.send_message(host, ": lstm_do finished!!!")

