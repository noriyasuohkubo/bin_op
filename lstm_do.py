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

# コンピュータ名を取得
host = socket.gethostname()

conf = conf_class.ConfClass()
print(conf.FILE_PREFIX)


#start = datetime(2004, 1, 1, )
#start = datetime(2010, 1, 1, )
start = datetime(2016, 1, 1, )

#end = datetime(2010, 1, 1)
#end = datetime(2017, 1, 1)
end = datetime(2023, 4, 1)


#学習開始時期がファイル名設定と合致しているかチェック
print(conf.SUFFIX.split("_")[1][:4],str(start.year))
if conf.SUFFIX.split("_")[1][:4] != str(start.year):
    exit(1)

start_eval = datetime(2023, 4, 1, )
#start_eval = datetime(2024, 2, 21, )
end_eval = datetime(2024, 6, 30, )

start_test1 = datetime(2023, 4, 1,  )
#start_test1 = datetime(2024, 2, 21, )
end_test1 = datetime(2024, 6, 30,  )

#start_test2 = datetime(2021, 1, 1, )
#end_test2 = datetime(2022, 9, 1, )

print(datetime.now(), "lstm_do start!!")
conf.change_real_spread_flg(False)
conf.change_fx_real_spread_flg(False)
dataSequence2 = lstm_generator2.make_data(conf, start, end, False, False,)
print(datetime.now(), "dataSequence2 maked!!")


conf.change_real_spread_flg(False)
conf.change_fx_real_spread_flg(False)
dataSequence2_eval = lstm_generator2.make_data(conf, start_eval, end_eval, True, True,)
print(datetime.now(), "dataSequence2_eval maked!!")


#dataSequence2_test1 = testLstm2.make_data(conf, start_test1, end_test1, target_spreads=conf.TARGET_SPREAD_LISTS_TEST, spread_correct=1, sub_force = True)
#print(datetime.now(), "dataSequence2_test1 maked!!")

"""
conf.change_real_spread_flg(False)
conf.change_fx_real_spread_flg(True)
dataSequence2_test2 = testLstmFX2_answer.make_data(conf, start_test1, end_test1, conf.TARGET_SPREAD_LISTS)
print(datetime.now(), "dataSequence2_test2 maked!!")
"""

if host != 'ub3':
    # メモリ節約のためredis停止
    #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
    sudo_password = 'Reikou0129'
    command = 'systemctl stop redis'.split()
    p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]
    # メモリ空き容量を取得
    print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")



#learning_rates = [ 0.001]
learning_rates = [ 0.0005,]
lstm_unit_lists = [[30,30,24],]
dense_lists = [[48,24,12],]
#dense_lists = [[],]
drop_lists = [[0.0,0.0], ]
#lkrates = [ "1-0.0001",  ]
#lkrates = [ "2-0.00001","2-0.00005","2-0.0001", "2-0.0005",  "2-0.001", ]
#lkrates = [ "12-0.0001",]

for learning_rate in learning_rates:
    conf.change_learning_rate(learning_rate)

    for lstm_unit in lstm_unit_lists:
        conf.change_lstm_unit(lstm_unit)

        for dense in dense_lists:
            conf.change_dense_unit(dense)

            for drop in drop_lists:
                conf.change_lstm_do(drop[0])
                conf.change_drop_out(drop[1])

                conf.numbering()#モデル番号付与

                lstm_generator2.do_train(conf, dataSequence2, dataSequence2_eval)
                chk_to_mdl.chk(conf)

                #testLstm2.do_predict(conf, dataSequence2_test1, conf.TARGET_SPREAD_LISTS_TEST, False)

                #testLstmFX2_answer.do_predict(conf, dataSequence2_test2,)

#終わったらメールで知らせる
mail.send_message(host, ": lstm_do finished!!!")


if host != 'ub3':
    sudo_password = 'Reikou0129'
    command = 'systemctl restart redis'.split()
    p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]
