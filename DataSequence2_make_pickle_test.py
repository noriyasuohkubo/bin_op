import conf_class
import send_mail as mail
import socket
import psutil
from datetime import datetime
import subprocess
from DataSequence2 import DataSequence2
import pickle
from util import *
import redis
import json
import time
from DataSequence2_copy import DataSequence2_copy

"""
DataSequence2のインスタンスを作成し、pickleに保存する
"""

start_time = time.perf_counter()

# コンピュータ名を取得
host = socket.gethostname()

conf = conf_class.ConfClass()
print(conf.FILE_PREFIX)

###設定ここから

#dataSequence2を保存するフラグ
save_ds2_flg = False

#学習用make_predict_db.py用
#conf.DB_EVAL_NO = 3
#conf.FX_TICK_DB = "" #tickは不要なので空文字にしておく
#make_predict_db.py用 ここまで

data = "vantage"

conf.DB_EVAL_NO = 2

if data == "vantage":
    conf.DB_EVAL_NO = 1
elif data == "duka":
    conf.DB_EVAL_NO = 2

print("data:", data)

start = datetime(2024, 12, 1, )
end = datetime(2026, 6, 27)

start_str = start.strftime('%Y%m%d')
end_str = end.strftime('%Y%m%d')

date_str = "_" + start_str + "-" + end_str

batch_size = int(1024 * 20)

# train_list_minimum_flg:train_listに保存する変数を以下に限定する。メモリ節約のため
#    tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp
train_list_minimum_flg = True
TLMF_STR = "_TLMF" if train_list_minimum_flg else ""

dir_cnt = 1

#dir_cntでファイル名を割った時の余りの数字で保存先を指定する
save_dir_path = "/nvme2/dataSequence2/" + conf.SYMBOL

drop_last = False
DL_STR = "_DL" + str(drop_last)

test_flg = True
eval_flg = False

real_spread_flg = True
fx_real_spread_flg = True

redis_stop = False
###設定ここまで

TF_STR = "_TF" if test_flg else ""
EF_STR = "_EF" if eval_flg else ""

RSF_STR = "_RSF" if real_spread_flg else ""
FRSF_STR = "_FRSF" if fx_real_spread_flg else ""

conf.change_real_spread_flg(real_spread_flg)
conf.change_fx_real_spread_flg(fx_real_spread_flg)

print(datetime.now(), "dataSequence2 make start")

dataSequence2 = DataSequence2(conf, start, end, test_flg, eval_flg, batch_size=batch_size, return_all=False, train_list_minimum_flg=train_list_minimum_flg, redis_stop=redis_stop)

print(datetime.now(), "dataSequence2 make end", time.perf_counter() - start_time)

data_length = dataSequence2.__len__()
print("dataSequence2 data_length:", data_length)

steps_per_epoch = dataSequence2.get_steps_per_epoch(batch_size, drop_last=drop_last)

print("steps_per_epoch:", steps_per_epoch)

file_name = conf.FILE_PREFIX + TF_STR + EF_STR + RSF_STR + FRSF_STR + TLMF_STR + "_BS" + str(batch_size) + DL_STR + date_str

# win2のDBを参照してモデルのナンバリングを行う
db_name_file = "DS2_FILE_NO_" + conf.SYMBOL

r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET DS2_FILE_NO")
    exit(1)

newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

"""
for line in result:
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)
    tmp_name = tmps.get("input_name")
    if tmp_name == file_name:
        # 同じファイルがないが確認
        print("The File Already Exists!!!")
        exit(1)
"""

# DBにモデルを登録
child = {
    'input_name': file_name,
    'no': newest_no,
    'data_length': data_length,
    'batch_size': batch_size,
    'drop_last': drop_last,
    'steps_per_epoch': steps_per_epoch,
    'data': data,
    'include_hl': 1 if conf.INCLUDE_HL_FLG else 0
}
r.zadd(db_name_file, json.dumps(child), newest_no)

print("newest_no:", newest_no)

"""
if host != 'ub3' or conf.BET_TERM < 1:
    # メモリ節約のためredis停止
    #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
    sudo_password = 'Reikou0129'
    command = 'systemctl stop redis'.split()
    p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]
    # メモリ空き容量を取得
    print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
"""

#ディレクトリが既に存在していないか確認
tmp_path = save_dir_path + "/DS2F" + str(newest_no) + "-0"
if os.path.exists(tmp_path):
    print("dir already exists", tmp_path)
    exit(1)
else:
    #ディレクトリを作成する
    makedirs(tmp_path)

#バッチごとにpickle保存する
for idx in range(steps_per_epoch):

    print("idx:", idx)
    ret = dataSequence2.__getitem__(idx)

    save_path = save_dir_path + "/DS2F" + str(newest_no) + "-0/" + "BF" + str(idx)

    if os.path.isfile(save_path):
        #既に存在するならエラー
        print("file already exists:", save_path)

    ### pickleで保存
    with open(save_path, mode='wb') as f:
        pickle.dump(ret, f, protocol=pickle.HIGHEST_PROTOCOL)

dataSequence2.delete_method()

if save_ds2_flg:
    save_path = save_dir_path + "/DS2F" + str(newest_no) + "-0/" + "DataSequence2_raw.pickle"
    with open(save_path, mode='wb') as f:
        pickle.dump(dataSequence2, f, protocol=pickle.HIGHEST_PROTOCOL)


#dataSequence2自体をDBデータとtrain_listを一旦削除
dataSequence2.delete_db()
print(datetime.now(), "after delete ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

#testLstmFX2_answer.pyに必要なものだけ保存
ds2 = DataSequence2_copy(dataSequence2)

print(datetime.now(), "after remake ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

save_path = save_dir_path + "/DS2F" + str(newest_no) + "-0/" + "DataSequence2.pickle"
with open(save_path, mode='wb') as f:
    pickle.dump(ds2, f, protocol=pickle.HIGHEST_PROTOCOL)

print("DataSequence2_make_pickle finished!!!", time.perf_counter() - start_time)

#終わったらメールで知らせる
mail.send_message(host, ": DataSequence2_make_pickle_test finished!!!")

