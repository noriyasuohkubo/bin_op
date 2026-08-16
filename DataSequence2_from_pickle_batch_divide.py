import random
from datetime import datetime

import psutil
import time
import redis
import json
from util import *
import logging.config
import pickle
import socket
import send_mail as mail
import subprocess
import gc

host = socket.gethostname()

#DataSequence2_make_pickle.pyで既に作成してあるpickleデータのバッチサイズを小さくする

###設定ここから
symbol = "USDJPY"
db_name_file = "DS2_FILE_NO_" + symbol

#新しいバッチサイズ:現在のバッチサイズを割り切れる数
batch_size_new = int(30720 / 5 )

#テスト用ファイルでない場合のretYのリスト
#テスト用ならば設定必要なし
retY_list = "CATEGORY-d-0.5:CATEGORY-d-0.7:CATEGORY-d-1.0:REGRESSION-d-".split(":")

#テスト用ファイルであるかどうか
test_flg = False

save_score = 307

# win2のDBを参照してナンバリングを行う
r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET NO")
    exit(1)

save_score_new = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)


###設定ここまで

save_dir_path = "/nvme1/dataSequence2/" + symbol + "/DS2F" + str(save_score) + "-0"
save_dir_path_new = "/nvme1/dataSequence2/" + symbol + "/DS2F" + str(save_score_new) + "-0"

r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
# win2のDBを参照してpickleデータの内容を参照する
db_name_file = "DS2_FILE_NO_" + symbol

result = r.zrangebyscore(db_name_file, int(save_score), int(save_score), withscores=True)  # 全件取得
if len(result) == 0:
    print("CANNOT GET DS2_FILE_NO")
    exit(1)

line = result[0]
body = line[0]
tmps = json.loads(body)

steps_per_epoch = int(tmps.get('steps_per_epoch'))
batch_size_old = int(tmps.get('batch_size'))
drop_last = bool(tmps.get('drop_last'))

#旧バッチサイズを新バッチサイズで割り切れる数かチェック
if float(batch_size_old % batch_size_new) != 0.0:
    print("batch_size_new is incorrect:", batch_size_old % batch_size_new)

#ディレクトリ存在チェック
if os.path.exists(save_dir_path) == False:
    print("dir not exists", save_dir_path)
    exit(1)

#新ディレクトリ作成
if os.path.exists(save_dir_path_new) == True:
    print("dir exists", save_dir_path_new)
    exit(1)
else:
    makedirs(save_dir_path_new)

print(datetime.now(), "start")

get_time_list = []
write_time_list = []

#新ファイル作成ループ回数を求める
loop_cnt = int(batch_size_old/batch_size_new)

new_file_cnt = 0

for i in list(range(steps_per_epoch)):

    pickle_path = save_dir_path + "/BF" + str(i)

    start_time = time.perf_counter()

    with open(pickle_path, 'rb') as f:
        if test_flg:
            retX = pickle.load(f)
        else:
            retX,retY = pickle.load(f)

    end_time = time.perf_counter()

    get_time_list.append(end_time - start_time)

    x_col_len = len(retX)#データのリストの長さ

    if i == 0:
        #データのリストの長さが正しいかチェック
        print("X col length:", x_col_len)
        print("batch lenght:", len(retX[0]))

    if (i + 1) == steps_per_epoch and (drop_last == False):
        #最後のループの場合の新ファイル作成ループ回数を求める
        print("last step size:", len(retX[0]))
        loop_cnt = int(len(retX[0]) / batch_size_new)
        if float(len(retX[0]) % batch_size_new) != 0.0:
            #割り切れない場合はプラス1
            loop_cnt += 1

    slice_start = 0
    for j in range(loop_cnt):
        retX_new = []
        for k in range(x_col_len):
            if (j + 1) == loop_cnt:
                retX_new.append(retX[k][slice_start:])
            else:
                slice_end = slice_start + batch_size_new
                retX_new.append(retX[k][slice_start:slice_end])

        if test_flg == False:
            retY_new = {}
            for k, y_col in enumerate(retY_list):
                if (j + 1) == loop_cnt:
                    retY_new[y_col] = retY[y_col][slice_start:]
                else:
                    slice_end = slice_start + batch_size_new
                    retY_new[y_col] = retY[y_col][slice_start:slice_end]

        slice_start += batch_size_new

        save_path = save_dir_path_new + "/" + "BF" + str(new_file_cnt)

        if os.path.isfile(save_path):
            #既に存在するならエラー
            print("file already exists:", save_path)
            exit(1)

        ### pickleで保存
        start_time = time.perf_counter()

        with open(save_path, mode='wb') as f:
            if test_flg:
                pickle.dump(retX_new, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump((retX_new, retY_new), f, protocol=pickle.HIGHEST_PROTOCOL)

        end_time = time.perf_counter()
        write_time_list.append(end_time - start_time)

        new_file_cnt += 1

    if i % 100 == 0:
        print(datetime.now(), i, "get_time_avg:", sum(get_time_list) / len(get_time_list), "write_time_avg:", sum(write_time_list) / len(write_time_list))
        get_time_list = []
        write_time_list = []
    """    
    if i % 1000 == 0:
        print(datetime.now(), "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        gc.collect()
        print(datetime.now(), "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    """

# DBにモデルを登録
tmps['batch_size'] = batch_size_new
tmps['no'] = save_score_new
tmps['steps_per_epoch'] = new_file_cnt
tmps['data_length'] = new_file_cnt

input_name = tmps.get('input_name') + "_ORGDS" + str(save_score)

r.zadd(db_name_file, json.dumps(tmps), save_score_new)



print(datetime.now(), "end")
# 終わったらメールで知らせる
mail.send_message(host, ": DataSequence2_from_pickle_batch_divide finished!!!")