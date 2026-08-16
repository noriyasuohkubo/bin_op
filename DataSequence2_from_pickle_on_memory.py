import gc
import random
from datetime import datetime

import psutil
from tensorflow.keras.utils import Sequence
import time
import redis
import json
from util import *
import logging.config
import pickle
import socket
import send_mail as mail
import sys

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

#DataSequence2_make_pickle.pyで既に作成してあるpickleデータを読み込んで返す
class DataSequence2_from_pickle_on_memory(Sequence):

    def __init__(self, c, pickle_conf, test_flg):

        self.c = c
        self.epoch_cnt = 0
        self.get_time_list = []
        self.test_flg = test_flg

        self.pickle_confs = pickle_conf

        self.steps_per_epoch = 0
        self.memory = {}

        adj_step = 0

        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        # win2のDBを参照してpickleデータの内容を参照する

        for i, tmp_c in enumerate(self.pickle_confs):
            tmp_symbol = tmp_c["save_dir_path"].split("/")[3]
            c_score = tmp_c["score"]
            db_name_file = "DS2_FILE_NO_" + tmp_symbol
            result = r.zrangebyscore(db_name_file, int(c_score), int(c_score), withscores=True)  # 全件取得
            if len(result) == 0:
                print("CANNOT GET DS2_FILE_NO")
                exit(1)

            line = result[0]
            body = line[0]
            tmps = json.loads(body)

            batch_size = int(tmps.get('batch_size'))

            # batch_sizeがあっているか確認
            if batch_size != c.BATCH_SIZE:
                print("BATCH_SIZE incorrect")

            drop_last = bool(tmps.get('drop_last'))

            # drop_lastがあっているか確認
            if drop_last != c.DROP_LAST:
                print("DROP_LAST incorrect")

            steps_per_epoch = int(tmps.get('steps_per_epoch'))

            self.steps_per_epoch += steps_per_epoch
            self.pickle_confs[i]["sum_step"] = self.steps_per_epoch
            self.pickle_confs[i]["adj_step"] = adj_step

            #dir_pathが合っているか確認
            if (tmp_c["score"] in tmp_c["save_dir_path"]) == False:
                print("save_dir_path not correct", tmp_c["save_dir_path"])
                exit(1)

            # ディレクトリ存在確認
            if os.path.exists(tmp_c["save_dir_path"]) == False:
                print("dir not exists", tmp_c["save_dir_path"])
                exit(1)

            #メモリに保存
            print(datetime.now()," pickle memory load start")
            print(datetime.now(), "memory before load", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
            start_time = time.perf_counter()

            #self.pickle_confs[i]["memory"] = {}
            for j in range(steps_per_epoch):
                #self.pickle_confs[i]["memory"][j] = {}

                tmp_pickle_path = tmp_c["save_dir_path"] + "/BF" + str(j)

                with open(tmp_pickle_path, 'rb') as f:
                    retX, retY = pickle.load(f)
                    retY = retY[self.c.GET_Y_STR]
                    self.memory[j + adj_step] = [retX, retY]

                    #self.pickle_confs[i]["memory"][j]["retX"] = retX
                    #self.pickle_confs[i]["memory"][j]["retY"] = retY
                    if j == 0:
                        print("getsizeof retX:", sys.getsizeof(retX))
                        print("getsizeof retY:", sys.getsizeof(retY))

            print(datetime.now(), "memory after load", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
            print("memory load time:", time.perf_counter() - start_time)

            adj_step += steps_per_epoch

        self.step_list = list(range(self.steps_per_epoch))

    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        if idx == 0:
            print("step_list:")
            print(self.step_list[:10])

        start_time = time.perf_counter()
        pickle_path = None
        target_idx = self.step_list[idx]

        retX, retY = self.memory[target_idx]
        """
        try:
            for i, tmp_c in enumerate(self.pickle_confs):
                if target_idx < tmp_c["sum_step"]:

                    tmp_no = target_idx - tmp_c["adj_step"]

                    retX = self.pickle_confs[i]["memory"][tmp_no]["retX"]
                    retY = self.pickle_confs[i]["memory"][tmp_no]["retY"]
                    break
        except Exception as e:
            print("i:",i, "tmp_no:", tmp_no, "target_idx:", target_idx, "adj_step:", tmp_c["adj_step"])
            exit(1)
                
        """

        end_time = time.perf_counter()
        #output(end_time - start_time, idx, self.test_flg, psutil.getloadavg())
        self.get_time_list.append(end_time - start_time)

        if idx % 1000 == 0 and idx != 0:
            if (sum(self.get_time_list) / len(self.get_time_list)) > 0.5:
                print("avg load pickle time over:", str(sum(self.get_time_list) / len(self.get_time_list)))
                mail.send_message(host, " avg load pickle time over:" + str(sum(self.get_time_list) / len(self.get_time_list)))
                raise Exception("avg load pickle time over:" + str(sum(self.get_time_list) / len(self.get_time_list)))
        """
        if idx % 1000 == 0:
            print(datetime.now(), "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
            gc.collect()
            print(datetime.now(), "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        """

        if idx % 100 == 0:
            print(datetime.now(), idx, "get_time_avg:", sum(self.get_time_list) / len(self.get_time_list))
            self.get_time_list = []


        #time.sleep(0.01)
        return retX, retY

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch

    def get_batch_size(self):
        return self.batch_size

    def get_drop_last(self):
        return self.drop_last
    
    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def rotate_train_list(self):
        if self.test_flg == False:
            if self.c.DATA_SHUFFLE == "ROTATE":
                rotate_num = int(len(self.step_list) / self.c.EPOCH)
                self.step_list = rotate(self.step_list, rotate_num)

            elif self.c.DATA_SHUFFLE == "SHUFFLE":
                random.seed(self.c.SEED)
                random.shuffle(self.step_list)

    def on_epoch_end(self):
        self.epoch_cnt += 1

        if self.test_flg == False:
            if self.c.DATA_SHUFFLE == "ROTATE":
                rotate_num = int(len(self.step_list) / self.c.EPOCH)
                self.step_list = rotate(self.step_list, rotate_num)

            elif self.c.DATA_SHUFFLE == "SHUFFLE":
                random.seed(self.c.SEED)
                random.shuffle(self.step_list)

        print("get_time_len:", len(self.get_time_list))
        if len(self.get_time_list) != 0:
            print("get_time_avg:", sum(self.get_time_list)/len(self.get_time_list))
            print("get_time_sum:", sum(self.get_time_list))

        self.get_time_list = []

