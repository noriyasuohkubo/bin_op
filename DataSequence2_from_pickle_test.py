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

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

#DataSequence2_make_pickle.pyで既に作成してあるpickleデータを読み込んで返す
class DataSequence2_from_pickle_test(Sequence):

    def __init__(self, c, ):

        self.c = c
        self.epoch_cnt = 0
        self.get_time_list = []
        score = c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["score"]

        # win2のDBを参照してpickleデータの内容を参照する
        db_name_file = "DS2_FILE_NO_" + c.SYMBOL

        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        result = r.zrangebyscore(db_name_file, int(score), int(score), withscores=True)  # 全件取得
        if len(result) == 0:
            print("CANNOT GET DS2_FILE_NO")
            exit(1)

        line = result[0]
        body = line[0]
        #print(body)
        tmps = json.loads(body)
        self.data_length= int(tmps.get('data_length'))
        self.batch_size = int(tmps.get('batch_size'))
        self.drop_last = bool(tmps.get('drop_last'))
        self.steps_per_epoch = int(tmps.get('steps_per_epoch'))
        self.step_list = list(range(self.steps_per_epoch))

        # dir_pathが合っているか確認
        if (c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["score"] in self.c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["save_dir_path"]) == False:
            print("save_dir_path not correct", self.c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["save_dir_path"])
            exit(1)

        self.save_dir_path = self.c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST["save_dir_path"]
        # ディレクトリが既に存在しているか確認
        if os.path.exists(self.save_dir_path) == False:
            print("dir not exists", self.save_dir_path)
            exit(1)


    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        """
        if idx == 0:
            print("step_list:")
            print(self.step_list[:10])
        """
        start_time = time.perf_counter()

        pickle_path = self.save_dir_path + "/BF" + str(self.step_list[idx])

        with open(pickle_path, 'rb') as f:
            retX = pickle.load(f)

        end_time = time.perf_counter()
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

        if idx % 1000 == 0:
            print(datetime.now(), idx, "get_time_avg:", sum(self.get_time_list) / len(self.get_time_list))
            self.get_time_list = []

        return retX

    def load_ds2(self):
        pickle_path = self.save_dir_path + "/DataSequence2.pickle"
        print(datetime.now(), "before load ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        with open(pickle_path, 'rb') as f:
            self.ds2 = pickle.load(f,)
        print(datetime.now(), "after load ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch

    def get_batch_size(self):
        return self.batch_size

    def get_drop_last(self):
        return self.drop_last
    
    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def get_correct_list(self):
        retY = np.array(self.ds2.get_correct_list)
        return retY

    def get_pred_close_list(self):
        return self.ds2.pred_close_list

    def get_real_close_list(self):
        return self.ds2.real_close_list

    def get_score_list(self):
        return self.ds2.score_list

    def get_close_list(self):
        return self.ds2.close_list

    def get_spread_list(self):
        return self.ds2.spread_list

    def get_tick_list(self):
        return self.ds2.tick_list

    def get_jpy_list(self):
        return self.ds2.jpy_list

    def get_spread_percent_list(self):
        return self.ds2.spread_percent_list

    def get_target_spread_list(self):
        return self.ds2.target_spread_list

    def get_target_spread_end_list(self):
        return self.ds2.target_spread_end_list

    def get_target_divide_prev_list(self):
        return self.ds2.target_divide_prev_list

    def get_target_divide_aft_list(self):
        return self.ds2.target_divide_aft_list

    def get_train_score_list(self):
        return self.ds2.train_score_list

    def get_train_list_index(self):
        return self.ds2.train_list_idx

    def get_target_predict_list(self):
        return self.ds2.target_predict_list

    def get_answer_rate_list(self):
        return self.ds2.target_answer_rate_list

    def get_answer_score_list(self):
        return self.ds2.target_answer_score_list

    def get_hor_list(self):
        return self.ds2.hor_list

    def get_atr_list(self):
        return self.ds2.atr_list

    def get_ind_list(self):
        return self.ds2.ind_list

    def get_output_dict(self):
        return self.ds2.output_dict

    def del_ds2(self):
        del self.ds2
        self.ds2 = None

    def get_target_highest_close_list(self):
        return self.ds2.target_highest_close_list

    def get_target_lowest_close_list(self):
        return self.ds2.target_lowest_close_list