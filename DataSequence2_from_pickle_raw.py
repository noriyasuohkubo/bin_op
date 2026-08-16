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

#学習時のデータを既にDataSequence2_make_pickle.pyで作成してあるDataSequence2で取得する場合の設定
class DataSequence2_from_pickle_raw(Sequence):

    def __init__(self, c, pickle_conf):

        self.c = c
        self.epoch_cnt = 0
        self.get_time_list = []

        self.pickle_confs = pickle_conf

        self.steps_per_epoch = 0
        adj_step = 0

        r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
        # win2のDBを参照してpickleデータの内容を参照する
        db_name_file = "DS2_FILE_NO_" + c.SYMBOL

        c_score = pickle_conf["score"]
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

        #dir_pathが合っているか確認
        if (pickle_conf["score"] in pickle_conf["save_dir_path"]) == False:
            print("save_dir_path not correct", pickle_conf["save_dir_path"])
            exit(1)

        save_dir_path = pickle_conf["save_dir_path"]
        # ディレクトリが既に存在しているか確認
        if os.path.exists(save_dir_path) == False:
            print("dir not exists", save_dir_path)
            exit(1)

        pickle_path = save_dir_path + "/DataSequence2_raw.pickle"
        output(datetime.now(), "before load ds2_raw ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        with open(pickle_path, 'rb') as f:
            self.ds2_raw = pickle.load(f)
        output(datetime.now(), "after load ds2_raw ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        #vectorizeメソッドを復活させる
        self.ds2_raw.add_method()

        #retyを作成するモデルの目的変数に合わせる
        self.ds2_raw.return_all = False
        self.ds2_raw.c.GET_Y_STR = self.c.GET_Y_STR

    def get_ds2(self):
        return self.ds2_raw