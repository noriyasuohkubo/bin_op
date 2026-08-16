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
class DataSequence2_from_pickle_test_raw(Sequence):

    def __init__(self, c, ):

        self.c = c

        self.save_dir_path = self.c.DATA_SEQUENCE_FROM_PICKLE_CONF_TEST_RAW["save_dir_path"]
        # ディレクトリが既に存在しているか確認
        if os.path.exists(self.save_dir_path) == False:
            print("dir not exists", self.save_dir_path)
            exit(1)

        pickle_path = self.save_dir_path + "/DataSequence2_raw.pickle"
        output(datetime.now(), "before load ds2_raw ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        with open(pickle_path, 'rb') as f:
            self.ds2_raw = pickle.load(f)
        output(datetime.now(), "after load ds2_raw ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        self.ds2_raw.add_method()

    def get_ds2(self):
        return self.ds2_raw