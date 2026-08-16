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
from DataSequence2 import DataSequence2

class DataSequence2_copy():
    def __init__(self,dataSequence2):

        self.spread_list = dataSequence2.get_spread_list() #testLstmFX2_rgr_limit用
        self.tick_list = dataSequence2.get_tick_list() #testLstmFX2_rgr_limit用
        self.atr_list = dataSequence2.get_atr_list() #testLstmFX2_rgr_limit用
        self.ind_list = dataSequence2.get_ind_list() #testLstmFX2_rgr_limit用

        self.close_list = dataSequence2.get_close_list()
        self.score_list = dataSequence2.get_score_list()
        self.jpy_list = dataSequence2.get_jpy_list()
        self.output_dict = dataSequence2.get_output_dict()
        self.pred_close_list = dataSequence2.get_pred_close_list()
        self.real_close_list = dataSequence2.get_real_close_list()
        self.target_answer_rate_list = dataSequence2.get_answer_rate_list()
        self.target_answer_score_list = dataSequence2.get_answer_score_list()
        self.target_spread_list = dataSequence2.get_target_spread_list()
        self.target_spread_end_list = dataSequence2.get_target_spread_end_list()
        self.train_score_list = dataSequence2.get_train_score_list()
        self.train_list_idx = dataSequence2.get_train_list_index()
        self.target_highest_close_list = dataSequence2.get_target_highest_close_list()
        self.target_lowest_close_list = dataSequence2.get_target_lowest_close_list()


    #def get_correct_list(self):
    #    return self.correct_list

    def get_pred_close_list(self):
        return self.pred_close_list

    def get_real_close_list(self):
        return self.real_close_list

    def get_score_list(self):
        return self.score_list

    def get_close_list(self):
        return self.close_list

    def get_spread_list(self):
        return self.spread_list

    def get_tick_list(self):
        return self.tick_list

    def get_jpy_list(self):
        return self.jpy_list

    #def get_spread_percent_list(self):
    #    return self.spread_percent_list

    def get_target_spread_list(self):
        return self.target_spread_list

    def get_target_spread_end_list(self):
        return self.target_spread_end_list

    def get_target_divide_prev_list(self):
        return self.target_divide_prev_list

    def get_target_divide_aft_list(self):
        return self.target_divide_aft_list

    def get_train_score_list(self):
        return self.train_score_list

    def get_train_list_index(self):
        return self.train_list_idx

    def get_target_predict_list(self):
        return self.target_predict_list

    def get_answer_rate_list(self):
        return self.target_answer_rate_list

    def get_answer_score_list(self):
        return self.target_answer_score_list

    def get_target_highest_close_list(self):
        return self.target_highest_close_list

    def get_target_lowest_close_list(self):
        return self.target_lowest_close_list

    #def get_hor_list(self):
    #    return self.hor_list

    #def get_atr_list(self):
    #    return self.atr_list

    #def get_ind_list(self):
    #    return self.ind_list

    def get_output_dict(self):
        return self.output_dict

if __name__ == "__main__":
    pickle_path = "/nvme1/dataSequence2/USDJPY/DS2F60-0" + "/DataSequence2_old.pickle"
    print(datetime.now(), "before load ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
    with open(pickle_path, 'rb') as f:
        dataSequence2 = pickle.load(f)
    print(datetime.now(), "after load ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    ds2 = DataSequence2_copy(dataSequence2)
    print(datetime.now(), "after remake ds2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    save_path = "/nvme1/dataSequence2/USDJPY/DS2F60-0" + "/DataSequence2.pickle"
    with open(save_path, mode='wb') as f:
        pickle.dump(ds2, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 終わったらメールで知らせる
    mail.send_message(socket.gethostname(), ": DataSequence2_copy finished!!!")

    print("DataSequence2_copy finished!!!")