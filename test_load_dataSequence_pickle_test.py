import csv
import os
import numpy as np
import random
from datetime import datetime
import subprocess

import psutil
import redis
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import pickle
import time
import socket

import conf_class
from DataSequence2 import DataSequence2


from DataSequence2_from_pickle import DataSequence2_from_pickle
from DataSequence2_from_pickle_test import DataSequence2_from_pickle_test
from DataSequence2_from_pickle_test_raw import DataSequence2_from_pickle_test_raw
from util import *

import warnings

"""
NVMEの読み込み速度確認
"""

os.environ["OMP_NUM_THREADS"] = "1"

# FutureWarningの出力を抑制
# see https://note.nkmk.me/python-warnings-ignore-warning/
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)


if __name__ == "__main__":


    conf = conf_class.ConfClass()
    get_time_list = []

    dataSequence2 = DataSequence2_from_pickle_test(conf, )
    #ds2_c = DataSequence2_from_pickle_test_raw(conf, )
    #dataSequence2 = ds2_c.get_ds2()

    steps_per_epoch = 20000
    #LOAD_EPOCH = conf.LOAD_CONF_DICT["LOAD_EPOCH"]

    # train_listaをエポック分シャッフルする
    #for j in range(LOAD_EPOCH):
    #    train_dataset.rotate_train_list()

    for idx in range(steps_per_epoch):
        start_time = time.perf_counter()

        inputs = dataSequence2.__getitem__(idx)

        end_time = time.perf_counter()

        get_time_list.append(end_time - start_time)
        if idx % 10 == 0:
            print(datetime.now(), idx, "get_time_avg:", sum(get_time_list) / len(get_time_list))

        #time.sleep(0.01)
