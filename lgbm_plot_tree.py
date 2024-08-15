import pickle
import time

from dtreeviz import dtreeviz
from matplotlib import pyplot as plt
from datetime import datetime
import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import conf_class_lgbm
import numpy as np
import socket
from util import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import send_mail as mail
from testLstmFX2_answer import get_result_rgr_both, showDetail,showProfitInd,showProfitTime,showProfitIndUpDown,showPipsPerSpread
from lgbm_make_data import LgbmMakeData
from sklearn import tree

import graphviz
import dtreeviz

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

conf = conf_class_lgbm.ConfClassLgbm()
file = "MN266"
bst = lgb.Booster(model_file=conf.MODEL_DIR + file,)

"""
df = bst.trees_to_dataframe()
for index,item in df.iterrows():
    print("index : ", index)
    print("item  :\n", item)
    print("----------------\n")
"""


lgb.plot_tree(booster=bst,
              tree_index=0,
              show_info='internal_value',
              )
plt.show()



