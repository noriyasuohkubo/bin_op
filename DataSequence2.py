import random
from tensorflow.keras.utils import Sequence
from datetime import datetime
import time
import redis
import json
import psutil
from util import *
import logging.config
import pandas as pd
import socket
import subprocess
import send_mail as mail
import gc
from important_index import *

host = socket.gethostname()
output_log_name = "/home/reicou/ds2_" + host + ".txt"
output = output_log(output_log_name)

class DataSequence2(Sequence):

    #train_list_minimum_flg:train_listに保存する変数を以下に限定する。メモリ節約のため
    #    tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp

    def __init__(self, c, startDt, endDt, test_flg, eval_flg, target_spread_list=[], spread_correct=None, target_spread_percent_list=[], sub_force = False, batch_size=None,
                 drop_last=False, return_all=False, train_list_minimum_flg=False, redis_stop=False):
        #tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp,
        try:
            self.return_all = return_all
            self.get_time_list = []
            self.c = c
            self.sub_force = sub_force #答えを算出にsubを使用する場合(学習やeval時にはdiv)

            self.batch_size = batch_size if batch_size != None else c.BATCH_SIZE
            self.drop_last = drop_last

            if c.EXCEPT_IMPORTANT_INDEX_RANGE != None:
                importantAnswer = ImportantIndex(importance=c.EXCEPT_IMPORTANT_INDEX_IMPORTANCE, range=c.EXCEPT_IMPORTANT_INDEX_RANGE, startDt=startDt, endDt=endDt)

            TARGET_SPREAD_LIST = target_spread_list

            output("TARGET_SPREAD_LIST:",TARGET_SPREAD_LIST)
            output("TARGET_SPREAD_PERCENT_LIST:",target_spread_percent_list)

            SPREAD_CORRECT = c.SPREAD
            if spread_correct != None:
                SPREAD_CORRECT = spread_correct
            output("SPREAD_CORRECT:",SPREAD_CORRECT)

            self.opt_lists = c.OUTPUT_DATA.split("_")

            #コンストラクタ
            self.epoch_cnt = 0
            self.db_fake = self.make_input_data_list(c.INPUT_DATAS[0])
            self.db_fake_score = {}
            self.db_fake_score_list = [] #単純にscoreのリスト
            self.db1 = self.make_input_data_list(c.INPUT_DATAS[0]) #divide値などを保持
            self.db1_score = {} #scoreとdb1リストのインデックスを保持
            self.db1_score_list = [] #データ確認用
            self.db2 = self.make_input_data_list(c.INPUT_DATAS[1]) if len(self.c.INPUT_LEN) >=2 else None
            self.db2_score = {}
            self.db2_score_list = [] #単純にscoreのリスト
            self.db3 = self.make_input_data_list(c.INPUT_DATAS[2]) if len(self.c.INPUT_LEN) >=3 else None
            self.db3_score = {}
            self.db3_score_list = []
            self.db4 = self.make_input_data_list(c.INPUT_DATAS[3]) if len(self.c.INPUT_LEN) >=4 else None
            self.db4_score = {}
            self.db4_score_list = []
            self.db5 = self.make_input_data_list(c.INPUT_DATAS[4]) if len(self.c.INPUT_LEN) >=5 else None
            self.db5_score = {}
            self.db5_score_list = []


            self.db_extra_1 = [] #1秒データを格納
            self.db_extra_1_score = {}

            #分足用
            self.db_foots = {}
            self.db_foots_score = {}

            #LSTM8用
            self.db_volume = []
            self.db_volume_score = {}
            self.db_volume_score_list = []

            # LSTM9用
            self.db_close = []
            self.db_pred = []
            self.db_preds = {}
            if c.METHOD == "LSTM9":
                for dbname in c.LSTM9_PRED_DBS:
                    self.db_preds[dbname] = []

            self.db9_name = c.LSTM9_PRED_DB_DEF

            #OPTION用 scoreをキーに、そのscoreで使用するインデックスを入れていく
            self.option_score = {}

            self.data_checked = False

            # 学習対象のみの各DBのインデックスと,DB内のインデックスおよび正解ラベルが入った子リストを保持する
            # このインデックスを元に配列から指定期間分のデータと正解ラベルを取得する
            # ex:
            # [
            # [ [0,1,0], [100], [101], [45] ], これで学習データ1つ分 左から正解ラベル, DB1内のインデックス, DB2内のインデックス, DB3内のインデックス
            # [ [1,0,0], [10], [6], [200] ],
            # ]
            self.start_score = int(time.mktime(startDt.timetuple()))
            self.end_score = int(time.mktime(endDt.timetuple())) -1
            self.test_flg = test_flg
            self.eval_flg = eval_flg
            self.time_list = []
            #testの場合のみ正解ラベルをリターンする
            self.correct_list = [] #test用 正解ラベルを保持
            self.train_list = []
            self.train_dict = {} #test用
            self.train_dict_ex = {}  # test用 score_listと長さを合わせる
            self.pred_close_list = [] #test用 予想時のレート保持
            self.real_close_list = [] #test用 正解レートを保持
            self.score_list = [] #test用
            self.score_dict = {} #test用
            self.close_list = [] #test用
            self.spread_list = [] #test用
            self.tick_list = [] #FX test用
            self.jpy_list = [] #FX test用
            self.spread_percent_list = [] #FX BITCOIN test用 spreadの％表記

            self.output_dict = {}
            self.output_answer_dict = {}
            for tmp_k in self.opt_lists:
                self.output_dict[tmp_k] = []
                self.output_answer_dict[tmp_k] = []

            self.train_score_list = [] #予想対象のスコアを保持
            self.train_list_idx = [] #test用
            self.spread_cnt_dict = {} #test用 スプレッド毎の件数を保持
            self.spread_cnt = 0
            self.target_spread_list = [] #test用
            self.target_spread_end_list = [] #test用
            self.target_divide_prev_list = [] #test用
            self.target_divide_aft_list = []  # test用
            self.target_predict_list = [] #test用
            self.target_answer_rate_list = [] #test用
            self.target_answer_score_list = [] #test用
            self.target_highest_close_list = [] #test用
            self.target_lowest_close_list = [] #test用

            self.atr_list = []
            self.atr_dict = {}

            self.hor_list = []
            self.hor_dict = []
            if len(c.HOR_DB_CORE_LIST) != 0:
                for i in c.HOR_DB_CORE_LIST:
                    self.hor_dict.append({})

            self.hl_dict = {}

            self.highlow_dict = {}

            self.oanda_ord_dict = {}
            self.oanda_pos_dict = {}

            self.ind_list = []
            self.ind_score_dict = {}

            self.answer_dict = {}

            self.tick_dict = {} #keyをscore,valueにそのスコアから始まるtick情報をいれる

            self.category_ocops_cnt = 0
            self.category_ocops_up_cnt = 0
            self.category_ocops_dw_cnt = 0

            self.same_db_flg = True #すべて同じ足の長さのDBを使うかどうか 例)すべてGBPJPY_2_0のDBをつかう

            self.db_no = c.DB_NO
            self.real_spread_flg = c.REAL_SPREAD_FLG

            self.ind_foot_dict = {}

            if "OCO" in c.ANSWER_DB:
                self.tmp_tp, self.tmp_sl = c.ANSWER_DB_TYPE.split(":")[1].split("-") if c.ANSWER_DB_TYPE != "" else [None, None]
                output("tmp_tp", self.tmp_tp, "tmp_sl", self.tmp_sl)
            self.ocoa_skip_cnt = 0

            if self.test_flg:
                self.db_no = c.DB_EVAL_NO

            if self.eval_flg:
                self.db_no = c.DB_EVAL_NO
                self.real_spread_flg = c.REAL_SPREAD_EVAL_FLG

            self.fragments_list = []

            if c.FRAGMENT_NUM != None:
                #参照するレートのインデックスを指定する場合
                r = redis.Redis(host="win2", port=6379, db=1, decode_responses=True)
                result = r.zrangebyscore("FRAGMENTS", c.FRAGMENT_NUM, c.FRAGMENT_NUM, withscores=True)
                for j, line in enumerate(result):
                    body = line[0]
                    tmps = json.loads(body)
                    list_str = tmps.get("list_str")

                    for k in list_str.split(","):
                        self.fragments_list.append(int(k))

                if len(self.fragments_list) != c.FRAGMENTS_INPUT_LEN +1:
                    #fragments_listの長さはdivやsubを取るためにinput_lenより1多いはず
                    print("FRAGMENTS_INPUT_LENの長さが不正")
                    exit(1)

            self.additional_data_list = []
            for i, a in enumerate(c.ADDITIONAL_DATA_LIST):
                input_type =a['input_type']

                r = redis.Redis(host="win2", port=6379, db=1, decode_responses=True)
                result = r.zrangebyscore("FRAGMENTS", a["fragment_num"], a["fragment_num"], withscores=True)
                for j, line in enumerate(result):
                    body = line[0]
                    tmps = json.loads(body)
                    list_str = tmps.get("list_str")
                    tmp_list = []
                    for k in list_str.split(","):
                        if input_type == "sma":
                            tmp_list.append(k)
                        else:
                            tmp_list.append(int(k))
                    c.ADDITIONAL_DATA_LIST[i]["fragments_list"] = tmp_list

                if len(c.ADDITIONAL_DATA_LIST[i]["fragments_list"]) != a["input_len"]:
                    print("ADDITIONAL_DATAのINPUT_LENの長さが不正",len(c.ADDITIONAL_DATA_LIST[i]["fragments_list"]) , a["input_len"])
                    exit(1)

                db_name = c.SYMBOL + "_" + a['length']

                r = redis.Redis(host=c.DB_HOST, port=6379, db=self.db_no, decode_responses=True)
                result = r.zrangebyscore(db_name, self.start_score - (3600 * 24 * 30), self.end_score, withscores=True)
                if input_type == "c":
                    data_list = []
                    data_score_dict = {}
                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)
                        data_list.append(float(tmps.get(input_type)))
                        data_score_dict[score] = j

                    self.additional_data_list.append(
                        {
                            "data_list":np.array(data_list),
                            "score_dict":data_score_dict,
                        }
                    )
                elif input_type == "sma":
                    data_dict = {}
                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)
                        data_dict[score] = tmps

                    self.additional_data_list.append(
                        {
                            "data_dict": data_dict,
                        }
                    )
            #すべて同じ足の長さのDBを使うかどうか判定
            for i, db in enumerate(c.INPUT_LEN):
                if i != len(c.INPUT_LEN) -1:
                    if c.DB_TERMS[i] != c.DB_TERMS[i + 1]:
                        self.same_db_flg = False

            output("same_db_flg:", self.same_db_flg)

            r = redis.Redis(host=c.DB_HOST, port=6379, db=self.db_no, decode_responses=True)

            if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False and c.FX_TICK_DB != c.DB1_LIST[0]:
                result = r.zrangebyscore(c.FX_TICK_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)

                for j, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.tick_dict[score] = tmps.get("tk")
                del result

            # Fake DB
            db_fake_index = 0
            if c.DB_FAKE_TERM != 0:
                # メモリ空き容量を取得
                output("before fake db ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                for i, db in enumerate(c.DB_FAKE_LIST):
                    #1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)
                        if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                            self.db_fake["c"].append(tmps.get("c"))
                        else:
                            self.db_fake["d"].append(tmps.get("d"))
                        self.db_fake_score[score] = db_fake_index
                        self.db_fake_score_list.append(score)

                        db_fake_index += 1

                    del result

            if c.METHOD == "LSTM8":
                db_volume_index = 0
                if c.DB_VOLUME_TERM != 0:
                    # メモリ空き容量を取得
                    output("before volume db ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                    for i, db in enumerate(c.DB_VOLUME_LIST):

                        # 1日余分に読み込む
                        result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                        #output(db, len(result))
                        for j, line in enumerate(result):
                            body = line[0]
                            score = float(line[1])
                            tmps = json.loads(body)

                            self.db_volume.append(tmps.get("v"))
                            self.db_volume_score[score] = db_volume_index
                            self.db_volume_score_list.append(score)

                            db_volume_index += 1

                        del result

            #分足用データ取得
            for db_tmp in c.FOOT_DBS:
                d_term, d_len, d_unit, d_x, db_name, separate_flg = db_tmp
                ipt_list_foot = d_x.split("_")
                db_foot_idx = 0
                #endscoreより7日余分に読み込む
                result = r.zrangebyscore(db_name, self.start_score - 3600 * 24 * 7, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(db_name) #メモリ節約のため参照したDBは削除する
                #tmp_list = []
                tmp_dict = {}
                tmp_x_dict = self.make_input_data_list(d_x)
                for j, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)

                    #特徴力にnanがないかチェック
                    ok_flg = True
                    for ipt_foot in ipt_list_foot:
                        t_input = tmps.get(ipt_foot)
                        if t_input == None or np.isnan(t_input):
                            ok_flg = False
                            output("FOOT_DB non data", db_tmp, ipt_foot, score)
                            break
                    if ok_flg == False:
                        continue
                    else:
                        for ipt_foot in ipt_list_foot:
                            tmp_x_dict[ipt_foot].append(tmps.get(ipt_foot))

                    tmp_dict[score] = db_foot_idx

                    db_foot_idx += 1
                self.db_foots[d_term] = self.make_input_data_list_np(tmp_x_dict, d_x)
                self.db_foots_score[d_term] = tmp_dict

                del result

            if c.DB_EXTRA_1 != "":
                # メモリ空き容量を取得
                output("before db extra 1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                db_extra_1_index = 0
                #endscoreより1日余分に読み込む
                result = r.zrangebyscore(c.DB_EXTRA_1, self.start_score - 3600 * 24, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(c.DB_EXTRA_1) #メモリ節約のため参照したDBは削除する

                for j, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.db_extra_1.append(tmps.get("d"))
                    self.db_extra_1_score[score] = db_extra_1_index

                    db_extra_1_index += 1

                del result

            if len(c.OPTIONS) != 0:
                # メモリ空き容量を取得
                output("before db opt ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                #endscoreより1日余分に読み込む
                result = r.zrangebyscore(c.OPTIONS_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)

                    try:
                        start_score = result[l - c.OPTIONS_NEED_LEN][1]
                        end_score = score

                        if end_score != get_decimal_add(start_score, get_decimal_multi(c.OPTIONS_NEED_LEN, c.BET_TERM)):
                            # 時刻がつながっていないものはNoneにしてあとで学習対象外としてはじく
                            self.option_score[score] = None
                            continue
                    except Exception:
                        # start_score end_scoreのデータなしの場合Noneにしてあとで学習対象外としてはじく
                        self.option_score[score] = None
                        continue

                    tmp_list = []
                    for op in c.OPTIONS:
                        tmp_list.append(tmps.get(op))

                    self.option_score[score] = tmp_list

                del result
                #if test_flg == False:
                #    r.delete(db) #メモリ節約のため参照したDBは削除する

            for j, hor in enumerate(c.HOR_DB_CORE_LIST):
                hor_db_name = c.SYMBOL + "_" + hor + "_HOR"

                # 60日余分に読み込む
                result = r.zrangebyscore(hor_db_name, self.start_score - 3600 * 24 * 60, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(hor_db_name)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.hor_dict[j][score] = tmps.get("data")

                del result

            if (c.INCLUDE_HL_FLG and test_flg and eval_flg == False) or c.LEARNING_TYPE_STOPLOSS:
                hl_db_name = c.SYMBOL + "_" + str(c.BET_TERM) + "_HL-" + str(int(c.TERM))
                result_hl = r.zrangebyscore(hl_db_name, self.start_score, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(hl_db_name)  # メモリ節約のため参照したDBは削除する

                for res in result_hl:
                    body = res[0]
                    score = float(res[1])
                    tmps = json.loads(body)

                    self.hl_dict[score] = {
                        "h": float(tmps.get("h")),
                        "l": float(tmps.get("l")),
                    }

            if c.HIGHLOW_DB_CORE != "":
                # 2日余分に読み込む
                result = r.zrangebyscore(c.HIGHLOW_DB, self.start_score - 3600 * 24 * 2, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(c.HIGHLOW_DB)  # メモリ節約のため参照したDBは削除する

                for res in result:
                    body = res[0]
                    score = float(res[1])
                    tmps = json.loads(body)

                    start_v = 1
                    while True:
                        if start_v > c.HIGHLOW_DATA_NUM:
                            break

                        data_length = str(int(get_decimal_multi(c.HIGHLOW_TERM, start_v)))
                        if start_v == 1:
                            self.highlow_dict[score] = {
                                data_length + "_h": tmps.get(data_length + "_h"),
                                data_length + "_l": tmps.get(data_length + "_l"),
                            }
                        else:
                            self.highlow_dict[score][data_length + "_h"] = tmps.get(data_length + "_h")
                            self.highlow_dict[score][data_length + "_l"] = tmps.get(data_length + "_l")
                        start_v += 1
                output("highlow dict lungth:", len(self.highlow_dict))
                del result

            if c.OANDA_ORD_DB != "":
                # 1日余分に読み込む
                result = r.zrangebyscore(c.OANDA_ORD_DB, self.start_score - 3600 * 24 * 1, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(c.OANDA_ORD_DB)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.oanda_ord_dict[score] = [tmps.get("wid"), tmps.get("data")]

                del result

            if c.OANDA_POS_DB != "":
                # 1日余分に読み込む
                result = r.zrangebyscore(c.OANDA_POS_DB, self.start_score - 3600 * 24 * 1, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(c.OANDA_POS_DB)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.oanda_pos_dict[score] = [tmps.get("wid"), tmps.get("data")]

                del result

            if len(c.IND_FOOT_COL) != 0 :
                # 1日余分に読み込む
                result = r.zrangebyscore(c.IND_FOOT_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(c.IND_FOOT_DB)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)

                    tmp_dict = {}
                    for col in c.IND_FOOT_COL:
                        tmp_dict[col] = tmps.get(col)
                    self.ind_foot_dict[score] = tmp_dict

                del result

            #ATRを特徴量としてでなく学習対象の絞り込みなど他の用途で使用する場合
            if c.ATR_COL != "":
                #endscoreより1日余分に読み込む
                result = r.zrangebyscore(c.OPTIONS_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)
                #if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                #    r.delete(c.OPTIONS_DB)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    try:
                        start_score = result[l - c.ATR_NEED_LEN][1]
                        end_score = score
                        if end_score != get_decimal_add(start_score, get_decimal_multi(c.ATR_NEED_LEN, c.BET_TERM)):
                            # 時刻がつながっていないものはNoneにしてあとで学習対象外としてはじく
                            self.atr_dict[score] = None
                            continue
                    except Exception:
                        # start_score end_scoreのデータなしの場合Noneにしてあとで学習対象外としてはじく
                        self.atr_dict[score] = None
                        continue

                    self.atr_dict[score] = tmps.get(c.ATR_COL)

                del result

            if len(c.IND_COLS) != 0:
                #endscoreより1日余分に読み込む
                result = r.zrangebyscore(c.IND_COLS_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)
                #if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                #    r.delete(c.IND_COLS_DB)  # メモリ節約のため参照したDBは削除する

                for l, line in enumerate(result):
                    body = line[0]
                    score = float(line[1])
                    tmps = json.loads(body)
                    self.ind_score_dict[score] = []
                    for k,col in enumerate(c.IND_COLS):
                        try:
                            tmp_need_len = c.IND_NEED_LENS[k]
                            start_score = result[l - tmp_need_len][1]
                            end_score = score

                            if end_score != get_decimal_add(start_score, get_decimal_multi(tmp_need_len, c.BET_TERM)):
                                # 時刻がつながっていないものはNoneにしてあとで学習対象外としてはじく
                                self.ind_score_dict[score].append(None)
                                continue
                        except Exception:
                            # start_score end_scoreのデータなしの場合Noneにしてあとで学習対象外としてはじく
                            self.ind_score_dict[score].append(None)
                            continue

                        self.ind_score_dict[score].append(tmps.get(col))

                del result

            if c.ANSWER_DB != "":
                tmp_df = pd.read_pickle(c.ANSWER_DB_FILE)

                start_sc = self.start_score
                end_sc = self.end_score
                tmp_df = tmp_df.query('@start_sc <= sc < @end_sc')

                #if c.ANSWER_ATR_COL != "":
                #    self.answer_atr = dict(zip(tmp_df.index, tmp_df.loc[:,c.ANSWER_ATR_COL]))

                #dfを辞書にする
                for k, v in zip(tmp_df.index, tmp_df.to_dict(orient='records')):
                    self.answer_dict[k] = v

            if self.same_db_flg == False:

                # メモリ空き容量を取得
                output(datetime.now(), "before db2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                # 長い足のDBから先に全件取得
                db2_index = 0

                for i, db in enumerate(c.DB2_LIST):
                    #1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                    if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                        r.delete(db)  # メモリ節約のため参照したDBは削除する

                    #output(db ,len(result))
                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)
                        if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                            self.db2["c"].append(tmps.get("c"))
                        else:
                            ipt_data = c.INPUT_DATAS[1]
                            ipt_lists = ipt_data.split("_")
                            ok_flg = True
                            #nanのチェック
                            for ipt in ipt_lists:
                                t_input = tmps.get(ipt)
                                if t_input == None or np.isnan(t_input):
                                    ok_flg = False
                                    break
                            if ok_flg == False:
                                continue
                            else:
                                for ipt in ipt_lists:
                                    t_input = tmps.get(ipt)
                                    if ipt == "std~d1" and c.STD_D1_MAX != 0:
                                        if c.STD_D1_MAX <= t_input:
                                            t_input = c.STD_D1_MAX
                                        elif t_input <= (c.STD_D1_MAX * -1):
                                            t_input = c.STD_D1_MAX * -1
                                    self.db2[ipt].append(t_input)

                        self.db2_score[score] = db2_index
                        self.db2_score_list.append(score)

                        db2_index += 1

                    del result

                self.db2 = self.make_input_data_list_np(self.db2, c.INPUT_DATAS[1]) if len(self.c.INPUT_LEN) >= 2 else None

                # メモリ空き容量を取得
                output(datetime.now(), "before db3 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                db3_index = 0
                for i, db in enumerate(c.DB3_LIST):

                    #endscoreより1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                    if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                        r.delete(db)  # メモリ節約のため参照したDBは削除する

                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)
                        if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                            self.db3["c"].append(tmps.get("c"))
                        else:
                            ipt_data = c.INPUT_DATAS[2]
                            ipt_lists = ipt_data.split("_")
                            ok_flg = True
                            # nanのチェック
                            for ipt in ipt_lists:
                                t_input = tmps.get(ipt)
                                if t_input == None or np.isnan(t_input):
                                    ok_flg = False
                                    break
                            if ok_flg == False:
                                continue
                            else:
                                for ipt in ipt_lists:
                                    t_input = tmps.get(ipt)
                                    if ipt == "std~d1" and c.STD_D1_MAX != 0:
                                        if c.STD_D1_MAX <= t_input:
                                            t_input = c.STD_D1_MAX
                                        elif t_input <= (c.STD_D1_MAX * -1):
                                            t_input = c.STD_D1_MAX * -1
                                    self.db3[ipt].append(t_input)

                        self.db3_score[score] = db3_index
                        self.db3_score_list.append(score)

                        db3_index += 1

                    del result

                self.db3 = self.make_input_data_list_np(self.db3, c.INPUT_DATAS[2]) if len(self.c.INPUT_LEN) >= 3 else None

                # メモリ空き容量を取得
                output(datetime.now(), "before db4 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                db4_index = 0
                for i, db in enumerate(c.DB4_LIST):

                    #endscoreより1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                    if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                        r.delete(db)  # メモリ節約のため参照したDBは削除する

                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)

                        if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                            self.db4["c"].append(tmps.get("c"))
                        else:
                            ipt_data = c.INPUT_DATAS[3]
                            ipt_lists = ipt_data.split("_")
                            ok_flg = True
                            #nanのチェック
                            for ipt in ipt_lists:
                                t_input = tmps.get(ipt)
                                if t_input == None or np.isnan(t_input):
                                    ok_flg = False
                                    break
                            if ok_flg == False:
                                continue
                            else:
                                for ipt in ipt_lists:
                                    t_input = tmps.get(ipt)
                                    if ipt == "std~d1" and c.STD_D1_MAX != 0:
                                        if c.STD_D1_MAX <= t_input:
                                            t_input = c.STD_D1_MAX
                                        elif t_input <= (c.STD_D1_MAX * -1):
                                            t_input = c.STD_D1_MAX * -1
                                    self.db4[ipt].append(t_input)

                        self.db4_score[score] = db4_index
                        self.db4_score_list.append(score)

                        db4_index += 1

                    del result

                self.db4 = self.make_input_data_list_np(self.db4, c.INPUT_DATAS[3]) if len(self.c.INPUT_LEN) >= 4 else None

                # メモリ空き容量を取得
                output(datetime.now(), "before db5 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                db5_index = 0
                for i, db in enumerate(c.DB5_LIST):
                    #endscoreより1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                    if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                        r.delete(db)  # メモリ節約のため参照したDBは削除する

                    for j, line in enumerate(result):
                        body = line[0]
                        score = float(line[1])
                        tmps = json.loads(body)

                        if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                            self.db5["c"].append(tmps.get("c"))
                        else:
                            ipt_data = c.INPUT_DATAS[4]
                            ipt_lists = ipt_data.split("_")
                            ok_flg = True
                            #nanのチェック
                            for ipt in ipt_lists:
                                t_input = tmps.get(ipt)
                                if t_input == None or np.isnan(t_input):
                                    ok_flg = False
                                    break
                            if ok_flg == False:
                                continue
                            else:
                                for ipt in ipt_lists:
                                    t_input = tmps.get(ipt)
                                    if ipt == "std~d1" and c.STD_D1_MAX != 0:
                                        if c.STD_D1_MAX <= t_input:
                                            t_input = c.STD_D1_MAX
                                        elif t_input <= (c.STD_D1_MAX * -1):
                                            t_input = c.STD_D1_MAX * -1
                                    self.db5[ipt].append(t_input)

                        self.db5_score[score] = db5_index
                        self.db5_score_list.append(score)

                        db5_index += 1

                    del result

                self.db5 = self.make_input_data_list_np(self.db5, c.INPUT_DATAS[4]) if len(self.c.INPUT_LEN) >= 5 else None


            # メモリ空き容量を取得
            output(datetime.now(), "before db1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            take_profit_cnt = 0
            stop_loss_cnt = 0
            up = 0
            down = 0

            db1_index = 0
            for i, db in enumerate(c.DB1_LIST):

                list_idx = db1_index

                result = r.zrangebyscore(db, self.start_score, self.end_score, withscores=True)

                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(db)  # メモリ節約のため参照したDBは削除する
                output(datetime.now(), "after db1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                close_tmp, devide_tmp, score_tmp, spread_tmp, jpy_tmp, spread_percent_tmp  = [], [], [], [], [], []
                predict_tmp = []
                tag_tmp = []

                atr_tmp = []
                ind_tmp = []

                output_tmp_dict = {}
                for tmp_k in self.opt_lists:
                    output_tmp_dict[tmp_k] = []

                prev_dict = {"c":0, "spr":0,  "jpy":0, }

                for tmp_o in self.opt_lists:
                    prev_dict[tmp_o] = 0

                output("result length:", len(result))
                for line in result:
                    body = line[0]
                    score = float(line[1])
                    try:
                        tmps = json.loads(body)
                    except Exception as e:
                        output(body)
                        output(score)
                        output(tracebackPrint(e))
                        exit(1)
                    c_tmp = float(tmps.get("c"))

                    #特徴量にnanがないかチェック
                    if c.DIVIDE_ALL_FLG or c.DIRECT_FLG:
                        self.db1["c"].append(c_tmp)
                    else:
                        ipt_data = c.INPUT_DATAS[0]
                        ipt_lists = ipt_data.split("_")
                        ok_flg = True
                        for ipt in ipt_lists:
                            t_input = tmps.get(ipt)
                            if t_input == None or np.isnan(t_input):
                                ok_flg = False
                                break
                        if ok_flg == False:
                            continue
                        else:
                            for ipt in ipt_lists:
                                t_input = tmps.get(ipt)
                                if ipt == "std~d1" and c.STD_D1_MAX != 0:
                                    if c.STD_D1_MAX <= t_input:
                                        t_input = c.STD_D1_MAX
                                    elif t_input <= (c.STD_D1_MAX * -1):
                                        t_input = c.STD_D1_MAX * -1

                                self.db1[ipt].append(t_input)

                    score_tmp.append(score)
                    close_tmp.append(c_tmp)

                    j_tmp = tmps.get("jpy")
                    if c.JPY_FLG == False:
                        j_tmp = float(j_tmp) if j_tmp != None else c.JPY_FIX
                    jpy_tmp.append(j_tmp)

                    sp_tmp = tmps.get("sp")
                    sp_tmp = float(sp_tmp) if sp_tmp != None else None
                    spread_percent_tmp.append(sp_tmp)

                    if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                        if c.FX_TICK_DB != c.DB1_LIST[0]:
                            tick_tmp = self.tick_dict.get(score)
                        else:
                            tick_tmp = tmps.get("tk")
                            self.tick_dict[score] = tick_tmp

                        if tick_tmp == None:
                            print("tick is None:", score)
                            exit(1)

                    a_tmp = self.atr_dict.get(score)
                    atr_tmp.append(a_tmp)

                    i_tmp = self.ind_score_dict.get(score)
                    if len(c.IND_COLS) !=0 and i_tmp == None:
                        #scoreで該当するINDがない場合はIND_COLS数分の空の配列をつくり、ind_listのshapeをそろえる
                        #そろえないとlistからndarrayにした時にきれいに変換されず行や列指定で値を取得できない
                        i_tmp = []
                        for col in c.IND_COLS:
                            i_tmp.append(None)
                    ind_tmp.append(i_tmp)

                    for tmp_k in self.opt_lists:
                        tmp_v = tmps.get(tmp_k)
                        tmp_v = float(tmp_v) if tmp_v != None else tmp_v
                        output_tmp_dict[tmp_k].append(tmp_v)

                    if c.METHOD == "LSTM2":
                        if tmps.get("p") != None:
                            predict_tmp.append(tmps.get("p"))
                        else:
                            predict_tmp.append(None)
                    else:
                        predict_tmp.append(None)

                    if c.METHOD == "LSTM9":
                        if c.LSTM9_USE_CLOSE:
                            self.db_close.append(c_tmp)

                        for dbname in c.LSTM9_PRED_DBS:
                            if tmps.get(dbname) != None:
                                self.db_preds[dbname].append(tmps.get(dbname))
                            else:
                                self.db_preds[dbname].append(-1)
                    if c.TAG != "":
                        tag_tmp.append(tmps.get(c.TAG))

                    #self.db1_score[score] = db1_index
                    #self.db1_score_list.append(score)

                    spr = 0
                    #ハイロー,FXでSpreadデータを使用する場合
                    if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
                        spr = tmps.get("s")
                        if spr == None:
                            spr = 0
                        else:
                            spr = float(spr)
                            if spr <1 and 0 < spr:
                                spr = float(spr * 10) #sprがpips形式(0.1など)で入っている場合
                            else:
                                spr = float(spr)
                        spread_tmp.append(spr)

                    else:
                        spr = SPREAD_CORRECT -1
                        spread_tmp.append(spr)

                    #test用にscoreをキーにレートとスプレッドを保持
                    #レートはそのscoreのopenレートとする
                    if test_flg:
                        self.score_dict[score] = {}
                        if prev_dict["c"] == 0:
                            # prev_cがない最初のレコードの場合、しょうがないので現在のレートを入れる
                            self.score_dict[score]["c"] = c_tmp
                            self.score_dict[score]["spr"] = spr
                            self.score_dict[score]["atr"] = a_tmp
                            self.score_dict[score]["jpy"] = j_tmp
                            self.score_dict[score]["sp"] = sp_tmp
                            self.score_dict[score]["ind"] = i_tmp

                            if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                                self.score_dict[score]["tk"] = tick_tmp

                            for tmp_k in self.opt_lists:
                                tmp_v = tmps.get(tmp_k)
                                tmp_v = float(tmp_v) if tmp_v != None else tmp_v
                                self.score_dict[score][tmp_k] = tmp_v
                        else:
                            self.score_dict[score]["c"] = prev_dict["c"]
                            self.score_dict[score]["spr"] = prev_dict["spr"]
                            self.score_dict[score]["atr"] = a_tmp #atrは予想時の値が入っているのでそのまま設定
                            self.score_dict[score]["jpy"] = prev_dict["jpy"]
                            self.score_dict[score]["sp"] = prev_dict["sp"]
                            self.score_dict[score]["ind"] = i_tmp #indは予想時の値が入っているのでそのまま設定

                            if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                                self.score_dict[score]["tk"] = prev_dict["tk"]

                            for tmp_k in self.opt_lists:
                                self.score_dict[score][tmp_k] = prev_dict[tmp_k]

                        prev_dict["c"] = c_tmp
                        prev_dict["spr"] = spr
                        prev_dict["jpy"] = j_tmp
                        prev_dict["sp"] = sp_tmp

                        if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                            prev_dict["tk"] = tick_tmp

                        for tmp_k in self.opt_lists:
                            tmp_v = tmps.get(tmp_k)
                            tmp_v = float(tmp_v) if tmp_v != None else tmp_v
                            prev_dict[tmp_k] = tmp_v

                    db1_index += 1

                del result

                #output(datetime.fromtimestamp(min(self.db1_score)))
                #output(datetime.fromtimestamp(max(self.db1_score)))

                list_idx = list_idx -1

                for i in range(len(score_tmp)):
                    list_idx += 1

                    if list_idx % 1000000 == 0:
                        output(datetime.now(), list_idx, psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
                        output("train_list size:", self.train_list.__sizeof__())
                    """
                    if list_idx % 10000000 == 0:
                        output(datetime.now(), list_idx, "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
                        gc.collect()
                        output(datetime.now(), list_idx, "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
                    """
                    now_score_tmp = score_tmp[i]
                    sec = datetime.fromtimestamp(now_score_tmp).second
                    minute = datetime.fromtimestamp(now_score_tmp).minute
                    hour = datetime.fromtimestamp(now_score_tmp).hour
                    week = get_weeknum(datetime.fromtimestamp(now_score_tmp).weekday(), datetime.fromtimestamp(now_score_tmp).day)

                    need_len = c.INPUT_LEN[0] + c.INPUT_DATA_LENGTHS[0]
                    if c.DIVIDE_ALL_FLG:
                        need_len = c.INPUT_LEN[0] + c.INPUT_DATA_LENGTHS[0]

                    #inputデータが足りない場合スキップ
                    if i < need_len:
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    try:
                        start_score = score_tmp[i - need_len]
                        end_score = score_tmp[i + c.PRED_TERM + c.END_TERM -1]
                        if end_score != get_decimal_add(start_score, get_decimal_multi((need_len + c.PRED_TERM + c.END_TERM - 1), c.DB1_TERM)):
                            #時刻がつながっていないものは除外 たとえば日付またぎなど
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    except Exception:
                        #start_score end_scoreのデータなしなのでスキップ
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    prev_close = close_tmp[i-1 + c.START_TERM]
                    pred_close = close_tmp[i-1 + c.PRED_TERM + c.END_TERM ]
                    prev_spread = spread_tmp[i - 1]

                    if c.METHOD == "LSTM9":
                        skip_flg = False
                        for ipt9 in c.LSTM9_INPUTS:
                            if self.db_preds[self.db9_name][i - ipt9] == -1:
                                # 予想がない場合はスキップ
                                skip_flg = True
                        if skip_flg:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    db_volume_index_tmp = None
                    db2_index_tmp = None
                    db3_index_tmp = None
                    db4_index_tmp = None
                    db5_index_tmp = None

                    db_extra_1_index_tmp = None

                    db_foot_idxs = {} if len(c.FOOT_DBS) != 0 else None
                    break_flg = False
                    for db_tmp in c.FOOT_DBS:
                        d_term, d_len, d_unit, d_x, db_name, separate_flg = db_tmp
                        try:
                            #直近の足のスコア
                            #tmp_score_end = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(str(d_term)))) - d_term
                            tmp_score_end = get_decimal_sub(get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, d_term)), d_term)

                            #データの開始、終了がそれぞれ存在するかチェック　例外がなけられれば存在しない
                            tmp_idx_end = self.db_foots_score[d_term][tmp_score_end]
                            if tmp_idx_end < (d_len -1):
                                #データが足りていないので学習対象外
                                break_flg = True
                                break

                            db_foot_idxs[d_term] = tmp_idx_end

                        except Exception as e:
                            # データがないのでスキップ
                            break_flg = True
                            break

                    if break_flg:
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    # DB_FAKEを使う場合
                    if c.DB_FAKE_TERM != 0:
                        need_len = c.DB_FAKE_INPUT_LEN + c.INPUT_DATA_LENGTHS[0]
                        if c.DIVIDE_ALL_FLG:
                            need_len = c.DB_FAKE_INPUT_LEN + 1
                        try:
                            db_fake_index_tmp = self.db_fake_score[now_score_tmp]  # scoreからインデックスを取得
                            start_score = self.db_fake_score_list[db_fake_index_tmp - need_len]
                            end_score = self.db_fake_score_list[db_fake_index_tmp]
                            if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB_FAKE_TERM)):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if c.METHOD == "LSTM8":
                        need_len = c.DB_VOLUME_INPUT_LEN + c.INPUT_DATA_LENGTHS[0]
                        try:
                            db_volume_index_tmp = self.db_volume_score[now_score_tmp]  # scoreからインデックスを取得
                            start_score = self.db_volume_score_list[db_volume_index_tmp - need_len]
                            end_score = self.db_volume_score_list[db_volume_index_tmp]
                            if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB_VOLUME_TERM)):
                                # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    ind_foot_list = None
                    if len(c.IND_FOOT_COL) != 0 :
                        ind_foot_list = []
                        try:
                            # 直近の足のスコア
                            tmp_term_ind = c.IND_FOOT_COL.split("-")[0]
                            #tmp_score = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(tmp_term_ind)))
                            tmp_score = get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, tmp_term_ind))
                            tmp_foot_dict = self.ind_foot_dict[tmp_score]
                            break_flg = False
                            for col in c.IND_FOOT_COL:
                                tmp_ind = tmp_foot_dict.get(col)

                                if tmp_ind == None or np.isnan(tmp_ind) :
                                    # 値がない場合はスキップ
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    break_flg = True
                                    break

                                if "sma" in col:
                                    ind_foot_list.append(get_divide(prev_close, tmp_ind))
                                elif "bbu" in col or "bbl" in col:
                                    ind_foot_list.append(get_divide(prev_close, tmp_ind))
                                else:
                                    ind_foot_list.append(tmp_ind)
                            if break_flg:
                                continue

                        except Exception as e:
                            output(e)
                            # データがないのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    hor_val = None
                    for j, hor in c.HOR_DB_CORE_LIST:
                        hor_width = float(hor.split("_")[2])  # 同一線上とみなすレート幅
                        hor_term_sec = float(hor.split("_")[2])  # データ作成に使用した足(sec)

                        try:
                            #直近の足のスコア
                            tmp_s = get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, hor_term_sec))
                            hor_data = self.hor_dict[j][tmp_s]

                            tmp_dict = {}
                            for tmp_data in hor_data.split(","):
                                tmp_rate, hit_cnt = tmp_data.split(":")

                                hit_cnt = int(hit_cnt)
                                tmp_dict[tmp_rate] = hit_cnt

                            hor_val_tmp = []
                            #今のレートのあたりにある意識すべき水平線のレート数を保存
                            target = get_decimal_sub(prev_close, Decimal(str(prev_close)) % Decimal(str(hor_width)))

                            tmp_hit_cnt = tmp_dict.get(target)
                            if tmp_hit_cnt ==  None:
                                tmp_hit_cnt = 0

                            hor_val_tmp.append(tmp_hit_cnt)

                            if c.HOR_LINE_NUM == 3:
                                low_target = get_decimal_sub(target, hor_width)
                                tmp_hit_cnt_low = tmp_dict.get(low_target)
                                if tmp_hit_cnt_low == None:
                                    tmp_hit_cnt_low = 0
                                hor_val_tmp.append(tmp_hit_cnt_low)

                                high_target = get_decimal_add(target, hor_width)
                                tmp_hit_cnt_high = tmp_dict.get(high_target)
                                if tmp_hit_cnt_high == None:
                                    tmp_hit_cnt_high = 0

                                hor_val_tmp.append(tmp_hit_cnt_high)

                            hor_val.append(hor_val_tmp)

                        except Exception as e:
                            #データがないのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    highlow_val = None
                    if c.HIGHLOW_DB_CORE != "":
                        try:
                            #直近の足のスコア
                            tmp_score = float(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(c.HIGHLOW_TERM)))
                            highlow_data = self.highlow_dict[tmp_score]

                            highlow_val = []
                            start_v = 1
                            while True:
                                if start_v > c.HIGHLOW_DATA_NUM:
                                    break

                                data_length = str(int(get_decimal_multi(c.HIGHLOW_TERM, start_v)))
                                high = highlow_data[data_length + "_h"]
                                low = highlow_data[data_length + "_l"]

                                highlow_val.append(get_decimal_sub(high, prev_close))
                                highlow_val.append(get_decimal_sub(low, prev_close))

                                start_v += 1

                        except Exception as e:
                            #output(tracebackoutput(e))
                            #データがないのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue


                    oanda_ord_list = None
                    if c.OANDA_ORD_DB != "":
                        try:
                            #直近の足のスコア
                            tmp_score = float(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal("300")))
                            wid, ord_data = self.oanda_ord_dict[tmp_score]
                            wid = float(wid)

                            tmp_val_list = []
                            mid_ind = None

                            for k, tmp_data in enumerate(ord_data.split(",")):
                                tmp_rate, tmp_val = tmp_data.split(":")
                                tmp_rate = float(tmp_rate)
                                tmp_val = float(tmp_val)
                                tmp_val_list.append(tmp_val)

                                if mid_ind == None and tmp_rate <= prev_close and prev_close <tmp_rate + wid:
                                    #output(tmp_rate, prev_close, k)
                                    #現在レートが属するレンジが何番目か特定
                                    mid_ind = k

                            if mid_ind == None:
                                # 該当レンジがないのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue
                            else:
                                if c.OANDA_ORD_NUM != 0:
                                    #現在レートより下のレンジデータを追加
                                    oanda_ord_list = tmp_val_list[mid_ind - c.OANDA_ORD_NUM: mid_ind]
                                    # 現在レートのレンジ追加
                                    oanda_ord_list.append(tmp_val_list[mid_ind])
                                    #現在レートより上のレンジデータを追加
                                    oanda_ord_list.extend(tmp_val_list[mid_ind + 1: mid_ind + 1 + c.OANDA_ORD_NUM])
                                else:
                                    oanda_ord_list = [tmp_val_list[mid_ind]]

                                if len(oanda_ord_list) < int(c.OANDA_ORD_NUM * 2 + 1):
                                    # データが足りてないのでスキップ
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                        except Exception as e:
                            #データがないのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    oanda_pos_list = None
                    if c.OANDA_POS_DB != "":
                        try:
                            # 直近の足のスコア
                            tmp_score = float(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal("300")))
                            wid, pos_data = self.oanda_pos_dict[tmp_score]
                            wid = float(wid)

                            tmp_val_list = []
                            mid_ind = None
                            for k, tmp_data in enumerate(pos_data.split(",")):
                                tmp_rate, tmp_val = tmp_data.split(":")
                                tmp_rate = float(tmp_rate)
                                tmp_val = float(tmp_val)
                                tmp_val_list.append(tmp_val)

                                if mid_ind == None and tmp_rate <= prev_close and prev_close <tmp_rate + wid:
                                    # 現在レートが属するレンジが何番目か特定
                                    mid_ind = k

                            if mid_ind == None:
                                # 該当レンジがないのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue
                            else:
                                if c.OANDA_POS_NUM != 0:
                                    # 現在レートより下のレンジデータを追加
                                    oanda_pos_list = tmp_val_list[mid_ind - c.OANDA_POS_NUM: mid_ind]
                                    # 現在レートのレンジ追加
                                    oanda_pos_list.append(tmp_val_list[mid_ind])
                                    # 現在レートより上のレンジデータを追加
                                    oanda_pos_list.extend(tmp_val_list[mid_ind + 1: mid_ind + 1 + c.OANDA_POS_NUM])
                                else:
                                    oanda_pos_list = [tmp_val_list[mid_ind]]

                                if len(oanda_pos_list) < int(c.OANDA_POS_NUM * 2 + 1):
                                    #データが足りてないのでスキップ
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                        except Exception as e:
                            # データがないのでスキップ
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if self.same_db_flg == False:

                        #DB2を使う場合
                        if len(c.INPUT_LEN) > 1:
                            need_len = c.INPUT_LEN[1] + c.INPUT_DATA_LENGTHS[1]
                            if c.DIVIDE_ALL_FLG:
                                need_len = c.INPUT_LEN[1] + c.INPUT_DATA_LENGTHS[1]
                            try:
                                db2_index_tmp = self.db2_score[now_score_tmp]  # scoreからインデックスを取得
                                start_score = self.db2_score_list[db2_index_tmp - need_len]
                                end_score = self.db2_score_list[db2_index_tmp]
                                if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB2_TERM)):
                                    #時刻がつながっていないものは除外 たとえば日付またぎなど
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                            except Exception:
                                #start_scoreのデータなしなのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        # DB3を使う場合
                        if len(c.INPUT_LEN) > 2:
                            need_len = c.INPUT_LEN[2] + c.INPUT_DATA_LENGTHS[2]
                            if c.DIVIDE_ALL_FLG:
                                need_len = c.INPUT_LEN[2] + c.INPUT_DATA_LENGTHS[2]
                            try:
                                db3_index_tmp = self.db3_score[now_score_tmp]  # scoreからインデックスを取得
                                start_score = self.db3_score_list[db3_index_tmp - need_len]
                                end_score = self.db3_score_list[db3_index_tmp]
                                if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB3_TERM)):
                                    # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                            except Exception:
                                # start_scoreのデータなしなのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        # DB4を使う場合
                        if len(c.INPUT_LEN) > 3:
                            need_len = c.INPUT_LEN[3] + c.INPUT_DATA_LENGTHS[3]
                            if c.DIVIDE_ALL_FLG:
                                need_len = c.INPUT_LEN[3] + c.INPUT_DATA_LENGTHS[3]
                            try:
                                db4_index_tmp = self.db4_score[now_score_tmp]  # scoreからインデックスを取得
                                start_score = self.db4_score_list[db4_index_tmp - need_len]
                                end_score = self.db4_score_list[db4_index_tmp]
                                if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB4_TERM)):
                                    # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                            except Exception:
                                # start_scoreのデータなしなのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        # DB5を使う場合
                        if len(c.INPUT_LEN) > 4:
                            need_len = c.INPUT_LEN[4] + c.INPUT_DATA_LENGTHS[4]
                            if c.DIVIDE_ALL_FLG:
                                need_len = c.INPUT_LEN[4] + c.INPUT_DATA_LENGTHS[4]
                            try:
                                db5_index_tmp = self.db5_score[now_score_tmp]  # scoreからインデックスを取得
                                start_score = self.db5_score_list[db5_index_tmp - need_len]
                                end_score = self.db5_score_list[db5_index_tmp]
                                if end_score != get_decimal_add(start_score, get_decimal_multi(need_len, c.DB5_TERM)):
                                    # 時刻がつながっていないものは除外 たとえば日付またぎなど
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                            except Exception:
                                # start_scoreのデータなしなのでスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        # DB EXTRA 1を使う場合
                        if c.DB_EXTRA_1 != "":
                            db_extra_1_index_tmp = self.db_extra_1_score[now_score_tmp]  # scoreからインデックスを取得

                    #取引時間外を対象からはずす
                    if self.test_flg == False and len(c.EXCEPT_LIST) != 0:
                        if datetime.fromtimestamp(now_score_tmp).hour in c.EXCEPT_LIST:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    # スプレッドごとの取引時間外を対象からはずす
                    if prev_spread in c.EXCEPT_LIST_BY_SPERAD:
                        if datetime.fromtimestamp(now_score_tmp).hour in c.EXCEPT_LIST_BY_SPERAD[prev_spread]:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    # 指定スプレッド以外のトレードは無視する
                    if len(TARGET_SPREAD_LIST) != 0:
                        if not (prev_spread in TARGET_SPREAD_LIST):
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    # BITCOIN用 指定スプレッドパーセント以外のトレードは無視する
                    if len(target_spread_percent_list) != 0:
                        if not (spread_percent_tmp[i -1] in target_spread_percent_list):
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if get_decimal_mod(now_score_tmp, c.LEARNING_SHIFT) != 0.0 and test_flg == False:
                        #指定したシフトでなければ学習対象外
                        continue

                    if get_decimal_mod(now_score_tmp, c.BET_SHIFT) != 0.0:
                        #指定したシフトでなければ無視
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    # 0秒のデータのみ学習する場合
                    if self.test_flg == False and c.ZERO_SEC_FLG and sec != 0:
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    # オプションの値がnanやNoneの場合は無視する
                    tmp_opt = None
                    if len(c.OPTIONS) != 0:
                        try:
                            tmp_opt = self.option_score[now_score_tmp]
                            if tmp_opt == None:
                                #データが続いていない場合Noneが入っている
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue
                            else:
                                for op in tmp_opt:
                                    if op == None or np.isnan(op):
                                        if test_flg and eval_flg == False:
                                            self.train_dict_ex[now_score_tmp] = None
                                        continue
                        except Exception:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    #jpyの通貨ペアでない場合、日本円レートがDBにない場合はポジション計算できない(get_fx_position_jpy)のでスキップ
                    if c.JPY_FLG == False:
                        now_jpy_tmp = jpy_tmp[i - 1]
                        if now_jpy_tmp == None or np.isnan(now_jpy_tmp):
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if c.ATR_COL != "":
                        now_atr_tmp = atr_tmp[i]
                        if now_atr_tmp == None or np.isnan(now_atr_tmp):
                            #データが続いていない場合Noneが入っている
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue
                        else:
                            # ATRを使用する場合は値で絞る
                            if len(c.ATR) != 0:
                                ok_flg = False
                                for t_atr in c.ATR:
                                    atr_min, atr_max = t_atr.split("-")
                                    if (float(atr_min) <= now_atr_tmp and now_atr_tmp < float(atr_max)) == True:
                                        ok_flg = True
                                        break

                                if ok_flg == False:
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                    if len(c.IND_COLS) != 0:
                        break_flg = False
                        now_ind_tmp = ind_tmp[i]
                        for j, col in enumerate(c.IND_COLS):
                            val_tmp = now_ind_tmp[j]
                            if val_tmp == None or np.isnan(val_tmp):
                                #データが続いていない場合Noneが入っている
                                break_flg = True
                                break
                            else:
                                if len(c.IND_RANGES[j]) != 0:
                                    ok_flg = False
                                    for r in c.IND_RANGES[j]:
                                        r_min, r_max = r.split("-")
                                        if (float(r_min) <= val_tmp and val_tmp < float(r_max)) == True:
                                            ok_flg = True
                                            break

                                    if ok_flg == False:
                                        break_flg = True
                                        break
                        if break_flg:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if test_flg == False and c.TAG != "":
                        #TAG指定されている場合、タグがついていないデータは対象外とする
                        if tag_tmp[i] == None:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if c.METHOD == "LSTM2":
                        if predict_tmp[i] == None:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                        """
                        #予想値をdivide / predict_tmp[i]にしている場合、0を除外。0除算エラーとなるため
                        if not(test_flg and eval_flg == False) and predict_tmp[i] == 0:
                            self.train_dict_ex[now_score_tmp] = None
                            continue
                        """

                    #直近の変化率
                    divide_prev_length = int(get_decimal_divide(c.DIVIDE_PREV_SEC, c.DB1_TERM))

                    try:
                        start_score = score_tmp[i - 1 - divide_prev_length]
                        end_score = score_tmp[i - 1]
                        if end_score != get_decimal_add(start_score, get_decimal_multi(divide_prev_length, c.DB1_TERM)):
                            #時刻がつながっていないものは除外 たとえば日付またぎなど
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    except Exception:
                        #start_score end_scoreのデータなしなのでスキップ
                        if test_flg and eval_flg == False:
                            self.train_dict_ex[now_score_tmp] = None
                        continue

                    try:
                        divide_prev = abs(get_divide(close_tmp[i - 1 - divide_prev_length], prev_close))
                    except Exception:
                        print("divide_prev_length short:", score_tmp[i])
                        exit(1)


                    #直近の変化率によって対象から外す
                    if c.EXCEPT_DIVIDE_MIN !=0 and c.EXCEPT_DIVIDE_MIN > divide_prev:
                        continue
                    if c.EXCEPT_DIVIDE_MAX != 0 and c.EXCEPT_DIVIDE_MAX < divide_prev:
                        continue

                    if test_flg == False and c.EXCEPT_DIVIDE_MIN_AFTER != 0:
                        # 答えまでの変化率でしぼる場合
                        divide_after = abs(get_divide(prev_close, pred_close))

                        if c.EXCEPT_DIVIDE_MIN_AFTER > divide_after:
                            continue

                    #経済指標発表時前後を除外する
                    if test_flg == False:
                        if c.EXCEPT_IMPORTANT_INDEX_RANGE != None:
                            if importantAnswer.is_except(now_score_tmp):
                                continue

                    if len(c.SUB) != 0 and test_flg == False:
                        tmp_sub = abs(get_sub(prev_close, pred_close))
                        #現在から答えのレートまでの差の大きさで絞る
                        ok_flg = False
                        for t_sub in c.SUB:
                            sub_min, sub_max = t_sub.split("-")
                            if (float(sub_min) <= tmp_sub and tmp_sub < float(sub_max)) == True:
                                ok_flg = True
                                break

                        if ok_flg == False:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            continue

                    if c.LEARNING_TYPE == "CATEGORY_BIN" and (test_flg == False or (test_flg and eval_flg)):
                        #CATEGORY_BINの場合はレート変化ない場合は学習対象外とする
                        if c.OUTPUT_TYPE == "d" and self.sub_force == False:
                            # CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する
                            divide_t = get_decimal_multi(get_divide(prev_close, pred_close), c.OUTPUT_MULTI)
                            if abs(divide_t) <= c.BORDER_DIV:
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                        elif c.OUTPUT_TYPE == "sub" or self.sub_force == True:
                            #CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する

                            sub_t = get_sub(prev_close, pred_close, c.OUTPUT_MULTI)
                            if abs(sub_t) <= float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                    opt_bef = {}
                    opt_aft = {}
                    ok_flg = True
                    for tmp_k in self.opt_lists:
                        if c.OUTPUT_DATA_BEF_C: #変化前の基準をcloseにする場合
                            opt_bef_tmp = prev_close
                        else:
                            opt_bef_tmp = output_tmp_dict[tmp_k][i - 1]
                        opt_aft_tmp = output_tmp_dict[tmp_k][i - 1 + c.PRED_TERM]
                        #答えの基準となる値がNoneならスキップ
                        if opt_bef_tmp == None or opt_aft_tmp == None:
                            if test_flg and eval_flg == False:
                                self.train_dict_ex[now_score_tmp] = None
                            ok_flg = False
                            break

                        opt_bef[tmp_k] = opt_bef_tmp
                        opt_aft[tmp_k] = opt_aft_tmp

                    if ok_flg == False:
                        continue

                    ok_flg = True
                    for k, a in enumerate(c.ADDITIONAL_DATA_LIST):
                        input_type = a["input_type"]

                        if a["length"] == "H1":
                            target_score = get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, 3600))  # 自分の所属する足のスコア
                        elif a["length"] == "M1":
                            target_score = get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, 60))  # 自分の所属する足のスコア

                        if input_type == "c":
                            target_idx = self.additional_data_list[k]["score_dict"].get(target_score)  # 自分の所属する足のindexを取得する

                            if target_idx == None:
                                #足が存在しなければスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                ok_flg = False
                                break
                        elif input_type == "sma":
                            if a["length"] == "H1":
                                target_score = get_decimal_sub(target_score,3600)  # 自分の所属する足の一つ前
                            elif a["length"] == "M1":
                                target_score = get_decimal_sub(target_score,60)  # 自分の所属する足の一つ前

                            target_data = self.additional_data_list[k]["data_dict"].get(target_score)  # 自分の所属する足の一つ前を取得する

                            if target_data == None:
                                #足が存在しなければスキップ
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                ok_flg = False
                                break

                    if ok_flg == False:
                        continue

                    # 正解をいれていく
                    bef = prev_close
                    aft = pred_close

                    # output d用
                    divide = get_divide(bef, aft)
                    sub = get_sub(bef, aft)

                    if test_flg == False:
                        test_divide = divide

                        if c.DIVIDE_MAX !=0 :
                            if abs(test_divide) < c.DIVIDE_MIN or c.DIVIDE_MAX < abs(test_divide) :
                                #変化率が大きすぎる場合 外れ値とみなして除外
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue
                        else:
                            if abs(test_divide) < c.DIVIDE_MIN :
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue

                    if c.ANSWER_DB != "":
                        #ifd_data = r.zrangebyscore(c.ANSWER_DB, now_score_tmp, now_score_tmp, withscores=True)
                        if self.test_flg and self.eval_flg == False:
                            try:
                                ifd_data = self.answer_dict[now_score_tmp]
                            except Exception:
                                if test_flg and eval_flg == False:
                                    self.train_dict_ex[now_score_tmp] = None
                                continue
                        else:
                            #テスト時ではなくても"OCOPS"の時は学習に使用するので読み込む
                            if "OCOPS:" in c.ANSWER_DB:
                                try:
                                    ifd_data = self.answer_dict[now_score_tmp]
                                except Exception:
                                    if test_flg and eval_flg == False:
                                        self.train_dict_ex[now_score_tmp] = None
                                    continue

                    #正解の差を入れていく
                    #for tmp_k in self.opt_lists:
                    #    self.output_answer_dict[tmp_k].append(abs(opt_aft[tmp_k] - opt_bef[tmp_k]))

                    if test_flg and eval_flg == False:
                        #spread情報取得
                        if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
                            self.spread_cnt += 1
                            spr = prev_spread

                            if self.spread_cnt_dict.get(spr) == None:
                                self.spread_cnt_dict[spr] = 1
                            else:
                                self.spread_cnt_dict[spr] += 1

                    #以下はcontinue処理なし
                    if len(c.HOR_DB_CORE_LIST) != 0:
                        self.hor_list.append(hor_val)

                    #正解までの変化率
                    divide_aft = abs(divide)

                    answer_rate_up = sub
                    answer_score_up = get_decimal_add(now_score_tmp, c.TERM)
                    answer_rate_dw = (sub * -1)
                    answer_score_dw = get_decimal_add(now_score_tmp, c.TERM)

                    answer_rate = [answer_rate_up, answer_rate_dw]
                    answer_score = [answer_score_up, answer_score_dw]

                    spread_correct = SPREAD_CORRECT
                    spread_t = prev_spread
                    spread_t_end = spread_tmp[i - 1 + c.PRED_TERM]

                    if c.FX == False and self.real_spread_flg:
                        spread_correct = prev_spread + 1

                    tmp_label = []
                    if test_flg and eval_flg == False:
                        #テスト時は対象の正解のみ求める
                        c.RETURN_Y_LIST = [c.GET_Y_STR]

                    for tmp_y in c.RETURN_Y_LIST:
                        LEARNING_TYPE = tmp_y.split("-")[0]
                        OUTPUT_TYPE = tmp_y.split("-")[1]
                        BORDER = tmp_y.split("-")[2]
                        if LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_FOUR", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW"]:
                            BORDER = float(BORDER)

                        if LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_FOUR", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW"] or \
                                (LEARNING_TYPE == "CATEGORY_BIN" and test_flg and eval_flg ==False):
                            if OUTPUT_TYPE == "d" and self.sub_force == False:
                                # CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する
                                divide_t = get_decimal_multi(get_divide(opt_bef[c.OUTPUT_DATA],  opt_aft[c.OUTPUT_DATA]), c.OUTPUT_MULTI)
                                if divide_t > BORDER:
                                    # 上がった場合
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            lowest_rate =  tmp_hl_data["l"]
                                            stoploss_tmp = get_decimal_add(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if lowest_rate <= stoploss_tmp:
                                                # ストップロスに引っかかった場合SAMEを正解とする
                                                tmp_label.append([0, 1, 0])
                                            else:
                                                tmp_label.append([1, 0, 0])
                                        else:
                                            tmp_label.append([1, 0, 0])

                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            lowest_rate =  tmp_hl_data["l"]
                                            stoploss_tmp = get_decimal_add(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if lowest_rate <= stoploss_tmp:
                                                # ストップロスに引っかかった場合
                                                tmp_label.append([0, 1])
                                            else:
                                                tmp_label.append([1, 0])
                                        else:
                                            tmp_label.append([1, 0])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        tmp_label.append([0, 1])

                                    if tmp_y == c.GET_Y_STR:
                                        up = up + 1

                                elif divide_t < get_decimal_multi(BORDER, -1):
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            highest_rate = tmp_hl_data["h"]
                                            stoploss_tmp = get_decimal_sub(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if highest_rate >= stoploss_tmp:
                                                # ストップロスに引っかかった場合SAMEを正解とする
                                                tmp_label.append([0, 1, 0])
                                            else:
                                                tmp_label.append([0, 0, 1])
                                        else:
                                            tmp_label.append([0, 0, 1])

                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        tmp_label.append([0, 1])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            highest_rate = tmp_hl_data["h"]
                                            stoploss_tmp = get_decimal_sub(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if highest_rate >= stoploss_tmp:
                                                # ストップロスに引っかかった場合
                                                tmp_label.append([0, 1])
                                            else:
                                                tmp_label.append([1, 0])
                                        else:
                                            tmp_label.append([1, 0])

                                    if tmp_y == c.GET_Y_STR:
                                        down = down + 1
                                else:
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        tmp_label.append([0, 1, 0])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        tmp_label.append([0, 1])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        tmp_label.append([0, 1])

                            elif OUTPUT_TYPE == "sub" or self.sub_force == True:
                                #CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する

                                bef_sub = opt_bef[c.OUTPUT_DATA]
                                aft_sub = opt_aft[c.OUTPUT_DATA]
                                sub_t = get_sub(bef_sub, aft_sub, c.OUTPUT_MULTI)
                                if sub_t > float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                                    # 上がった場合
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            lowest_rate =  tmp_hl_data["l"]
                                            stoploss_tmp = get_decimal_add(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if lowest_rate <= stoploss_tmp:
                                                # ストップロスに引っかかった場合SAMEを正解とする
                                                tmp_label.append([0, 1, 0])
                                            else:
                                                tmp_label.append([1, 0, 0])
                                        else:
                                            tmp_label.append([1, 0, 0])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            lowest_rate =  tmp_hl_data["l"]
                                            stoploss_tmp = get_decimal_add(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if lowest_rate <= stoploss_tmp:
                                                # ストップロスに引っかかった場合
                                                tmp_label.append([0, 1])
                                            else:
                                                tmp_label.append([1, 0])
                                        else:
                                            tmp_label.append([1, 0])

                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        tmp_label.append([0, 1])

                                    if tmp_y == c.GET_Y_STR:
                                        up = up + 1
                                elif get_decimal_multi(sub_t, -1) > float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            highest_rate = tmp_hl_data["h"]
                                            stoploss_tmp = get_decimal_sub(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if highest_rate >= stoploss_tmp:
                                                # ストップロスに引っかかった場合SAMEを正解とする
                                                tmp_label.append([0, 1, 0])
                                            else:
                                                tmp_label.append([0, 0, 1])
                                        else:
                                            tmp_label.append([0, 0, 1])

                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        tmp_label.append([0, 1])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        if c.LEARNING_TYPE_STOPLOSS:
                                            # STOPLOSSに引っかかる場合は不正解とする
                                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                                            highest_rate = tmp_hl_data["h"]
                                            stoploss_tmp = get_decimal_sub(bef, c.LEARNING_TYPE_STOPLOSS_PRICE)
                                            if highest_rate >= stoploss_tmp:
                                                # ストップロスに引っかかった場合
                                                tmp_label.append([0, 1])
                                            else:
                                                tmp_label.append([1, 0])
                                        else:
                                            tmp_label.append([1, 0])

                                    if tmp_y == c.GET_Y_STR:
                                        down = down + 1
                                else:
                                    if LEARNING_TYPE == "CATEGORY" or LEARNING_TYPE == "CATEGORY_BIN" or LEARNING_TYPE == "CATEGORY_BIN_BOTH" or LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                        tmp_label.append([0, 1, 0])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_UP":
                                        tmp_label.append([0, 1])
                                    elif LEARNING_TYPE == "CATEGORY_BIN_DW":
                                        tmp_label.append([0, 1])
                            else:
                                output("WRONG OUTPUT_TYPE!!!", OUTPUT_TYPE)
                                exit(1)

                        elif LEARNING_TYPE == "CATEGORY_BIN":

                            if OUTPUT_TYPE == "d" and self.sub_force == False:
                                divide_t = get_decimal_multi(get_divide(opt_bef[c.OUTPUT_DATA],  opt_aft[c.OUTPUT_DATA]), c.OUTPUT_MULTI)
                                if divide_t > BORDER:
                                    # 上がった場合
                                    tmp_label.append([1, 0])

                                    if tmp_y == c.GET_Y_STR:
                                        up = up + 1
                                elif divide_t < get_decimal_multi(BORDER, -1):
                                    tmp_label.append([0, 1])

                                    if tmp_y == c.GET_Y_STR:
                                        down = down + 1

                            elif OUTPUT_TYPE == "sub" or self.sub_force == True:
                                bef_sub = opt_bef[c.OUTPUT_DATA]
                                aft_sub = opt_aft[c.OUTPUT_DATA]
                                sub_t = get_sub(bef_sub, aft_sub, c.OUTPUT_MULTI)
                                if sub_t > float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                                    # 上がった場合
                                    tmp_label.append([1, 0])
                                    if tmp_y == c.GET_Y_STR:
                                        up = up + 1
                                elif sub_t < get_decimal_multi(float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))), -1):
                                    tmp_label.append([0, 1])

                                    if tmp_y == c.GET_Y_STR:
                                        down = down + 1

                        elif LEARNING_TYPE in ["REGRESSION", "REGRESSION_SIGMA"]:
                            if c.METHOD == "LSTM2":
                                if predict_tmp[i] != None:
                                    tmp_label.append(divide - predict_tmp[i])
                                #else:
                                #    #予想はかならずあるはずなので、なければエラー
                                #    output("Error!!! predict is None! score:", now_score_tmp)
                                #    sys.exit(1)
                            else:
                                if c.DIRECT_FLG:
                                    tmp_label.append(aft)
                                else:
                                    if OUTPUT_TYPE == "d":
                                        l_list = []
                                        for tmp_k in self.opt_lists:
                                            l_list.append(get_decimal_multi(get_divide(opt_bef[tmp_k], opt_aft[tmp_k]), c.OUTPUT_MULTI ))
                                        tmp_label.append(l_list)

                                    elif OUTPUT_TYPE == "sub":
                                        l_list = []
                                        for tmp_k in self.opt_lists:
                                            l_list.append(get_sub(opt_bef[tmp_k], opt_aft[tmp_k], c.OUTPUT_MULTI))
                                        tmp_label.append(l_list)

                                    else:
                                        output("WRONG OUTPUT_TYPE!!!", OUTPUT_TYPE)
                                        exit(1)

                    if c.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN", "REGRESSION","REGRESSION_OCOPS", "CATEGORY_OCOPS"]:
                        if c.ANSWER_DB != "" and self.test_flg and self.eval_flg == False:
                            answer_rate_up = ifd_data[str(c.TERM) + "-bp"]
                            answer_score_up = ifd_data["bds"]  # 決済時のスコア
                            answer_rate_dw = ifd_data[str(c.TERM) + "-sp"]
                            answer_score_dw = ifd_data["sds"]  # 決済時のスコア

                            if answer_rate_up == None or answer_score_up == None or answer_rate_dw == None or answer_score_dw == None:
                                output("answer_rate or answer_score is null!!  score:", now_score_tmp)
                                exit(1)

                            answer_rate = [answer_rate_up, answer_rate_dw]
                            answer_score = [answer_score_up, answer_score_dw]

                    if c.SEC_OH_LEN_FIX_FLG:
                        sec_oh = sec
                    else:
                        if c.BET_SHIFT < 1:
                            sec_oh = sec
                        else:
                            sec_oh = int(Decimal(str(sec)) / Decimal(str(c.BET_SHIFT)) )  # 2秒間隔データなら０から29に変換しなければならないのでbet_termで割る

                    if (c.METHOD in ["LSTM3","LSTM4","LSTM5","LSTM6","LSTM7","LSTM8","LSTM10"]) == False:
                        #必要ない変数はNoneにしてメモリ節約
                        sec_oh = None
                        minute = None
                        hour = None
                        week = None

                    if test_flg and eval_flg == False:
                        if c.INCLUDE_HL_FLG:
                            #予想結果までの最安値を取得
                            tmp_hl_data = self.hl_dict.get(now_score_tmp)
                            if tmp_hl_data != None:
                                highest_rate = tmp_hl_data["h"]
                                lowest_rate = tmp_hl_data["l"]
                        else:
                            highest_rate = None
                            lowest_rate = None

                        # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                        # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                        # 複数のDBを使用した場合に結果を時系列順にならべて確認するため
                        if len(self.train_dict) == 0:
                            output("first score", now_score_tmp)
                        self.train_dict[now_score_tmp] = [tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp,
                                                         db5_index_tmp, prev_close, now_score_tmp, predict_tmp[i], bef, sec_oh, minute, hour, week, db_volume_index_tmp, db_extra_1_index_tmp, tmp_opt, ind_foot_list, oanda_ord_list, oanda_pos_list, hor_val,db_foot_idxs, highlow_val, aft, spread_t, divide_prev, divide_aft, spread_t_end,
                                                         answer_rate, answer_score, highest_rate, lowest_rate]
                    else:
                        if train_list_minimum_flg:
                            self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp, prev_close, now_score_tmp])
                        else:
                            self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp, prev_close, now_score_tmp, predict_tmp[i], bef, sec_oh, minute, hour, week, db_volume_index_tmp, db_extra_1_index_tmp, tmp_opt, ind_foot_list, oanda_ord_list, oanda_pos_list, hor_val, db_foot_idxs, highlow_val ])

                    del tmp_label

            if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                # メモリ節約のためDBは削除する
                r.flushdb()

            # メモリ空き容量を取得
            #output("before db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
            # メモリ節約のためredis停止
    
            if redis_stop:
                #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
                sudo_password = 'Reikou0129'
                command = 'systemctl stop redis'.split()
    
                p = subprocess.Popen(['sudo', '-S'] + command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
                sudo_prompt = p.communicate(sudo_password + '\n')[1]
    
                # メモリ空き容量を取得

            if test_flg and eval_flg == False:
                #一番短い足のDBを複数インプットする場合用に
                #score順にならべかえてtrain_listと正解ラベルおよびレート変化幅(aft-bef)をそれぞれリスト化する
                for data in sorted(self.train_dict.items()):
                    self.train_list.append(data[1][:23])
                    self.correct_list.append(data[1][0])
                    self.pred_close_list.append(data[1][9])
                    self.real_close_list.append(data[1][23])
                    self.target_spread_list.append(data[1][24])
                    self.target_divide_prev_list.append(data[1][25])
                    self.target_divide_aft_list.append(data[1][26])
                    self.target_spread_end_list.append(data[1][27])
                    self.target_answer_rate_list.append(data[1][28])
                    self.target_answer_score_list.append(data[1][29])
                    self.target_highest_close_list.append(data[1][30])
                    self.target_lowest_close_list.append(data[1][31])
                    self.train_score_list.append(data[0])
                    self.target_predict_list.append(data[1][8])

                #一番短い足のDBを複数インプットする場合用に
                #scoreのリストcloseのリストを作成 結果確認のグラフ描写で使うため
                for data in sorted(self.score_dict.items()):
                    self.score_list.append(data[0])
                    self.close_list.append(data[1]["c"])
                    self.spread_list.append(data[1]["spr"])
                    self.atr_list.append(data[1]["atr"])
                    self.jpy_list.append(data[1]["jpy"])
                    self.spread_percent_list.append(data[1]["sp"])
                    self.ind_list.append(data[1]["ind"])

                    if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                        self.tick_list.append(data[1]["tk"])
                    for tmp_k in self.opt_lists:
                        self.output_dict[tmp_k].append(data[1][tmp_k])

                #train対象のスコアと対象外のスコアを結合
                self.train_dict_ex.update(self.train_dict)
                cnt_ex = 0
                for data in sorted(self.train_dict_ex.items()):
                    if data[1] == None:
                        #train対象外のスコアは-1をいれる
                        self.train_list_idx.append(-1)
                    else:
                        #train対象であればscore順のtrain_dictのインデックスをいれる
                        self.train_list_idx.append(cnt_ex)
                        cnt_ex += 1

                del self.train_dict, self.score_dict

            output("before del ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
            # メモリ解放
            del close_tmp
            del score_tmp, predict_tmp,
            del self.hor_dict, self.oanda_ord_dict, self.oanda_pos_dict , self.ind_foot_dict
            del self.ind_score_dict,self.atr_dict, self.db_foots_score

            del self.db2_score, self.db2_score_list
            del self.db3_score, self.db3_score_list
            del self.db4_score, self.db4_score_list
            del self.db5_score, self.db5_score_list

            output("after del ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            #numpy化してsliceを早くする
            self.db1 = self.make_input_data_list_np(self.db1, c.INPUT_DATAS[0])

            self.db_volume_np = np.array(self.db_volume)
            del self.db_volume

            if c.METHOD == "LSTM9":
                self.db_pred_np = np.array(self.db_preds[self.db9_name])

            if c.DB_EXTRA_1 != "":
                self.db_extra_1_np = np.array(self.db_extra_1)
            del self.db_extra_1

            self.data_length = len(self.train_list)
            self.cut_num = (self.data_length) % self.batch_size

            output(datetime.now(), "data length:" , self.data_length)
            output(datetime.now(), "ocoa_skip_cnt:", self.ocoa_skip_cnt)

            if self.cut_num !=0:
                # エポック毎の学習ステップ数
                # バッチサイズが1ステップで学習するデータ数
                # self.cut_num(総学習データ÷バッチサイズ)で余りがあるなら余りがある分ステップ数が1多い
                self.steps_per_epoch = int(self.data_length / self.batch_size) +1
                if self.drop_last:
                    self.steps_per_epoch = int(self.data_length / self.batch_size)
            else:
                self.steps_per_epoch = int(self.data_length / self.batch_size)

            output("steps_per_epoch: ", self.steps_per_epoch)

            if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFD", "CATEGORY_BIN_DW_IFD",
                                   "CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO","CATEGORY_BIN_UP_TP", "CATEGORY_BIN_DW_TP",
                                   "CATEGORY_BIN_UP_OCO","CATEGORY_BIN_DW_OCO","CATEGORY_BIN_UP_OCOA","CATEGORY_BIN_DW_OCOA",
                                   "REGRESSION_UP", "REGRESSION_DW",
                                   ] and take_profit_cnt != 0:
                output("take profit rate:", take_profit_cnt/self.data_length)

            if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFDSF","CATEGORY_BIN_DW_IFDSF"] and stop_loss_cnt != 0:
                output("stop loss rate:", stop_loss_cnt/self.data_length)

            if c.LEARNING_TYPE == "CATEGORY_OCOPS":
                output("category_ocops_up_cnt",self.category_ocops_up_cnt)
                output("category_ocops_dw_cnt",self.category_ocops_dw_cnt)
                output("category_ocops_same_cnt",self.category_ocops_cnt - (self.category_ocops_up_cnt + self.category_ocops_dw_cnt))

            if up != 0 and down != 0:
                output("UP: ",up/self.data_length)
                if self.data_length - up - down != 0:
                    output("SAME: ", (self.data_length - up - down) / (self.data_length))
                output("DOWN: ",down/self.data_length)

            #Spreadの内訳を表示
            if test_flg and eval_flg == False:
                if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
                    output("spread total: ", self.spread_cnt)
                    if self.spread_cnt != 0:
                        for k, v in sorted(self.spread_cnt_dict.items()):
                            output(k, v, v / self.spread_cnt)


            """
            if tmp_k in self.opt_lists:
                ans_d_list = np.array(self.output_answer_dict[tmp_k])
                output("結果までの" + tmp_k + "の絶対差")
                output("avg:", np.average(ans_d_list))
                output("std:", np.std(ans_d_list))
                output("max:", np.max(ans_d_list))
            """
            if self.test_flg == False and c.DATA_SHUFFLE == "SHUFFLE":
                    #シャッフルする
                    random.seed(c.SEED)
                    random.shuffle(self.train_list)

            self.add_method()

            """
            self.create_db = np.vectorize(self.tmp_create_db, otypes=[np.ndarray])
            self.create_db_all = np.vectorize(self.tmp_create_db_all, otypes=[np.ndarray])
            self.create_label = np.vectorize(self.tmp_create_label, otypes=[np.ndarray])
            self.create_volume = np.vectorize(self.tmp_create_volume, otypes=[np.ndarray])
            self.create_predict = np.vectorize(self.tmp_create_predict, otypes=[np.ndarray])
            self.create_now_rate = np.vectorize(self.tmp_create_now_rate, otypes=[np.ndarray])
            self.create_sec = np.vectorize(self.tmp_create_sec, otypes=[np.ndarray])
            self.create_min = np.vectorize(self.tmp_create_min, otypes=[np.ndarray])
            self.create_hour = np.vectorize(self.tmp_create_hour, otypes=[np.ndarray])
            self.create_week = np.vectorize(self.tmp_create_week, otypes=[np.ndarray])
            self.create_pred = np.vectorize(self.tmp_create_pred, otypes=[np.ndarray])
            self.create_pred_close = np.vectorize(self.tmp_create_pred_close, otypes=[np.ndarray])
            self.create_db_extra = np.vectorize(self.tmp_create_db_extra, otypes=[np.ndarray])
            self.create_option = np.vectorize(self.tmp_create_option, otypes=[np.ndarray])
            self.create_ind_foot = np.vectorize(self.tmp_create_ind_foot, otypes=[np.ndarray])
            self.create_oanda_ord = np.vectorize(self.tmp_create_oanda_ord, otypes=[np.ndarray])
            self.create_oanda_pos = np.vectorize(self.tmp_create_oanda_pos, otypes=[np.ndarray])
            self.create_hor = np.vectorize(self.tmp_create_hor, otypes=[np.ndarray])
            self.create_foot_db = np.vectorize(self.tmp_create_foot_db, otypes=[np.ndarray])
            self.create_highlow = np.vectorize(self.tmp_create_highlow, otypes=[np.ndarray])
            self.create_non_lstm = np.vectorize(self.tmp_create_non_lstm, otypes=[np.ndarray])
        """

        except Exception as e:
            print("error occured")
            print(tracebackPrint(e))
            mail.send_message(host, ": DataSequence2 error!!!")

    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        start_time = time.perf_counter()
        # データの取得実装
        #output("idx:", idx)
        if idx == 0:
            if self.test_flg == False:
                tmp_l = []
                for j in range(20):
                    tmp_l.append(self.train_list[j][1])
                output(tmp_l)
        tmp_np = np.arange(self.batch_size)

        if self.drop_last == False:
            # self.cut_num(総学習データ÷バッチサイズ)で余りがあり、且つ最後のステップの場合、
            # リターンする配列のサイズはバッチサイズでなく、余りの数となる
            if idx == (self.steps_per_epoch -1) and self.cut_num != 0:
                tmp_np = np.arange(self.cut_num)

        #output("tmp_np:", tmp_np)
        #tmp_np = np.arange(3)
        #tmp = self.g(tmp_np)
        #new_tmp = tmp.tolist()
        #output(np.array(new_tmp).shape)

        # start_idxからバッチサイズ分取得開始インデックスをずらす
        start_idx = idx * self.batch_size

        retY = {}
        for i, tmp_y in enumerate(self.c.RETURN_Y_LIST):

            label_data_tmp= self.create_label(tmp_np, start_idx, i)
            label_data_tmp = label_data_tmp.tolist()
            retY[tmp_y] = np.array(label_data_tmp)

        if self.return_all == False:
            retY = retY[self.c.GET_Y_STR]

        retX = []

        if self.c.METHOD == "LSTM" or self.c.METHOD == "BY" or self.c.METHOD == "LSTM2" or self.c.METHOD == "LSTM3" or self.c.METHOD == "LSTM4" or self.c.METHOD == "LSTM5" or \
                self.c.METHOD == "LSTM6" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or self.c.METHOD == "LSTM9" or self.c.METHOD == "LSTM10" or \
                self.c.METHOD == "TCN" or self.c.METHOD == "TCN7":

            for i in range(len(self.c.INPUT_LEN)):
                if i == 0 and self.c.DB1_NOT_LEARN:
                    #DB1が学習対象でないならスキップ
                    continue

                ipt_data = self.c.INPUT_DATAS[i]
                ipt_list = ipt_data.split("_")
                if self.c.INPUT_SEPARATE_FLG == False or ipt_data == "" or len(ipt_list) == 1:
                    if self.c.FRAGMENT_NUM != None:
                        tmp_length = self.c.FRAGMENTS_INPUT_LEN
                    else:
                        tmp_length = self.c.INPUT_LEN[i]

                    for a in self.c.ADDITIONAL_DATA_LIST:
                        tmp_length += a["input_len"]

                    tmpX = np.zeros((len(tmp_np), tmp_length, len(ipt_list)), dtype=self.c.DTYPE)

                    for idx, ipt in enumerate(ipt_list):
                        tmp_arr = self.create_db(self.c, tmp_np, start_idx, i + 1, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr,dtype=self.c.DTYPE)
                        tmpX[:, :, idx] = tmp_arr[:]

                    retX.append(tmpX)
                else:
                    for ipt in ipt_list:
                        if self.c.FRAGMENT_NUM != None:
                            tmp_length = self.c.FRAGMENTS_INPUT_LEN
                        else:
                            tmp_length = self.c.INPUT_LEN[i]

                        for a in self.c.ADDITIONAL_DATA_LIST:
                            tmp_length += a["input_len"]

                        tmpX = np.zeros((len(tmp_np), tmp_length, 1),dtype=self.c.DTYPE)

                        tmp_arr = self.create_db(self.c, tmp_np, start_idx, i + 1, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr,dtype=self.c.DTYPE)
                        tmpX[:, :, 0] = tmp_arr[:]

                        retX.append(tmpX)

            for db_tmp in self.c.FOOT_DBS:
                d_term, d_len, d_unit, d_x, db_name, separate_flg = db_tmp
                ipt_list_foot = d_x.split("_")
                if self.c.INPUT_SEPARATE_FLG == False or len(ipt_list_foot) == 1:
                    tmpX = np.zeros((len(tmp_np), d_len, len(ipt_list_foot)))
                    for idx, ipt in enumerate(ipt_list_foot):
                        tmp_arr = self.create_foot_db(tmp_np, start_idx, d_term, d_len, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr)
                        tmpX[:, :, idx] = tmp_arr[:]

                    retX.append(tmpX)
                else:
                    for ipt in ipt_list_foot:
                        tmpX = np.zeros((len(tmp_np), d_len, 1))

                        tmp_arr = self.create_foot_db(tmp_np, start_idx, d_term, d_len, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr)
                        tmpX[:, :, 0] = tmp_arr[:]

                        retX.append(tmpX)


            if self.c.METHOD == "LSTM2":
                predict_data_tmp = self.create_predict(tmp_np, start_idx)
                predict_data_tmp = predict_data_tmp.tolist()
                retX.append(np.array(predict_data_tmp))

            if self.c.METHOD == "LSTM3" or self.c.METHOD == "LSTM6" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or self.c.METHOD == "LSTM9" or \
                    self.c.METHOD == "LSTM10" or self.c.METHOD == "TCN7":
                sec_data_tmp = self.create_sec(tmp_np, start_idx)
                sec_data_tmp = sec_data_tmp.tolist()
                retX.append(np.identity(self.c.SEC_OH_LEN,dtype='int8')[sec_data_tmp])

            if self.c.METHOD == "LSTM4" or self.c.METHOD == "LSTM5" or self.c.METHOD == "LSTM6" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or \
                    self.c.METHOD == "LSTM9" or self.c.METHOD == "LSTM10" or self.c.METHOD == "TCN7":
                min_data_tmp = self.create_min(tmp_np, start_idx)
                min_data_tmp = min_data_tmp.tolist()
                retX.append(np.identity(self.c.MIN_OH_LEN,dtype='int8')[min_data_tmp])

            if self.c.METHOD == "LSTM5" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or self.c.METHOD == "LSTM9" or \
                    self.c.METHOD == "TCN7" or self.c.METHOD == "LSTM10":
                hour_data_tmp = self.create_hour(tmp_np, start_idx)
                hour_data_tmp = hour_data_tmp.tolist()
                retX.append(np.identity(self.c.HOUR_OH_LEN,dtype='int8')[hour_data_tmp])

            if self.c.METHOD == "LSTM10":
                week_data_tmp = self.create_week(tmp_np, start_idx)
                week_data_tmp = week_data_tmp.tolist()
                retX.append(np.identity(self.c.WEEK_OH_LEN,dtype='int8')[week_data_tmp])

            for i, hor in enumerate(self.c.HOR_DB_CORE_LIST):
                for j in range(self.c.HOR_LINE_NUM):
                    data_tmp = self.create_hor(tmp_np, start_idx, i, j)
                    data_tmp = data_tmp.tolist()
                    retX.append(np.array(data_tmp))

            if self.c.HIGHLOW_DB_CORE != "":
                for i in range(self.c.HIGHLOW_DATA_NUM):
                    data_tmp = self.create_highlow(tmp_np, start_idx, i)
                    data_tmp = data_tmp.tolist()
                    retX.append(np.array(data_tmp))

            if len(self.c.NON_LSTM_LIST) != 0:
                for i in self.c.NON_LSTM_LIST:
                    tmp_db_no = i["db_no"]
                    tmp_inputs = i["inputs"]
                    tmp_length = i["length"]
                    for ipt in tmp_inputs:
                        data_tmp = self.create_non_lstm(tmp_np, start_idx, tmp_db_no, ipt, tmp_length)
                        data_tmp = data_tmp.tolist()
                        retX.append(np.array(data_tmp))

            if self.c.OANDA_ORD_DB != "":
                i_num = int(self.c.OANDA_ORD_NUM * 2 + 1)
                for i in range(i_num):
                    data_tmp = self.create_oanda_ord(tmp_np, start_idx, i)
                    data_tmp = data_tmp.tolist()
                    retX.append(np.array(data_tmp))

            if self.c.OANDA_POS_DB != "":
                i_num = int(self.c.OANDA_POS_NUM * 2 + 1)
                for i in range(i_num):
                    data_tmp = self.create_oanda_pos(tmp_np, start_idx, i)
                    data_tmp = data_tmp.tolist()
                    retX.append(np.array(data_tmp))

            if len(self.c.IND_FOOT_COL) != 0:
                for j in range(len(self.c.IND_FOOT_COL)):
                    data_tmp = self.create_ind_foot(tmp_np, start_idx, j)
                    data_tmp = data_tmp.tolist()
                    retX.append(np.array(data_tmp))

            if self.c.METHOD == "LSTM8":
                volume_data_tmp = self.create_volume(self.c, tmp_np, start_idx)
                volume_data_tmp = volume_data_tmp.tolist()
                retX.append(np.array(volume_data_tmp))

            if self.c.METHOD == "LSTM9":
                for ipt9 in self.c.LSTM9_INPUTS:
                    pred_data_tmp = self.create_pred(tmp_np, start_idx, ipt9)
                    pred_data_tmp = pred_data_tmp.tolist()
                    retX.append(np.array(pred_data_tmp))

                    if self.c.LSTM9_USE_CLOSE:
                        pred_close_data_tmp = self.create_pred_close(tmp_np, start_idx, ipt9)
                        pred_close_data_tmp = pred_close_data_tmp.tolist()
                        retX.append(np.array(pred_close_data_tmp))

            if self.c.DB_EXTRA_1 != "":
                tmpX = np.zeros((len(tmp_np), self.c.DB_EXTRA_1_LEN, 1))
                extra_data_tmp = self.create_db_extra(self.c, tmp_np, start_idx)
                extra_data_tmp = extra_data_tmp.tolist()
                extra_data_tmp = np.array(extra_data_tmp)
                tmpX[:, :, 0] = extra_data_tmp[:]

                retX.append(tmpX)

            if self.c.NOW_RATE_FLG == True:
                rate_data_tmp = self.create_now_rate(tmp_np, start_idx)
                rate_data_tmp = rate_data_tmp.tolist()
                retX.append(np.array(rate_data_tmp))

            if len(self.c.OPTIONS) != 0:
                for j in range(len(self.c.OPTIONS)):
                    tmp_option = self.create_option(tmp_np, start_idx, j)
                    opt_data_tmp = tmp_option.tolist()
                    retX.append(np.array(opt_data_tmp))

        elif self.c.METHOD == "NORMAL":

            for i in range(len(self.c.INPUT_LEN)):
                ipt_data = self.c.INPUT_DATAS[i]
                ipt_list = ipt_data.split("_")
                tmpX = np.zeros((len(tmp_np), self.c.INPUT_LEN[i], len(ipt_list)))

                for idx, ipt in enumerate(ipt_list):
                    tmp_arr = self.create_db(self.c, tmp_np, start_idx, i + 1, ipt)
                    tmp_arr = tmp_arr.tolist()
                    tmp_arr = np.array(tmp_arr)
                    tmpX[:, :, idx] = tmp_arr[:]

                retX.append(tmpX)

        """
        #テストデータを一件表示する
        if self.test_flg and self.eval_flg == False:
            if idx == 0:
                output("test data number1 :", retX[0])
        """
        end_time = time.perf_counter()
        get_take = end_time - start_time
        self.get_time_list.append(get_take)

        #テストの場合はテストデータのみ返す
        if self.test_flg:
            if self.eval_flg:
                return retX, retY
            else:
                return retX
        else:
            return retX, retY

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch


    def get_data_length(self):
        return self.data_length

    def get_steps_per_epoch(self, batch_size, drop_last):
        cut_num = (self.data_length) % batch_size

        if cut_num !=0:
            if drop_last:
                #全てのバッチサイズを揃えて、端数を持つバッチは作らない場合
                steps_per_epoch = int(self.data_length / batch_size)
            else:
                steps_per_epoch = int(self.data_length / batch_size) +1
        else:
            steps_per_epoch = int(self.data_length / batch_size)

        return steps_per_epoch

    def rotate_train_list(self):
        if self.test_flg == False:
            if self.c.DATA_SHUFFLE == "ROTATE":
                rotate_num = int(len(self.train_list) / self.c.EPOCH)
                self.train_list = rotate(self.train_list, rotate_num)

            elif self.c.DATA_SHUFFLE == "SHUFFLE":
                random.seed(self.c.SEED)
                random.shuffle(self.train_list)

    def on_epoch_end(self):
        self.epoch_cnt += 1

        self.rotate_train_list()

        output("get_time_len:", len(self.get_time_list))
        if len(self.get_time_list) != 0:
            output("get_time_avg:", sum(self.get_time_list)/len(self.get_time_list))
            output("get_time_sum:", sum(self.get_time_list))


        self.get_time_list = []

    def get_data_length(self):
        return self.data_length

    def get_correct_list(self):
        retY = np.array(self.correct_list)
        return retY

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

    def get_spread_percent_list(self):
        return self.spread_percent_list

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

    def get_hor_list(self):
        return self.hor_list

    def get_atr_list(self):
        return self.atr_list

    def get_ind_list(self):
        return self.ind_list

    def get_output_dict(self):
        return self.output_dict

    def set_db9_name(self, db_name):
        self.db9_name = db_name
        self.db_pred_np = np.array(self.db_preds[self.db9_name])

    def reset_db9_name(self, db_name):
        self.db_pred_np = None

        self.db9_name = db_name
        # testLstm用
        r = redis.Redis(host='localhost', port=6379, db=self.db_no, decode_responses=True)
        for i, db in enumerate(self.c.DB1_LIST):
            result = r.zrangebyscore(db, self.start_score, self.end_score, withscores=True)

            for line in result:
                body = line[0]
                tmps = json.loads(body)

                if self.c.METHOD == "LSTM9":
                    if tmps.get(self.db9_name) != None:
                        self.db_pred.append(tmps.get(self.db9_name))
                    else:
                        self.db_pred.append(-1)

        self.db_pred_np = np.array(self.db_pred)
        self.db_pred = None

    def make_input_data_list(self, ipt_list_str):
        return_dict = {}
        ipt_list = ipt_list_str.split("_")
        for ipt in ipt_list:
            return_dict[ipt] = []

        return return_dict

    def make_input_data_list_np(self, org_dict, ipt_list_str):
        return_dict = {}
        ipt_list = ipt_list_str.split("_")
        for ipt in ipt_list:
            return_dict[ipt] = np.array(org_dict[ipt],dtype=self.c.DTYPE)

        return return_dict


    def change_eval_flg(self, eval_flg):
        self.eval_flg = eval_flg

    def add_method(self):
        self.create_db = np.vectorize(self.tmp_create_db, otypes=[np.ndarray])
        self.create_db_all = np.vectorize(self.tmp_create_db_all, otypes=[np.ndarray])
        self.create_label = np.vectorize(self.tmp_create_label, otypes=[np.ndarray])
        self.create_volume = np.vectorize(self.tmp_create_volume, otypes=[np.ndarray])
        self.create_predict = np.vectorize(self.tmp_create_predict, otypes=[np.ndarray])
        self.create_now_rate = np.vectorize(self.tmp_create_now_rate, otypes=[np.ndarray])
        self.create_sec = np.vectorize(self.tmp_create_sec, otypes=[np.ndarray])
        self.create_min = np.vectorize(self.tmp_create_min, otypes=[np.ndarray])
        self.create_hour = np.vectorize(self.tmp_create_hour, otypes=[np.ndarray])
        self.create_week = np.vectorize(self.tmp_create_week, otypes=[np.ndarray])
        self.create_pred = np.vectorize(self.tmp_create_pred, otypes=[np.ndarray])
        self.create_pred_close = np.vectorize(self.tmp_create_pred_close, otypes=[np.ndarray])
        self.create_db_extra = np.vectorize(self.tmp_create_db_extra, otypes=[np.ndarray])
        self.create_option = np.vectorize(self.tmp_create_option, otypes=[np.ndarray])
        self.create_ind_foot = np.vectorize(self.tmp_create_ind_foot, otypes=[np.ndarray])
        self.create_oanda_ord = np.vectorize(self.tmp_create_oanda_ord, otypes=[np.ndarray])
        self.create_oanda_pos = np.vectorize(self.tmp_create_oanda_pos, otypes=[np.ndarray])
        self.create_hor = np.vectorize(self.tmp_create_hor, otypes=[np.ndarray])
        self.create_foot_db = np.vectorize(self.tmp_create_foot_db, otypes=[np.ndarray])
        self.create_highlow = np.vectorize(self.tmp_create_highlow, otypes=[np.ndarray])
        self.create_non_lstm = np.vectorize(self.tmp_create_non_lstm, otypes=[np.ndarray])

    def delete_method(self):
        del self.create_db
        del self.create_db_all
        del self.create_label
        del self.create_volume
        del self.create_predict
        del self.create_now_rate
        del self.create_sec
        del self.create_min
        del self.create_hour
        del self.create_week
        del self.create_pred
        del self.create_pred_close
        del self.create_db_extra
        del self.create_option
        del self.create_ind_foot
        del self.create_oanda_ord
        del self.create_oanda_pos
        del self.create_hor
        del self.create_foot_db
        del self.create_highlow
        del self.create_non_lstm

        output(datetime.now(), "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        gc.collect()
        output(datetime.now(), "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    def delete_db(self):
        del self.db1, self.db2, self.db3, self.db4, self.db5,self.train_list

        self.db1 = None
        self.db2 = None
        self.db3 = None
        self.db4 = None
        self.db5 = None

        self.train_list = None



        output(datetime.now(), "before GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        gc.collect()
        output(datetime.now(), "after GC", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

    def tmp_create_db(self, c, x, start_idx, db_no, ipt):
        target_idx = self.train_list[start_idx + x][db_no]
        now_score = self.train_list[start_idx + x][7]
        now_close = self.train_list[start_idx + x][6]
        tmp_return = []

        for i, a in enumerate(c.ADDITIONAL_DATA_LIST):
            input_type = a["input_type"]

            if a["length"] == "H1":
                target_score = get_decimal_sub(now_score, get_decimal_mod(now_score, 3600))  # 自分の所属する足のスコア
            elif a["length"] == "M1":
                target_score = get_decimal_sub(now_score, get_decimal_mod(now_score, 60))  # 自分の所属する足のスコア

            if input_type == "c":
                target_idx =  self.additional_data_list[i]["score_dict"][target_score]  # 自分の所属する足のindexを取得する
                target_idx_list = []
                # データ取得用インデックス作成
                for f in a["fragments_list"]:
                    target_idx_list.append(int(get_decimal_add(target_idx, f)))

                data_list = self.additional_data_list[i]["data_list"][target_idx_list].tolist()
                tmp_return.extend(data_list)

            elif input_type == "sma":
                if a["length"] == "H1":
                    target_score = get_decimal_sub(target_score, 3600)  # 自分の所属する足の一つ前
                elif a["length"] == "M1":
                    target_score = get_decimal_sub(target_score, 60)  # 自分の所属する足の一つ前

                target_data = self.additional_data_list[i]["data_dict"].get(target_score)  # 自分の所属する足の一つ前を取得する

                data_list = []
                for f in a["fragments_list"]:
                    data_list.append(float(target_data[input_type + f]))

                tmp_return.extend(data_list)

        #if start_idx == 0:
        #    print(target_score,tmp_return)

        if c.FRAGMENT_NUM != None:
            #データ取得用インデックス作成
            idx_list = []
            for f in self.fragments_list:
                idx_list.append(int(get_decimal_add(target_idx, f)))

        if self.same_db_flg == False:
            if db_no == 1:
                if c.FRAGMENT_NUM != None:
                    tmp_data_list = self.db1[ipt][idx_list]
                else:
                    tmp_data_list = self.db1[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

            elif db_no == 2:
                if c.FRAGMENT_NUM != None:
                    tmp_data_list = self.db2[ipt][idx_list]
                else:
                    tmp_data_list = self.db2[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

            elif db_no == 3:
                if c.FRAGMENT_NUM != None:
                    tmp_data_list = self.db3[ipt][idx_list]
                else:
                    tmp_data_list = self.db3[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

            elif db_no == 4:
                if c.FRAGMENT_NUM != None:
                    tmp_data_list = self.db4[ipt][idx_list]
                else:
                    tmp_data_list = self.db4[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

            elif db_no == 5:
                if c.FRAGMENT_NUM != None:
                    tmp_data_list = self.db5[ipt][idx_list]
                else:
                    tmp_data_list = self.db5[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

        else:
            if c.FRAGMENT_NUM != None:
                tmp_data_list = self.db1[ipt][idx_list]
            else:
                tmp_data_list = self.db1[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

        tmp_return.extend(tmp_data_list.tolist())
        tmp_return = np.array(tmp_return)

        if c.FRAGMENT_NUM != None:
            now_c = now_close
            tmp_arr = tmp_return
            tmp_return = []
            arr_length = len(tmp_arr)
            f_type, multi = c.FRAGMENTS_INPUT_TYPE.split("-")
            multi = int(multi)

            if f_type == "div" or f_type == "sub":
                for ind, l in enumerate(tmp_arr):
                    if arr_length == ind + 1:
                        break
                    else:
                        if f_type == "div":
                            tmp_return.append(get_divide(l, now_c, multi=multi))
                        elif f_type == "sub":
                            tmp_return.append(get_sub(l, now_c, multi=multi))

            elif f_type == "div1" or f_type == "sub1":
                if f_type == "div1":
                    d_bef = np.roll(tmp_arr, 1)
                    tmp_return = get_divide_arr(d_bef, tmp_arr, multi=multi).tolist()
                    tmp_return.pop(0)
                elif f_type == "sub1":
                    d_bef = np.roll(tmp_arr, 1)
                    tmp_return = get_sub_arr(d_bef, tmp_arr, multi=multi).tolist()
                    tmp_return.pop(0)

            else:
                print("f_type invalid")
                exit(1)

            tmp_return = np.array(tmp_return)

        if (ipt != "d-1" and ipt != "d1" and ipt != "md1-1" and ipt != "ehd1-1" and ipt != "eld1-1" and ipt != "wmd1-1") and self.epoch_cnt  == 0:
            #Noneやnanがはいっていたらエラー
            if np.any(tmp_return == None) or np.any(np.isnan(tmp_return)):
                output("return is None or nan!!!")
                output("target_idx:", target_idx)
                exit()

        return tmp_return

    def tmp_create_db_all(self, c, x, start_idx, db_no):
        target_idx = self.train_list[start_idx + x][db_no]

        if db_no == 1:
            base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db1_np[target_idx - 1])
            tmp_arr = self.db1_np[target_idx - c.INPUT_LEN[db_no - 1] - 1:target_idx - 1]
            return (tmp_arr / base_arr - 1) * 10000

        elif db_no == 2:
            base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db2_np[target_idx - 1])
            tmp_arr = self.db2_np[target_idx - c.INPUT_LEN[db_no - 1] - 1:target_idx - 1]
            return (tmp_arr / base_arr - 1) * 10000
        elif db_no == 3:
            base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db3_np[target_idx - 1])
            tmp_arr = self.db3_np[target_idx - c.INPUT_LEN[db_no - 1] - 1:target_idx - 1]
            return (tmp_arr / base_arr - 1) * 10000
        elif db_no == 4:
            base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db4_np[target_idx - 1])
            tmp_arr = self.db4_np[target_idx - c.INPUT_LEN[db_no - 1] - 1:target_idx - 1]
            return (tmp_arr / base_arr - 1) * 10000
        elif db_no == 5:
            base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db5_np[target_idx - 1])
            tmp_arr = self.db5_np[target_idx - c.INPUT_LEN[db_no - 1] - 1:target_idx - 1]
            return (tmp_arr / base_arr - 1) * 10000



    def tmp_create_label(self, x, start_idx, idx):
        return self.train_list[start_idx + x][0][idx]

    def tmp_create_predict(self, x, start_idx):
        return self.train_list[start_idx + x][8]

    def tmp_create_now_rate(self, x, start_idx):
        return self.train_list[start_idx + x][9]

    def tmp_create_sec(self, x, start_idx):
        # secをOne-Hotベクトルに変換
        # return np.identity(SEC_OH_LEN)[self.train_list[start_idx + x][10]]

        return self.train_list[start_idx + x][10]

    def tmp_create_min(self, x, start_idx):
        return self.train_list[start_idx + x][11]

    def tmp_create_hour(self, x, start_idx):
        return self.train_list[start_idx + x][12]

    def tmp_create_week(self, x, start_idx):
        return self.train_list[start_idx + x][13]

    def tmp_create_volume(self, c, x, start_idx):
        target_idx = self.train_list[start_idx + x][14]
        vol_arr = self.db_volume_np[target_idx - c.DB_VOLUME_INPUT_LEN:target_idx]
        ret = 0
        for vol in vol_arr:
            ret += vol

        return ret

    def tmp_create_pred(self, x, start_idx, len):
        target_idx = self.train_list[start_idx + x][1]
        return self.db_pred_np[target_idx - len]


    def tmp_create_pred_close(self, x, start_idx, len):
        target_idx = self.train_list[start_idx + x][1]

        bef = self.db_close[target_idx - 1 - len]
        aft = self.db_close[target_idx - 1]

        divide_org = aft / bef
        if aft == bef:
            divide_org = 1

        divide = 10000 * (divide_org - 1)
        return divide

    def tmp_create_db_extra(self, c, x, start_idx):
        target_idx = self.train_list[start_idx + x][15]
        return self.db_extra_1_np[target_idx - c.DB_EXTRA_1_LEN:target_idx]

    def tmp_create_option(self, x, start_idx, num):
        return self.train_list[start_idx + x][16][num]

    def tmp_create_ind_foot(self, x, start_idx, num):
        try:
            tmp_val = self.train_list[start_idx + x][17][num]
        except Exception as e:
            output(self.train_list[start_idx + x])
            exit(1)

        return tmp_val

    def tmp_create_oanda_ord(self, x, start_idx, num):
        try:
            return self.train_list[start_idx + x][18][num]
        except Exception as e:
            output(self.train_list[start_idx + x][18])
            exit(1)

    def tmp_create_oanda_pos(self, x, start_idx, num):

        return self.train_list[start_idx + x][19][num]

    def tmp_create_hor(self, x, start_idx, i, j):
        tmp_val = self.train_list[start_idx + x][20][i][j]
        return tmp_val

    def tmp_create_foot_db(self, x, start_idx, db_term, db_len, ipt):
        try:
            target_idx = self.train_list[start_idx + x][21][db_term]
        except Exception as e:
            output(self.train_list[start_idx + x][21])
            exit(1)

        target_idx_end = target_idx + 1
        tmp_return = self.db_foots[db_term][ipt][target_idx_end - db_len: target_idx_end]

        return tmp_return

    def tmp_create_highlow(self, x, start_idx, i):
        tmp_val = self.train_list[start_idx + x][22][i]
        return tmp_val

    def tmp_create_non_lstm(self, x, start_idx, db_no, ipt, i):
        target_idx = self.train_list[start_idx + x][db_no]
        tmp_return = None

        if db_no == 1:
            tmp_return = self.db1[ipt][target_idx - i:target_idx]
        elif db_no == 2:
            tmp_return = self.db2[ipt][target_idx - i:target_idx]
        elif db_no == 3:
            tmp_return = self.db3[ipt][target_idx - i:target_idx]
        elif db_no == 4:
            tmp_return = self.db4[ipt][target_idx - i:target_idx]
        elif db_no == 5:
            tmp_return = self.db5[ipt][target_idx - i:target_idx]


        return tmp_return