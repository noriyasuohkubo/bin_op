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

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")
myLogger = printLog(logger)

class DataSequence2(Sequence):

    def __init__(self, c, startDt, endDt, test_flg, eval_flg, target_spread_list=[], spread_correct=None, target_spread_percent_list=[], sub_force = False, ):
        self.c = c
        self.sub_force = sub_force #答えを算出にsubを使用する場合(学習やeval時にはdiv)

        TARGET_SPREAD_LIST = target_spread_list

        myLogger("TARGET_SPREAD_LIST:",TARGET_SPREAD_LIST)
        print("TARGET_SPREAD_PERCENT_LIST:",target_spread_percent_list)

        SPREAD_CORRECT = c.SPREAD
        if spread_correct != None:
            SPREAD_CORRECT = spread_correct
        print("SPREAD_CORRECT:",SPREAD_CORRECT)

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

        self.train_score_list = [] #test用 予想対象のスコアを保持
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

        self.atr_list = []
        self.atr_dict = {}

        self.hor_list = []
        self.hor_dict = {}

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
            print("tmp_tp", self.tmp_tp, "tmp_sl", self.tmp_sl)
        self.ocoa_skip_cnt = 0

        if self.test_flg:
            self.db_no = c.DB_EVAL_NO

        if self.eval_flg:
            self.db_no = c.DB_EVAL_NO
            self.real_spread_flg = c.REAL_SPREAD_EVAL_FLG

        #すべて同じ足の長さのDBを使うかどうか判定
        for i, db in enumerate(c.INPUT_LEN):
            if i != len(c.INPUT_LEN) -1:
                if c.DB_TERMS[i] != c.DB_TERMS[i + 1]:
                    self.same_db_flg = False

        print("same_db_flg:", self.same_db_flg)

        r = redis.Redis(host=c.DB_HOST, port=6379, db=self.db_no, decode_responses=True)

        if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
            result = r.zrangebyscore(c.FX_TICK_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)
                self.tick_dict[score] = tmps.get("tk")
            del result

        # Fake DB
        db_fake_index = 0
        if c.DB_FAKE_TERM != 0:
            # メモリ空き容量を取得
            print("before fake db ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            for i, db in enumerate(c.DB_FAKE_LIST):
                #1日余分に読み込む
                result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                for j, line in enumerate(result):
                    body = line[0]
                    score = int(line[1])
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
                print("before volume db ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

                for i, db in enumerate(c.DB_VOLUME_LIST):

                    # 1日余分に読み込む
                    result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)

                    #print(db, len(result))
                    for j, line in enumerate(result):
                        body = line[0]
                        score = int(line[1])
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
                score = int(line[1])
                tmps = json.loads(body)

                #特徴力にnanがないかチェック
                ok_flg = True
                for ipt_foot in ipt_list_foot:
                    t_input = tmps.get(ipt_foot)
                    if t_input == None or np.isnan(t_input):
                        ok_flg = False
                        print("FOOT_DB non data", db_tmp, ipt_foot, score)
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
            print("before db extra 1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            db_extra_1_index = 0
            #endscoreより1日余分に読み込む
            result = r.zrangebyscore(c.DB_EXTRA_1, self.start_score - 3600 * 24, self.end_score, withscores=True)
            if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                r.delete(c.DB_EXTRA_1) #メモリ節約のため参照したDBは削除する

            for j, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)
                self.db_extra_1.append(tmps.get("d"))
                self.db_extra_1_score[score] = db_extra_1_index

                db_extra_1_index += 1

            del result

        if len(c.OPTIONS) != 0:
            # メモリ空き容量を取得
            print("before db opt ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            #endscoreより1日余分に読み込む
            result = r.zrangebyscore(c.OPTIONS_DB, self.start_score - 3600 * 24, self.end_score, withscores=True)

            for l, line in enumerate(result):
                body = line[0]
                score = int(line[1])
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

        if c.HOR_DB_CORE != "":
            # 2日余分に読み込む
            result = r.zrangebyscore(c.HOR_DB, self.start_score - 3600 * 24 * 2, self.end_score, withscores=True)
            if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                r.delete(c.HOR_DB)  # メモリ節約のため参照したDBは削除する

            for l, line in enumerate(result):
                body = line[0]
                score = int(line[1])
                tmps = json.loads(body)
                self.hor_dict[score] = tmps.get("data")

            del result


        if c.HIGHLOW_DB_CORE != "":
            # 2日余分に読み込む
            result = r.zrangebyscore(c.HIGHLOW_DB, self.start_score - 3600 * 24 * 2, self.end_score, withscores=True)
            if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                r.delete(c.HIGHLOW_DB)  # メモリ節約のため参照したDBは削除する

            for res in result:
                body = res[0]
                score = int(res[1])
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
            print("highlow dict lungth:", len(self.highlow_dict))
            del result

        if c.OANDA_ORD_DB != "":
            # 1日余分に読み込む
            result = r.zrangebyscore(c.OANDA_ORD_DB, self.start_score - 3600 * 24 * 1, self.end_score, withscores=True)
            if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                r.delete(c.OANDA_ORD_DB)  # メモリ節約のため参照したDBは削除する

            for l, line in enumerate(result):
                body = line[0]
                score = int(line[1])
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
                score = int(line[1])
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
                score = int(line[1])
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
                score = int(line[1])
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
                score = int(line[1])
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
            print(datetime.now(), "before db2 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

            # 長い足のDBから先に全件取得
            db2_index = 0

            for i, db in enumerate(c.DB2_LIST):
                #1日余分に読み込む
                result = r.zrangebyscore(db, self.start_score - 3600 * 24, self.end_score, withscores=True)
                if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
                    r.delete(db)  # メモリ節約のため参照したDBは削除する

                #print(db ,len(result))
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
                                self.db2[ipt].append(t_input)

                    self.db2_score[score] = db2_index
                    self.db2_score_list.append(score)

                    db2_index += 1

                del result

            # メモリ空き容量を取得
            print(datetime.now(), "before db3 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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
                                self.db3[ipt].append(t_input)

                    self.db3_score[score] = db3_index
                    self.db3_score_list.append(score)

                    db3_index += 1

                del result

            # メモリ空き容量を取得
            print(datetime.now(), "before db4 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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
                                self.db4[ipt].append(t_input)

                    self.db4_score[score] = db4_index
                    self.db4_score_list.append(score)

                    db4_index += 1

                del result

            # メモリ空き容量を取得
            print(datetime.now(), "before db5 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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
                                self.db5[ipt].append(t_input)

                    self.db5_score[score] = db5_index
                    self.db5_score_list.append(score)

                    db5_index += 1

                del result

        # メモリ空き容量を取得
        print(datetime.now(), "before db1 ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

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

            for line in result:
                #print("result length:", len(result))
                body = line[0]
                score = float(line[1])
                tmps = json.loads(body)
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

                self.db1_score[score] = db1_index
                self.db1_score_list.append(score)

                spr = 0
                #ハイロー,FXでSpreadデータを使用する場合
                if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
                    spr = tmps.get("s")
                    if spr == None:
                        spr = 0
                    else:
                        spr = float(spr)
                        if spr <1 and 0 < spr:
                            spr = int(spr * 10) #sprがpips形式(0.1など)で入っている場合
                        else:
                            spr = int(spr)
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

                        for tmp_k in self.opt_lists:
                            self.score_dict[score][tmp_k] = prev_dict[tmp_k]

                    if c.FX and c.FX_TICK_DB != "" and self.test_flg and self.eval_flg == False:
                        if get_decimal_sub(score, c.BET_TERM) in self.tick_dict.keys():
                            self.score_dict[score]["tk"] = self.tick_dict[get_decimal_sub(score, c.BET_TERM)]
                        else:
                            self.score_dict[score]["tk"] = self.tick_dict[score]

                    prev_dict["c"] = c_tmp
                    prev_dict["spr"] = spr
                    prev_dict["jpy"] = j_tmp
                    prev_dict["sp"] = sp_tmp

                    for tmp_k in self.opt_lists:
                        tmp_v = tmps.get(tmp_k)
                        tmp_v = float(tmp_v) if tmp_v != None else tmp_v
                        prev_dict[tmp_k] = tmp_v

                db1_index += 1

            del result

            #print(datetime.fromtimestamp(min(self.db1_score)))
            #print(datetime.fromtimestamp(max(self.db1_score)))

            list_idx = list_idx -1

            for i in range(len(score_tmp)):
                list_idx += 1
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
                    self.train_dict_ex[now_score_tmp] = None
                    continue

                try:
                    start_score = score_tmp[i - need_len]
                    end_score = score_tmp[i + c.PRED_TERM + c.END_TERM -1]
                    if end_score != get_decimal_add(start_score, get_decimal_multi((need_len + c.PRED_TERM + c.END_TERM - 1), c.DB1_TERM)):
                        #時刻がつながっていないものは除外 たとえば日付またぎなど
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                except Exception:
                    #start_score end_scoreのデータなしなのでスキップ
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
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                db_volume_index_tmp = -1
                db2_index_tmp = -1
                db3_index_tmp = -1
                db4_index_tmp = -1
                db5_index_tmp = -1
                db6_index_tmp = -1
                db7_index_tmp = -1
                db_extra_1_index_tmp = -1

                db_foot_idxs = {}
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
                            self.train_dict_ex[now_score_tmp] = None
                            continue

                    except Exception:
                        # start_scoreのデータなしなのでスキップ
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
                            self.train_dict_ex[now_score_tmp] = None
                            continue

                    except Exception:
                        # start_scoreのデータなしなのでスキップ
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                ind_foot_list = []
                if len(c.IND_FOOT_COL) != 0 :
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
                        print(e)
                        # データがないのでスキップ
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                hor_val = None
                if c.HOR_DB_CORE != "":
                    try:
                        #直近の足のスコア
                        #tmp_s = int(Decimal(str(now_score_tmp)) - (Decimal(str(now_score_tmp)) % Decimal(c.HOR_TERM)))
                        tmp_s = get_decimal_sub(now_score_tmp, get_decimal_mod(now_score_tmp, c.HOR_TERM))
                        hor_data = self.hor_dict[tmp_s]
                        #if tmp_s == 1679405880:
                        #   print(self.hor_dict[tmp_s])
                        #   print(self.hor_dict[1679405880])
                        tmp_dict = {}
                        for tmp_data in hor_data.split(","):
                            tmp_rate, hit_cnt = tmp_data.split(":")

                            hit_cnt = int(hit_cnt)
                            tmp_dict[tmp_rate] = hit_cnt
                        hor_val = []
                        start_v = -1 * c.HOR_DATA_NUM
                        base = get_decimal_sub(prev_close, Decimal(str(prev_close)) % Decimal(str(c.HOR_WIDTH)))
                        while True:
                            if start_v > c.HOR_DATA_NUM:
                                break
                            target = str(get_decimal_add(base, get_decimal_multi(start_v, c.HOR_WIDTH)))
                            tmp_hit_cnt = tmp_dict.get(target)
                            if tmp_hit_cnt ==  None:
                                tmp_hit_cnt = 0
                            else:
                                tmp_hit_cnt = tmp_hit_cnt - 1  # ヒットした数が2以上のものしかDBにない、ヒットしなかったら0なので合わせるためにヒットする数から1マイナスする

                            hor_val.append(tmp_hit_cnt)
                            start_v += 1

                    except Exception as e:
                        #データがないのでスキップ
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
                        #print(tracebackPrint(e))
                        #データがないのでスキップ
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
                                #print(tmp_rate, prev_close, k)
                                #現在レートが属するレンジが何番目か特定
                                mid_ind = k

                        if mid_ind == None:
                            # 該当レンジがないのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                    except Exception as e:
                        #データがないのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                    except Exception as e:
                        # データがないのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            #start_scoreのデータなしなのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
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
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                        except Exception:
                            # start_scoreのデータなしなのでスキップ
                            self.train_dict_ex[now_score_tmp] = None
                            continue

                    # DB EXTRA 1を使う場合
                    if c.DB_EXTRA_1 != "":
                        db_extra_1_index_tmp = self.db_extra_1_score[now_score_tmp]  # scoreからインデックスを取得

                #取引時間外を対象からはずす
                if self.test_flg == False and len(c.EXCEPT_LIST) != 0:
                    if datetime.fromtimestamp(now_score_tmp).hour in c.EXCEPT_LIST:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                # スプレッドごとの取引時間外を対象からはずす
                if prev_spread in c.EXCEPT_LIST_BY_SPERAD:
                    if datetime.fromtimestamp(now_score_tmp).hour in c.EXCEPT_LIST_BY_SPERAD[prev_spread]:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                # 指定スプレッド以外のトレードは無視する
                if len(TARGET_SPREAD_LIST) != 0:
                    if not (prev_spread in TARGET_SPREAD_LIST):
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                # BITCOIN用 指定スプレッドパーセント以外のトレードは無視する
                if len(target_spread_percent_list) != 0:
                    if not (spread_percent_tmp[i -1] in target_spread_percent_list):
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                if get_decimal_mod(now_score_tmp, c.BET_SHIFT) != 0.0:
                    #指定したシフトでなければ無視
                    self.train_dict_ex[now_score_tmp] = None
                    continue

                # 0秒のデータのみ学習する場合
                if self.test_flg == False and c.ZERO_SEC_FLG and sec != 0:
                    self.train_dict_ex[now_score_tmp] = None
                    continue

                # オプションの値がnanやNoneの場合は無視する
                tmp_opt = None
                if len(c.OPTIONS) != 0:
                    try:
                        tmp_opt = self.option_score[now_score_tmp]
                        if tmp_opt == None:
                            #データが続いていない場合Noneが入っている
                            self.train_dict_ex[now_score_tmp] = None
                            continue
                        else:
                            for op in tmp_opt:
                                if op == None or np.isnan(op):
                                    self.train_dict_ex[now_score_tmp] = None
                                    continue
                    except Exception:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                #jpyの通貨ペアでない場合、日本円レートがDBにない場合はポジション計算できない(get_fx_position_jpy)のでスキップ
                if c.JPY_FLG == False:
                    now_jpy_tmp = jpy_tmp[i - 1]
                    if now_jpy_tmp == None or np.isnan(now_jpy_tmp):
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                if c.ATR_COL != "":
                    now_atr_tmp = atr_tmp[i]
                    if now_atr_tmp == None or np.isnan(now_atr_tmp):
                        #データが続いていない場合Noneが入っている
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
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                if test_flg == False and c.TAG != "":
                    #TAG指定されている場合、タグがついていないデータは対象外とする
                    if tag_tmp[i] == None:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                if c.METHOD == "LSTM2":
                    if predict_tmp[i] == None:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                    """
                    #予想値をdivide / predict_tmp[i]にしている場合、0を除外。0除算エラーとなるため
                    if not(test_flg and eval_flg == False) and predict_tmp[i] == 0:
                        self.train_dict_ex[now_score_tmp] = None
                        continue
                    """
                #直近の変化率
                divide_prev = prev_close / close_tmp[i - 1 - c.DIVIDE_PREV_LENGTH]
                if close_tmp[i - 1 - c.DIVIDE_PREV_LENGTH] == prev_close:
                    divide_prev = 1
                divide_prev = abs(10000 * (divide_prev - 1))

                if c.EXCEPT_DIVIDE_MIN !=0 and c.EXCEPT_DIVIDE_MIN > divide_prev:
                    self.train_dict_ex[now_score_tmp] = None
                    continue
                if c.EXCEPT_DIVIDE_MAX != 0 and c.EXCEPT_DIVIDE_MAX < divide_prev:
                    self.train_dict_ex[now_score_tmp] = None
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
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                if c.LEARNING_TYPE == "CATEGORY_BIN" and (test_flg == False or (test_flg and eval_flg)):
                    #CATEGORY_BINの場合はレート変化ない場合は学習対象外とする
                    if prev_close == pred_close:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                opt_bef = {}
                opt_aft = {}
                for tmp_k in self.opt_lists:
                    if c.OUTPUT_DATA_BEF_C: #変化前の基準をcloseにする場合
                        opt_bef_tmp = prev_close
                    else:
                        opt_bef_tmp = output_tmp_dict[tmp_k][i - 1]
                    opt_aft_tmp = output_tmp_dict[tmp_k][i - 1 + c.PRED_TERM]
                    #答えの基準となる値がNoneならスキップ
                    if opt_bef_tmp == None or opt_aft_tmp == None:
                        self.train_dict_ex[now_score_tmp] = None
                        continue

                    opt_bef[tmp_k] = opt_bef_tmp
                    opt_aft[tmp_k] = opt_aft_tmp

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
                            self.train_dict_ex[now_score_tmp] = None
                            continue
                    else:
                        if abs(test_divide) < c.DIVIDE_MIN :
                            self.train_dict_ex[now_score_tmp] = None
                            continue

                if c.ANSWER_DB != "":
                    #ifd_data = r.zrangebyscore(c.ANSWER_DB, now_score_tmp, now_score_tmp, withscores=True)
                    if self.test_flg and self.eval_flg == False:
                        try:
                            ifd_data = self.answer_dict[now_score_tmp]
                        except Exception:
                            self.train_dict_ex[now_score_tmp] = None
                            continue
                    else:
                        #テスト時ではなくても"OCOPS"の時は学習に使用するので読み込む
                        if "OCOPS:" in c.ANSWER_DB:
                            try:
                                ifd_data = self.answer_dict[now_score_tmp]
                            except Exception:
                                self.train_dict_ex[now_score_tmp] = None
                                continue

                #正解の差を入れていく
                for tmp_k in self.opt_lists:
                    self.output_answer_dict[tmp_k].append(abs(opt_aft[tmp_k] - opt_bef[tmp_k]))

                #spread情報取得
                if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
                    self.spread_cnt += 1
                    spr = prev_spread
                    flg = False
                    for k, v in c.SPREAD_LIST.items():
                        if spr > v[0] and spr <= v[1]:
                            self.spread_cnt_dict[k] = self.spread_cnt_dict.get(k, 0) + 1
                            flg = True
                            break
                    if flg == False:
                        if spr < 0:
                            self.spread_cnt_dict["spread0"] = self.spread_cnt_dict.get("spread0", 0) + 1
                        else:
                            self.spread_cnt_dict["spread16Over"] = self.spread_cnt_dict.get("spread16Over", 0) + 1

                #以下はcontinue処理なし
                self.hor_list.append(hor_val)

                #正解までの変化率
                divide_aft = abs(divide)

                tmp_label = None

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

                if c.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_FOUR", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW"] or \
                        (c.LEARNING_TYPE == "CATEGORY_BIN" and test_flg and eval_flg ==False):
                    if c.OUTPUT_TYPE == "d" and self.sub_force == False:
                        # CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する
                        divide_t = get_decimal_multi(get_divide(opt_bef[c.OUTPUT_DATA],  opt_aft[c.OUTPUT_DATA]), c.OUTPUT_MULTI)
                        if divide_t >= c.BORDER_DIV:
                            # 上がった場合
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([1, 0, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([1, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([0, 1])
                            up = up + 1
                        elif divide_t <= get_decimal_multi(c.BORDER_DIV, -1):
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([0, 0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([1, 0])
                            down = down + 1
                        else:
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([0, 1, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([0, 1])

                    elif c.OUTPUT_TYPE == "sub" or self.sub_force == True:
                        #CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する

                        bef_sub = opt_bef[c.OUTPUT_DATA]
                        aft_sub = opt_aft[c.OUTPUT_DATA]
                        sub_t = get_sub(bef_sub, aft_sub, c.OUTPUT_MULTI)
                        if sub_t >= float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                            # 上がった場合
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([1, 0, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([1, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([0, 1])
                            up = up + 1
                        elif get_decimal_multi(sub_t, -1) >= float(Decimal(str(c.PIPS)) * Decimal(str(spread_correct))):
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([0, 0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([1, 0])
                            down = down + 1
                        else:
                            if c.LEARNING_TYPE == "CATEGORY" or c.LEARNING_TYPE == "CATEGORY_BIN" or c.LEARNING_TYPE == "CATEGORY_BIN_BOTH" or c.LEARNING_TYPE == "CATEGORY_BIN_FOUR":
                                tmp_label = np.array([0, 1, 0])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_UP":
                                tmp_label = np.array([0, 1])
                            elif c.LEARNING_TYPE == "CATEGORY_BIN_DW":
                                tmp_label = np.array([0, 1])
                    else:
                        print("WRONG OUTPUT_TYPE!!!", c.OUTPUT_TYPE)
                        exit(1)

                elif c.LEARNING_TYPE == "CATEGORY_BIN":

                    #CATEGORY系の場合はc.OUTPUT_DATA１種類なのでc.OUTPUT_DATAで値を取得する
                    bef_sub = opt_bef[c.OUTPUT_DATA]
                    aft_sub = opt_aft[c.OUTPUT_DATA]
                    if float(Decimal(str(aft_sub)) - Decimal(str(bef_sub))) >= 0:
                        # 上がった場合
                        tmp_label = np.array([1, 0])
                        up = up + 1
                    else:
                        tmp_label = np.array([0, 1])
                        down = down + 1

                elif c.LEARNING_TYPE in ["CATEGORY_BIN_UP_OCOA","CATEGORY_BIN_DW_OCOA"]:
                    if self.test_flg and self.eval_flg == False:
                        #1件しかないので1件目を取得
                        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_OCOA"]:
                            answer_rate = ifd_data[str(c.TERM) + "-bp"]
                            answer_score = ifd_data["bds"] #決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp )

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                        elif c.LEARNING_TYPE in ["CATEGORY_BIN_DW_OCOA"]:
                            answer_rate = ifd_data[str(c.TERM) + "-sp"]
                            answer_score = ifd_data["sds"]  # 決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp)

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                elif c.LEARNING_TYPE in ["CATEGORY_BIN_UP_OCO","CATEGORY_BIN_DW_OCO"]:
                    if self.test_flg and self.eval_flg == False:
                        #1件しかないので1件目を取得
                        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_OCO"]:
                            answer_rate = ifd_data[str(c.TERM) + "-bp"]
                            answer_score = ifd_data["bds"] #決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp )

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                        elif c.LEARNING_TYPE in ["CATEGORY_BIN_DW_OCO"]:
                            answer_rate = ifd_data[str(c.TERM) + "-sp"]
                            answer_score = ifd_data["sds"]  # 決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp)

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                elif c.LEARNING_TYPE in ["CATEGORY_BIN_UP_TP","CATEGORY_BIN_DW_TP"]:
                    if self.test_flg and self.eval_flg == False:
                        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_TP"]:
                            answer_rate = ifd_data[str(c.TERM) + "-bp"]
                            answer_score = ifd_data["bds"] #決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp )

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                        elif c.LEARNING_TYPE in ["CATEGORY_BIN_DW_TP"]:
                            answer_rate = ifd_data[str(c.TERM) + "-sp"]
                            answer_score = ifd_data["sds"]  # 決済時のスコア

                            if answer_rate == None or answer_score == None:
                                print("answer_rate or answer_score is null!!  score:", now_score_tmp)

                            if answer_rate > 0:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                elif c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFD", "CATEGORY_BIN_DW_IFD","CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO"]:
                    if self.test_flg and self.eval_flg == False:
                        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFD","CATEGORY_BIN_UP_IFO"]:
                            btpf = ifd_data["btpf"]
                            if btpf == 1:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                            #指値が成立してなければDBに値がないので利益は0とする
                            answer_rate = ifd_data[str(c.TERM) + "-bp"] if ifd_data[str(c.TERM) + "-bp"] != None else 0
                        elif c.LEARNING_TYPE in ["CATEGORY_BIN_DW_IFD", "CATEGORY_BIN_DW_IFO"]:
                            stpf = ifd_data["stpf"]
                            if stpf == 1:
                                tmp_label = np.array([1, 0])
                                take_profit_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                            #指値が成立してなければDBに値がないので利益は0とする
                            answer_rate = ifd_data[str(c.TERM) + "-sp"] if ifd_data[str(c.TERM) + "-sp"] != None else 0

                elif c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFDSF","CATEGORY_BIN_DW_IFDSF"]:
                    if self.test_flg and self.eval_flg == False:
                        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFDSF"]:
                            bpf = ifd_data["bpf"]
                            bslf = ifd_data["bslf"]
                            if bpf == 1 and bslf == 1:
                                #ポジションを持てて、stoplossになるものを当てる
                                tmp_label = np.array([1, 0])
                                stop_loss_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                            #指値が成立してなければDBに値がないので利益は0とする
                            answer_rate = ifd_data[str(c.TERM) + "-bp"] if ifd_data[str(c.TERM) + "-bp"] != None else 0
                            answer_score = ifd_data["bds"] #決済時のスコア

                        elif c.LEARNING_TYPE in ["CATEGORY_BIN_DW_IFDSF",]:
                            spf = ifd_data["spf"]
                            sslf = ifd_data["sslf"]
                            if spf == 1 and sslf == 1:
                                #ポジションを持てて、stoplossになるものを当てる
                                tmp_label = np.array([1, 0])
                                stop_loss_cnt += 1
                            else:
                                tmp_label = np.array([0, 1])

                            #指値が成立してなければDBに値がないので利益は0とする
                            answer_rate = ifd_data[str(c.TERM) + "-sp"] if ifd_data[str(c.TERM) + "-sp"] != None else 0
                            answer_score = ifd_data["sds"]# 決済時のスコア

                elif c.LEARNING_TYPE in ["REGRESSION_UP","REGRESSION_DW"]:
                    if self.test_flg and self.eval_flg == False:
                        if c.LEARNING_TYPE in ["REGRESSION_UP"]:
                            if "IFOAA" in c.ANSWER_DB:
                                if ifd_data["bpf"] != 1:
                                    # 指値が成立してなければDBに値がないので利益は0とする
                                    answer_rate = 0
                                    # 指値が成立してなければ決済時のスコアはDBに値がないので予測時間とする
                                    answer_score = now_score_tmp + c.TERM
                                else:
                                    answer_rate = ifd_data[str(c.TERM) + "-bp"]
                                    answer_score = ifd_data["bds"]
                            else:
                                answer_rate = ifd_data[str(c.TERM) + "-bp"]
                                answer_score = ifd_data["bds"] #決済時のスコア

                                if answer_rate == None or answer_score == None:
                                    print("answer_rate or answer_score is null!!  score:", now_score_tmp )

                            tmp_label = answer_rate

                            if answer_rate > 0:
                                take_profit_cnt += 1

                        elif c.LEARNING_TYPE in ["REGRESSION_DW"]:
                            if "IFOAA" in c.ANSWER_DB:
                                if ifd_data["spf"] != 1:
                                    answer_rate = 0
                                    answer_score = now_score_tmp + c.TERM
                                else:
                                    answer_rate = ifd_data[str(c.TERM) + "-sp"]
                                    answer_score = ifd_data["sds"]  # 決済時のスコア
                            else:
                                answer_rate = ifd_data[str(c.TERM) + "-sp"]
                                answer_score = ifd_data["sds"]  # 決済時のスコア

                                if answer_rate == None or answer_score == None:
                                    print("answer_rate or answer_score is null!!  score:", now_score_tmp)

                            tmp_label = answer_rate

                            if answer_rate > 0:
                                take_profit_cnt += 1

                elif c.LEARNING_TYPE in ["CATEGORY_OCOPS"]:
                    btpf = ifd_data["btpf"]
                    b_answer_rate = ifd_data[str(c.TERM) + "-bp"]
                    b_answer_score = ifd_data["bds"]  # 決済時のスコア

                    if b_answer_rate == None or b_answer_score == None:
                        print("b_answer_rate or b_answer_score is null!!  score:", now_score_tmp)

                    stpf = ifd_data["stpf"]
                    s_answer_rate = ifd_data[str(c.TERM) + "-sp"]
                    s_answer_score = ifd_data["sds"]  # 決済時のスコア

                    if s_answer_rate == None or s_answer_score == None:
                        print("s_answer_rate or s_answer_score is null!!  score:", now_score_tmp)

                    self.category_ocops_cnt += 1
                    tmp_label = np.array([0, 1, 0]) #どちらもtkしていない場合

                    if btpf == 1:
                        if stpf == 0 or (stpf == 1 and s_answer_score > b_answer_score):
                            #buyでtakeprofitした場合、sellでtkしていないか、していてもbuyの方が早く決済していたら
                            tmp_label = np.array([1,0,0])
                            self.category_ocops_up_cnt += 1
                    if stpf == 1:
                        if btpf == 0 or (btpf == 1 and b_answer_score > s_answer_score):
                            tmp_label = np.array([0,0,1])
                            self.category_ocops_dw_cnt += 1

                elif c.LEARNING_TYPE in ["REGRESSION_OCOPS"]:
                    b_answer_rate = ifd_data[str(c.TERM) + "-bp"]
                    b_answer_score = ifd_data["bds"]  # 決済時のスコア

                    if b_answer_rate == None or b_answer_score == None:
                        print("b_answer_rate or b_answer_score is null!!  score:", now_score_tmp)

                    s_answer_rate = ifd_data[str(c.TERM) + "-sp"]
                    s_answer_score = ifd_data["sds"]  # 決済時のスコア

                    if s_answer_rate == None or s_answer_score == None:
                        print("s_answer_rate or s_answer_score is null!!  score:", now_score_tmp)

                    tmp_label = np.array([b_answer_rate, s_answer_rate])

                elif c.LEARNING_TYPE in ["REGRESSION", "REGRESSION_SIGMA"]:
                    if c.METHOD == "LSTM2":
                        if predict_tmp[i] != None:
                            tmp_label = np.array(divide - predict_tmp[i])
                        #else:
                        #    #予想はかならずあるはずなので、なければエラー
                        #    print("Error!!! predict is None! score:", now_score_tmp)
                        #    sys.exit(1)
                    else:
                        if c.DIRECT_FLG:
                            tmp_label = aft
                        else:

                            if c.OUTPUT_TYPE == "d":
                                l_list = []
                                for tmp_k in self.opt_lists:
                                    l_list.append(get_decimal_multi(get_divide(opt_bef[tmp_k], opt_aft[tmp_k]), c.OUTPUT_MULTI ))
                                tmp_label = np.array(l_list)

                            elif c.OUTPUT_TYPE == "sub":
                                l_list = []
                                for tmp_k in self.opt_lists:
                                    l_list.append(get_sub(opt_bef[tmp_k], opt_aft[tmp_k], c.OUTPUT_MULTI))
                                tmp_label = np.array(l_list)

                            else:
                                print("WRONG OUTPUT_TYPE!!!", c.OUTPUT_TYPE)
                                exit(1)

                if c.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN", "REGRESSION","REGRESSION_OCOPS", "CATEGORY_OCOPS"]:
                    if c.ANSWER_DB != "" and self.test_flg and self.eval_flg == False:
                        answer_rate_up = ifd_data[str(c.TERM) + "-bp"]
                        answer_score_up = ifd_data["bds"]  # 決済時のスコア
                        answer_rate_dw = ifd_data[str(c.TERM) + "-sp"]
                        answer_score_dw = ifd_data["sds"]  # 決済時のスコア

                        if answer_rate_up == None or answer_score_up == None or answer_rate_dw == None or answer_score_dw == None:
                            print("answer_rate or answer_score is null!!  score:", now_score_tmp)
                            exit(1)

                        answer_rate = [answer_rate_up, answer_rate_dw]
                        answer_score = [answer_score_up, answer_score_dw]

                if c.SEC_OH_LEN_FIX_FLG:
                    sec_oh = sec
                else:
                    sec_oh = int(Decimal(str(sec)) / Decimal(str(c.BET_SHIFT)) )  # 2秒間隔データなら０から29に変換しなければならないのでbet_termで割る

                if test_flg and eval_flg == False:
                    # 一旦scoreをキーに辞書に登録 後でスコア順にならべてtrain_listにいれる
                    # 複数のDBを使用した場合に結果を時系列順にならべて確認するため
                    if len(self.train_dict) == 0:
                        print("first score", now_score_tmp)
                    self.train_dict[now_score_tmp] = [tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp,
                                                     db5_index_tmp, db6_index_tmp, db7_index_tmp, predict_tmp[i], bef, sec_oh, minute, hour, week, db_volume_index_tmp, db_extra_1_index_tmp, tmp_opt, ind_foot_list, oanda_ord_list, oanda_pos_list, hor_val,db_foot_idxs, highlow_val, aft, spread_t, divide_prev, divide_aft, spread_t_end,
                                                     answer_rate, answer_score]
                else:
                    self.train_list.append([tmp_label, list_idx, db2_index_tmp, db3_index_tmp, db4_index_tmp, db5_index_tmp, db6_index_tmp, db7_index_tmp, predict_tmp[i], bef, sec_oh, minute, hour, week, db_volume_index_tmp, db_extra_1_index_tmp, tmp_opt, ind_foot_list, oanda_ord_list, oanda_pos_list, hor_val, db_foot_idxs, highlow_val ])

        if (self.test_flg and self.eval_flg == False and c.DELETE_TEST_FLG) or (self.test_flg == False and c.DELETE_LEARN_FLG):
            # メモリ節約のためDBは削除する
            r.flushdb()

        # メモリ空き容量を取得
        #print("before db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        # メモリ節約のためredis停止
        """
        if test_flg == False:
            #r.shutdown() #パスワード入力を求められる(権限がない)のでshutdownできない
            sudo_password = 'Reikou0129'
            command = 'systemctl stop redis'.split()

            p = Popen(['sudo', '-S'] + command, stdin=PIPE, stderr=PIPE,
                      universal_newlines=True)
            sudo_prompt = p.communicate(sudo_password + '\n')[1]

            # メモリ空き容量を取得
            print("after db shutdown ", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        """

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

        # メモリ解放
        del close_tmp
        del score_tmp, predict_tmp,
        del self.hor_dict, self.oanda_ord_dict, self.oanda_pos_dict , self.ind_foot_dict
        del self.ind_score_dict,self.atr_dict, self.db_foots_score


        #numpy化してsliceを早くする
        self.db1 = self.make_input_data_list_np(self.db1, c.INPUT_DATAS[0])
        self.db2 = self.make_input_data_list_np(self.db2, c.INPUT_DATAS[1]) if len(self.c.INPUT_LEN) >=2 else None
        self.db3 = self.make_input_data_list_np(self.db3, c.INPUT_DATAS[2]) if len(self.c.INPUT_LEN) >=3 else None
        self.db4 = self.make_input_data_list_np(self.db4, c.INPUT_DATAS[3]) if len(self.c.INPUT_LEN) >=4 else None
        self.db5 = self.make_input_data_list_np(self.db5, c.INPUT_DATAS[4]) if len(self.c.INPUT_LEN) >=5 else None

        self.db_volume_np = np.array(self.db_volume)
        del self.db_volume

        if c.METHOD == "LSTM9":
            self.db_pred_np = np.array(self.db_preds[self.db9_name])

        if c.DB_EXTRA_1 != "":
            self.db_extra_1_np = np.array(self.db_extra_1)
        del self.db_extra_1

        self.data_length = len(self.train_list)
        self.cut_num = (self.data_length) % c.BATCH_SIZE

        print(datetime.now(), "data length:" , self.data_length)
        print(datetime.now(), "ocoa_skip_cnt:", self.ocoa_skip_cnt)

        if self.cut_num !=0:
            # エポック毎の学習ステップ数
            # バッチサイズが1ステップで学習するデータ数
            # self.cut_num(総学習データ÷バッチサイズ)で余りがあるなら余りがある分ステップ数が1多い
            self.steps_per_epoch = int(self.data_length / c.BATCH_SIZE) +1
        else:
            self.steps_per_epoch = int(self.data_length / c.BATCH_SIZE)

        print("steps_per_epoch: ", self.steps_per_epoch)

        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFD", "CATEGORY_BIN_DW_IFD",
                               "CATEGORY_BIN_UP_IFO", "CATEGORY_BIN_DW_IFO","CATEGORY_BIN_UP_TP", "CATEGORY_BIN_DW_TP",
                               "CATEGORY_BIN_UP_OCO","CATEGORY_BIN_DW_OCO","CATEGORY_BIN_UP_OCOA","CATEGORY_BIN_DW_OCOA",
                               "REGRESSION_UP", "REGRESSION_DW",
                               ] and take_profit_cnt != 0:
            print("take profit rate:", take_profit_cnt/self.data_length)

        if c.LEARNING_TYPE in ["CATEGORY_BIN_UP_IFDSF","CATEGORY_BIN_DW_IFDSF"] and stop_loss_cnt != 0:
            print("stop loss rate:", stop_loss_cnt/self.data_length)

        if c.LEARNING_TYPE == "CATEGORY_OCOPS":
            print("category_ocops_up_cnt",self.category_ocops_up_cnt)
            print("category_ocops_dw_cnt",self.category_ocops_dw_cnt)
            print("category_ocops_same_cnt",self.category_ocops_cnt - (self.category_ocops_up_cnt + self.category_ocops_dw_cnt))

        if up != 0 and down != 0:
            print("UP: ",up/self.data_length)
            if self.data_length - up - down != 0:
                print("SAME: ", (self.data_length - up - down) / (self.data_length))
            print("DOWN: ",down/self.data_length)

        #Spreadの内訳を表示
        if (c.FX == False and self.real_spread_flg) or (c.FX and c.FX_REAL_SPREAD_FLG):
            print("spread total: ", self.spread_cnt)
            if self.spread_cnt != 0:
                for k, v in sorted(self.spread_cnt_dict.items()):
                    print(k, v, v / self.spread_cnt)

        if tmp_k in self.opt_lists:
            ans_d_list = np.array(self.output_answer_dict[tmp_k])
            print("結果までの" + tmp_k + "の絶対差")
            print("avg:", np.average(ans_d_list))
            print("std:", np.std(ans_d_list))
            print("max:", np.max(ans_d_list))

        if self.test_flg == False and c.DATA_SHUFFLE == "SHUFFLE":
                #シャッフルする
                random.seed(c.SEED)
                random.shuffle(self.train_list)


        def tmp_create_db(x, start_idx, db_no, ipt):
            target_idx = self.train_list[start_idx + x][db_no]
            tmp_return = None

            if self.same_db_flg == False:
                if db_no == 1:
                    tmp_return = self.db1[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]
                elif db_no == 2:
                    tmp_return = self.db2[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]
                elif db_no == 3:
                    tmp_return = self.db3[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]
                elif db_no == 4:
                    tmp_return = self.db4[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]
                elif db_no == 5:
                    tmp_return = self.db5[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]
            else:
                tmp_return = self.db1[ipt][target_idx - c.INPUT_LEN[db_no - 1]:target_idx]

            """
            if (ipt != "d-1" and ipt != "d1" and ipt != "md1-1" and ipt != "ehd1-1" and ipt != "eld1-1" and ipt != "wmd1-1") and self.epoch_cnt  == 0 and self.data_checked == False:
                #Noneやnanがはいっていたらエラー
                if np.any(tmp_return == None) or np.any(np.isnan(tmp_return)):
                    print("return is None or nan!!!")
                    print("target_idx:", target_idx)
                    exit()
            
            else:
                self.data_checked == True
            """
            return tmp_return

        self.create_db = np.vectorize(tmp_create_db, otypes=[np.ndarray])

        def tmp_create_db_all(x, start_idx, db_no):
            target_idx = self.train_list[start_idx + x][db_no]

            if db_no ==1 :
                base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db1_np[target_idx -1])
                tmp_arr = self.db1_np[target_idx - c.INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000

            elif db_no ==2 :
                base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db2_np[target_idx -1])
                tmp_arr = self.db2_np[target_idx - c.INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==3 :
                base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db3_np[target_idx -1])
                tmp_arr = self.db3_np[target_idx - c.INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==4 :
                base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db4_np[target_idx -1])
                tmp_arr = self.db4_np[target_idx - c.INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000
            elif db_no ==5 :
                base_arr = np.full(c.INPUT_LEN[db_no - 1], self.db5_np[target_idx -1])
                tmp_arr = self.db5_np[target_idx - c.INPUT_LEN[db_no - 1] -1:target_idx -1]
                return (tmp_arr/base_arr -1) * 10000

        self.create_db_all = np.vectorize(tmp_create_db_all, otypes=[np.ndarray])

        def tmp_create_label(x, start_idx):
            return self.train_list[start_idx + x][0]

        self.create_label = np.vectorize(tmp_create_label, otypes=[np.ndarray])

        def tmp_create_volume(x, start_idx):
            target_idx = self.train_list[start_idx + x][14]
            vol_arr = self.db_volume_np[target_idx - c.DB_VOLUME_INPUT_LEN:target_idx]
            ret = 0
            for vol in vol_arr:
                ret += vol

            return ret

        self.create_volume = np.vectorize(tmp_create_volume, otypes=[np.ndarray])

        def tmp_create_predict(x, start_idx):
            return self.train_list[start_idx + x][8]

        self.create_predict = np.vectorize(tmp_create_predict, otypes=[np.ndarray])

        def tmp_create_now_rate(x, start_idx):
            return self.train_list[start_idx + x][9]

        self.create_now_rate = np.vectorize(tmp_create_now_rate, otypes=[np.ndarray])

        def tmp_create_sec(x, start_idx):
            # secをOne-Hotベクトルに変換
            #return np.identity(SEC_OH_LEN)[self.train_list[start_idx + x][10]]

            return self.train_list[start_idx + x][10]

        self.create_sec = np.vectorize(tmp_create_sec, otypes=[np.ndarray])

        def tmp_create_min(x, start_idx):
            return self.train_list[start_idx + x][11]

        self.create_min = np.vectorize(tmp_create_min, otypes=[np.ndarray])

        def tmp_create_hour(x, start_idx):
            return self.train_list[start_idx + x][12]

        self.create_hour = np.vectorize(tmp_create_hour, otypes=[np.ndarray])

        def tmp_create_week(x, start_idx):
            return self.train_list[start_idx + x][13]

        self.create_week = np.vectorize(tmp_create_week, otypes=[np.ndarray])

        def tmp_create_pred(x, start_idx, len):
            target_idx = self.train_list[start_idx + x][1]
            return self.db_pred_np[target_idx - len]

        self.create_pred = np.vectorize(tmp_create_pred, otypes=[np.ndarray])

        def tmp_create_pred_close(x, start_idx, len):
            target_idx = self.train_list[start_idx + x][1]

            bef = self.db_close[target_idx - 1 - len]
            aft = self.db_close[target_idx - 1]

            divide_org = aft / bef
            if aft == bef:
                divide_org = 1

            divide = 10000 * (divide_org - 1)
            return divide

        self.create_pred_close = np.vectorize(tmp_create_pred_close, otypes=[np.ndarray])

        def tmp_create_db_extra(x, start_idx):
            target_idx = self.train_list[start_idx + x][15]
            return self.db_extra_1_np[target_idx - c.DB_EXTRA_1_LEN:target_idx]

        self.create_db_extra = np.vectorize(tmp_create_db_extra, otypes=[np.ndarray])

        def tmp_create_option(x, start_idx, num):
            return self.train_list[start_idx + x][16][num]

        self.create_option = np.vectorize(tmp_create_option, otypes=[np.ndarray])

        def tmp_create_ind_foot(x, start_idx, num):
            try:
                tmp_val = self.train_list[start_idx + x][17][num]
            except Exception as e:
                print(self.train_list[start_idx + x])
                exit(1)

            return tmp_val

        self.create_ind_foot = np.vectorize(tmp_create_ind_foot, otypes=[np.ndarray])

        def tmp_create_oanda_ord(x, start_idx, num):
            try:
                return self.train_list[start_idx + x][18][num]
            except Exception as e:
                print(self.train_list[start_idx + x][18])
                exit(1)

        self.create_oanda_ord = np.vectorize(tmp_create_oanda_ord, otypes=[np.ndarray])

        def tmp_create_oanda_pos(x, start_idx, num):

            return self.train_list[start_idx + x][19][num]

        self.create_oanda_pos = np.vectorize(tmp_create_oanda_pos, otypes=[np.ndarray])

        def tmp_create_hor(x, start_idx, i):
            tmp_val = self.train_list[start_idx + x][20][i]
            return tmp_val

        self.create_hor = np.vectorize(tmp_create_hor, otypes=[np.ndarray])

        def tmp_create_foot_db(x, start_idx, db_term, db_len, ipt):
            try:
                target_idx = self.train_list[start_idx + x][21][db_term]
            except Exception as e:
                print(self.train_list[start_idx + x][21])
                exit(1)

            target_idx_end  = target_idx+1
            tmp_return = self.db_foots[db_term][ipt][target_idx_end - db_len: target_idx_end]

            return tmp_return

        self.create_foot_db = np.vectorize(tmp_create_foot_db, otypes=[np.ndarray])

        def tmp_create_highlow(x, start_idx, i):
            tmp_val = self.train_list[start_idx + x][22][i]
            return tmp_val

        self.create_highlow = np.vectorize(tmp_create_highlow, otypes=[np.ndarray])

        def tmp_create_non_lstm(x, start_idx, db_no, ipt, i):
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

        self.create_non_lstm = np.vectorize(tmp_create_non_lstm, otypes=[np.ndarray])

    # 学習データを返すメソッド
    # idxは要求されたデータが何番目かを示すインデックス値
    # (訓練データ, 教師データ)のタプルを返す
    def __getitem__(self, idx):
        #start = time.time()
        # データの取得実装
        #print("idx:", idx)
        if idx == 0:
            if self.test_flg == False:
                tmp_l = []
                for j in range(20):
                    tmp_l.append(self.train_list[j][1])
                print(tmp_l)
        tmp_np = np.arange(self.c.BATCH_SIZE)
        # self.cut_num(総学習データ÷バッチサイズ)で余りがあり、且つ最後のステップの場合、
        # リターンする配列のサイズはバッチサイズでなく、余りの数となる
        if idx == (self.steps_per_epoch -1) and self.cut_num != 0:
            tmp_np = np.arange(self.cut_num)
        #print("tmp_np:", tmp_np)
        #tmp_np = np.arange(3)
        #tmp = self.g(tmp_np)
        #new_tmp = tmp.tolist()
        #print(np.array(new_tmp).shape)

        # start_idxからバッチサイズ分取得開始インデックスをずらす
        start_idx = idx * self.c.BATCH_SIZE

        label_data_tmp= self.create_label(tmp_np, start_idx)
        label_data_tmp = label_data_tmp.tolist()
        retY = np.array(label_data_tmp)

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
                    tmpX = np.zeros((len(tmp_np), self.c.INPUT_LEN[i], len(ipt_list)))
                    for idx, ipt in enumerate(ipt_list):
                        tmp_arr = self.create_db(tmp_np, start_idx, i + 1, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr)
                        tmpX[:, :, idx] = tmp_arr[:]

                    retX.append(tmpX)
                else:
                    for ipt in ipt_list:
                        tmpX = np.zeros((len(tmp_np), self.c.INPUT_LEN[i], 1))

                        tmp_arr = self.create_db(tmp_np, start_idx, i + 1, ipt)
                        tmp_arr = tmp_arr.tolist()
                        tmp_arr = np.array(tmp_arr)
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
                retX.append(np.identity(self.c.SEC_OH_LEN)[sec_data_tmp])

            if self.c.METHOD == "LSTM4" or self.c.METHOD == "LSTM5" or self.c.METHOD == "LSTM6" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or \
                    self.c.METHOD == "LSTM9" or self.c.METHOD == "LSTM10" or self.c.METHOD == "TCN7":
                min_data_tmp = self.create_min(tmp_np, start_idx)
                min_data_tmp = min_data_tmp.tolist()
                retX.append(np.identity(self.c.MIN_OH_LEN)[min_data_tmp])

            if self.c.METHOD == "LSTM5" or self.c.METHOD == "LSTM7" or self.c.METHOD == "LSTM8" or self.c.METHOD == "LSTM9" or \
                    self.c.METHOD == "TCN7" or self.c.METHOD == "LSTM10":
                hour_data_tmp = self.create_hour(tmp_np, start_idx)
                hour_data_tmp = hour_data_tmp.tolist()
                retX.append(np.identity(self.c.HOUR_OH_LEN)[hour_data_tmp])

            if self.c.METHOD == "LSTM10":
                week_data_tmp = self.create_week(tmp_np, start_idx)
                week_data_tmp = week_data_tmp.tolist()
                retX.append(np.identity(self.c.WEEK_OH_LEN)[week_data_tmp])

            if self.c.HOR_LEARN_ON:
                for i in range(self.c.HOR_DATA_NUM * 2 + 1):
                    data_tmp = self.create_hor(tmp_np, start_idx, i)
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
                volume_data_tmp = self.create_volume(tmp_np, start_idx)
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
                extra_data_tmp = self.create_db_extra(tmp_np, start_idx)
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
                    tmp_arr = self.create_db(tmp_np, start_idx, i + 1, ipt)
                    tmp_arr = tmp_arr.tolist()
                    tmp_arr = np.array(tmp_arr)
                    tmpX[:, :, idx] = tmp_arr[:]

                retX.append(tmpX)

        """
        #テストデータを一件表示する
        if self.test_flg and self.eval_flg == False:
            if idx == 0:
                print("test data number1 :", retX[0])
        """

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

    def on_epoch_end(self):
        self.epoch_cnt += 1

        if self.test_flg == False:
            if self.c.DATA_SHUFFLE == "ROTATE":
                rotate_num = int(len(self.train_list) / self.c.EPOCH)
                self.train_list = rotate(self.train_list, rotate_num)

            elif self.c.DATA_SHUFFLE == "SHUFFLE":
                random.seed(self.c.SEED)
                random.shuffle(self.train_list)

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
            return_dict[ipt] = np.array(org_dict[ipt])

        return return_dict

    def change_eval_flg(self, eval_flg):
        self.eval_flg = eval_flg