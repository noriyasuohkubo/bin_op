import copy
import json
import numpy as np
import os

import psutil
import redis
import datetime
import time
import gc
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from util import *
import pandas as pd
import socket
from dateutil.relativedelta import relativedelta

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + ".txt"
output = output_log(output_log_name)

"""
tsfreshライブラリを使用して特徴量を作成する
"""
def make_tsfresh_test(start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec,
                 data_len, symbol, org_db, new_db, targets, divs, subs, data_chain, test_term):
    total_start_time = time.perf_counter()


    #学習対象外の時間
    except_hour_list = [20,21,22]

    #もとの特徴量 cはopenレートを得るため必須
    inputs = copy.deepcopy(targets)
    if ('c' in inputs) == False:
        inputs.append('c')

    for d in divs:
        targets.append(d + "_d")

    for sub in subs:
        targets.append(sub + "_s")

    if len(db_list) == 0:
        db_list = []
        if term >= bet_term:
            for i in range(int(Decimal(str(term)) / Decimal(str(bet_term)))):
                db_list.append(symbol + "_" + str(term) + "_" + str(term - ((i + 1) * bet_term)))
        else:
            db_list.append(symbol + "_" + str(term) + "_0")

    #start = datetime(2022, 11, 1)
    start_stp = int(time.mktime(start.timetuple()))

    #end = datetime(2023, 9, 1)
    end_stp = int(time.mktime(end.timetuple())) -1

    output("start-end:",start, end)
    output("org_db_no:",org_db_no)
    output("new_db_no:",new_db_no)
    output("db_list:",db_list)
    output("bet_term:",bet_term)
    output("term:",term)
    output("pred_sec:",pred_sec)
    output("data_len:",data_len)
    output("symbol:",symbol)
    output("org_db:",org_db)
    output("new_db:",new_db)
    output("inputs:",inputs)
    output("targets:",targets)
    output("divs:",divs)
    output("subs:",subs)

    output("test_term:",test_term)

    output("y:","変化率(div)")
    output("")

    redis_db_org = redis.Redis(host=org_db, port=6379, db=org_db_no, decode_responses=True)


    if pred_sec < term:
        #answer用のレートを保持する
        answer_db_name = symbol + "_" + str(bet_term) + "_0"
        result_data_answer = redis_db_org.zrangebyscore(answer_db_name, start_stp, end_stp, withscores=True)
        answer_dict = {} #key:score, val:close

        for i, line in enumerate(result_data_answer):
            body = line[0]
            score = int(line[1])
            tmps = json.loads(body)
            answer_dict[score] = tmps['c']

        del result_data_answer

    id = 0
    y_list = []  # 正解は変化率とする

    # 以下は特徴量算出のためのdataframeをつくるためのデータ
    df_dict = {
        'id': [],
        'time': [],
    }
    for k in targets:
        df_dict[k] = []

    for db in db_list:
        print("db:", db)

        result_data = redis_db_org.zrangebyscore(db, start_stp, end_stp, withscores=True)
        output("result_data length:" + str(len(result_data)))

        score_index = {}  # scoreとlistsでのindexのひもづけ

        lists_dict = {}
        for inp in inputs:
            lists_dict[inp] =[]

        for i, line in enumerate(result_data):
            body = line[0]
            score = int(line[1])
            tmps = json.loads(body)

            for inp in inputs:
                lists_dict[inp].append(float(tmps.get(inp)))

            score_index[score] = i

        del result_data

        for d in divs:
            orglist = lists_dict[d]
            d_bef = np.roll(orglist, 1)
            target = get_divide_arr(d_bef, orglist, math_log=False)
            target[0] = None
            lists_dict[d + "_d"] = target

        for sub in subs:
            orglist = lists_dict[sub]
            sub_bef = np.roll(orglist, 1)
            target = get_sub_arr(sub_bef, orglist,)
            target[0] = None
            lists_dict[sub + "_s"] = target

        for score, idx in score_index.items():
            if data_chain:
                try:
                    start_score = score - (term * data_len)
                    start_idx = score_index[start_score]
                    if start_idx == 0 and (len(divs) != 0 or len(subs) != 0):
                        # div作成時に最初のデータにNoneが入るためスキップ
                        continue
                    end_score = start_score + (term * (data_len -1))
                    end_idx = score_index[end_score]

                    if (end_idx - start_idx) != (data_len -1):
                        #データが続いていなければスキップ
                        continue

                except Exception:
                    #データが存在しなければスキップ
                    continue
            else:
                if idx < data_len:
                    #データが足りないのでスキップ
                    continue

                end_score = score - term
                try:
                    end_idx = score_index[end_score]
                    if end_idx != (idx - 1):
                        continue
                except Exception:
                    #データが存在しなければスキップ
                    continue

                start_idx = end_idx - (data_len -1)
                if start_idx == 0 and (len(divs) != 0 or len(subs) != 0):
                    # div作成時に最初のデータにNoneが入るためスキップ
                    continue

            # 取引時間外を対象からはずす
            if datetime.fromtimestamp(score).hour in except_hour_list:
                continue
            # test_term秒に絞る
            if datetime.fromtimestamp(score).second % test_term != 0:
                continue

            #answerを求める
            if pred_sec < term:
                answer_score = score + pred_sec - bet_term
                try:
                    answer = answer_dict[answer_score]
                except Exception as e:
                    continue
            else:
                pred_term = int(Decimal(str(pred_sec)) / Decimal(str(term)))
                answer_score = end_score + (term * pred_term)
                try:
                    answer_idx = score_index[answer_score]
                    answer = lists_dict['c'][answer_idx]
                except Exception as e:
                    continue
            id += 1
            open = lists_dict['c'][end_idx]

            y_list.append(get_divide(open, answer))

            for j, k in enumerate(targets):

                tmp_list = lists_dict[k]
                x_list = tmp_list[start_idx: (end_idx + 1)]
                for l, x in enumerate(x_list):
                    df_dict[k].append(x)
                    if j == 0:
                        df_dict['id'].append(id)
                        df_dict['time'].append(l)

        del lists_dict

    x_df = pd.DataFrame(df_dict)

    del df_dict
    gc.collect()
    #print(y_series)
    #print(x_df)

    #特徴量作成
    start_time = time.perf_counter()

    y_index = [i + 1 for i in range(len(y_list))]
    y_series = pd.Series(data=y_list, index=y_index)

    #すべての特徴量を作成する場合
    extracted_features = extract_features(x_df,
                                          column_id="id",
                                          column_sort="time",
                                          )
    #tsfreshが使用できる特徴量一覧
    #https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    output("extracted_features.info")
    output(extracted_features.info)
    output("extracted_features takes:", time.perf_counter() - start_time)

    #欠損値、異常値（無限大）を処理する関数 select_features時にnanがあるとエラーになる
    start_time = time.perf_counter()
    impute(extracted_features)
    output("impute takes:", time.perf_counter() - start_time)

    #不要な特徴量を削除
    start_time = time.perf_counter()
    features_filtered = select_features(extracted_features, y_series)
    output("features_filtered.info")
    output(features_filtered.info)
    output("select_features takes:", time.perf_counter() - start_time)

    features_filtered_col_list = features_filtered.columns.tolist()
    output(list_to_str(features_filtered_col_list, spl="@"))

    #作成された特徴量を出力
    output("")
    for col in features_filtered_col_list:
        output(col)
    output("")

    output("total takes:", time.perf_counter() - total_start_time)



if __name__ == '__main__':

    # start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, targets, divs, subs, mode, data_chain をリストで保持させる
    # db_list:抽出元のDB名のリスト。空のリストの場合はbet_termとtermから自動で抽出元が決定される。空でない場合はbet_termは不要なのでNoneにする
    # data_len:特徴量算出のもとになる時系列データの数
    # targets:特徴量計算元の特徴量
    # div:divをとる特徴量
    # subs:subをとる特徴量
    # data_chain:データが続いていなければ対象としない。分足などは続いていなくとも(休日を挟んでいても)OKなのでFalseにする
    # test_term: test時に,テスト対象とするスコアをこの値で割った余りが0のものだけとする
    dates = [
        [datetime(2022, 11, 1), datetime(2023, 11, 4), 2, 2, ['USDJPY_10_0',], 2, 10, 30, 3, 'USDJPY', 'localhost','localhost', ['d1'], [], [], True, 10,],


    ]

    """
    #datesに日付をループさせて追加していく
    l = []
    month_term = 6
    start_dt = datetime(2017, 1, 1)
    end = datetime(2017, 7, 1) #終了日付

    end_dt = start_dt + relativedelta(months=month_term)

    while True:
        if end_dt > end:
            break
        tmp_l = [start_dt, end_dt] + [3, 3, [], 2, 10, 30, 300, 'USDJPY', 'localhost', 'localhost', ['d1'], [], [], True]
        l.append(tmp_l)
        start_dt = end_dt
        end_dt = end_dt + relativedelta(months=month_term)
    l.reverse()
    dates = dates + l
    """

    #スタートとエンドだけ試しに表示させる
    for d in dates:
        start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, targets, divs, subs, data_chain, test_term = d
        print(start, end)


    for d in dates:
        start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, targets, divs, subs, data_chain, test_term = d
        make_tsfresh_test(start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, targets, divs, subs, data_chain, test_term)

