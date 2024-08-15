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
def make_tsfresh(start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec,
                 data_len, symbol, org_db, new_db, fc_parameters, targets, divs, subs, data_chain, fill_in):
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
    output("fc_parameters:",fc_parameters)
    output("inputs:",inputs)
    output("targets:",targets)
    output("divs:",divs)
    output("subs:",subs)
    output("fill_in:", fill_in)

    output("")

    redis_db_org = redis.Redis(host=org_db, port=6379, db=org_db_no, decode_responses=True)

    dfs = []

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

        id = 0
        sc_list = []
        open_list = []

        #以下は特徴量算出のためのdataframeをつくるためのデータ

        df_dict = {
            'id':[],
            'time':[],
        }
        for k in targets:
            df_dict[k] = []

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

            id += 1
            open = lists_dict['c'][end_idx]

            sc_list.append(score)
            open_list.append(open)

            for j, k in enumerate(targets):

                tmp_list = lists_dict[k]
                x_list = tmp_list[start_idx: (end_idx + 1)]
                for l, x in enumerate(x_list):
                    df_dict[k].append(x)
                    if j == 0:
                        df_dict['id'].append(id)
                        df_dict['time'].append(l)

        del lists_dict

        o_index = [i + 1 for i in range(len(open_list))]
        x_df = pd.DataFrame(df_dict)

        del df_dict
        gc.collect()
        #print(y_series)
        #print(x_df)

        #特徴量作成
        start_time = time.perf_counter()

        #選別された特徴量のみ作成
        features_filtered = extract_features(
            x_df,
            kind_to_fc_parameters=fc_parameters,
            column_id='id',
            column_sort='time',
        )
        output("features_filtered.info")
        output(features_filtered.info)
        output("features_filtered takes:", time.perf_counter() - start_time)

        #欠損値、異常値（無限大）を処理する関数 select_features時にnanがあるとエラーになる
        start_time = time.perf_counter()
        impute(features_filtered)
        output("impute takes:", time.perf_counter() - start_time)

        features_filtered_col_list = features_filtered.columns.tolist()
        output(list_to_str(features_filtered_col_list, spl="@"))

        print("memory1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        open_series = pd.Series(data=open_list, name='o', index=o_index)
        score_series = pd.Series(data=sc_list, name='score', index=o_index)

        # ダブルクォートはLGBMで使えないので置換
        col_name_org = features_filtered.columns.tolist()
        col_name_new = {}
        for col in col_name_org:
            col_n = col.replace('"', '')
            col_n = col_n.replace(',', '_')  # LGBMで使えないので置換
            col_n = col_n.replace('(', '')  # LGBMで使えないので置換
            col_n = col_n.replace(')', '')  # LGBMで使えないので置換
            col_n = col_n.replace(' ', '')  # LGBMで使えないので置換
            col_name_new[col] = col_n + '_' + str(term) + '_' + str(pred_sec) + '_' + str(data_len) #カラム名がtermが違う場合に重複することを防ぐ為にカラム名を変更
            if fill_in:
                col_name_new[col] = col_name_new[col] + '_foot'

        features_filtered.rename(columns=col_name_new, inplace=True)

        if fill_in:
            #次のスコアまで間を埋める
            cols = features_filtered.columns.tolist()

            for index, row in features_filtered.iterrows():
                tmp_l = []
                for col in cols:
                    tmp_l.append(row[col])
                tmp_score = index
                while True:
                    tmp_score = tmp_score + bet_term
                    if tmp_score >= index + term:
                        break

                    features_filtered.loc[tmp_score] = tmp_l

        new_df = pd.concat([score_series, open_series, features_filtered], axis=1)
        del open_series, score_series, features_filtered
        gc.collect()

        print("concat finish. memory2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

        dfs.append(new_df)


    #すでに作成したdfから一気にpickleにする
    new_df = pd.concat(dfs, axis=0)
    del dfs
    gc.collect()

    # 欠損値がある行を削除
    new_df.dropna(inplace=True)
    new_df.sort_values('score', ascending=True,inplace=True)  # scoreの昇順　古い順にする
    new_df.set_index('score', drop=False,inplace=True)

    output("new_df.info")
    output(new_df.info)

    csv_regist_cols = new_df.columns.tolist()

    csv_regist_cols.sort()#カラムを名前順にする
    # o,scoreは最後にしたいので削除後、追加
    if 'o' in csv_regist_cols:
        csv_regist_cols.remove('o')
        csv_regist_cols.append('o')
    if 'score' in csv_regist_cols:
        csv_regist_cols.remove('score')
        csv_regist_cols.append('score')

    print(new_df[:100])
    print(new_df[-100:])

    input_name = list_to_str(csv_regist_cols, "@")
    end_tmp = end + timedelta(days=-1)

    tmp_file_name = symbol + "_T" + str(term) + "_DL" + str(data_len) + "_IN-" + input_name + "_" + \
                    date_to_str(start,format='%Y%m%d') + "-" + date_to_str(end_tmp,format='%Y%m%d') + "_" + socket.gethostname()
    db_name_file = "TSFRESH_FILE_NO_" + symbol
    # win2のDBを参照してモデルのナンバリングを行う
    r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
    result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
    if len(result) == 0:
        print("CANNOT GET TSFRESH_FILE_NO")
        exit(1)
    else:
        newest_no = int(result[0][1]) + 1  # 次に割り振る番号(最新に1足す)

        for line in result:
            body = line[0]
            score = int(line[1])
            tmps = json.loads(body)
            tmp_name = tmps.get("input_name")
            if tmp_name == tmp_file_name:
                # 同じファイルがないが確認
                print("The File Already Exists!!!")
                exit(1)

        # DBにモデルを登録
        child = {
            'input_name': tmp_file_name,
            'no': newest_no
        }
        r.zadd(db_name_file, json.dumps(child), newest_no)

        csv_path = "/db2/lgbm/" + symbol + "/tsfresh_file/" + "TSF" + str(newest_no)
        pickle_file_name = csv_path + ".pickle"

        output("file_name", "TSF" + str(newest_no))
        output("input_name",tmp_file_name)
        output("new_df.info")
        output(new_df.info)

        tmp_dict = {}
        for col in csv_regist_cols:
            if col != 'score' and col != 'o':
                tmp_dict[col] = 'float32'
        # score以外をfloat32に型変換
        new_df = new_df.astype(tmp_dict)

        print("memory3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
        new_df.to_pickle(pickle_file_name)

    output("total takes:", time.perf_counter() - total_start_time)



if __name__ == '__main__':

    #作成する特徴量がわかっている場合
    #for sec2 data_len300
    #selected_features = 'd1__large_standard_deviation__r_0.15000000000000002@d1__large_standard_deviation__r_0.1@d1__variance_larger_than_standard_deviation@d1__fft_coefficient__attr_"angle"__coeff_0@d1__ar_coefficient__coeff_0__k_10@d1__fft_coefficient__attr_"real"__coeff_0@d1__sum_values@d1__mean@d1__linear_trend__attr_"slope"@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"@d1__linear_trend__attr_"rvalue"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"@d1__skewness@d1__fft_coefficient__attr_"imag"__coeff_10@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"@d1__has_duplicate_min@d1__large_standard_deviation__r_0.05@d1__has_duplicate_max@d1__longest_strike_above_mean@d1__count_below_mean@d1__count_above_mean@d1__longest_strike_below_mean@d1__fft_coefficient__attr_"real"__coeff_1@d1__fft_coefficient__attr_"angle"__coeff_10@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"@d1__fft_coefficient__attr_"imag"__coeff_2@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"@d1__fft_coefficient__attr_"imag"__coeff_8@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0@d1__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"@d1__fft_coefficient__attr_"real"__coeff_2@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0@d1__has_duplicate@d1__fft_coefficient__attr_"imag"__coeff_55@d1__last_location_of_maximum@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2@d1__fft_coefficient__attr_"imag"__coeff_1@d1__first_location_of_maximum@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0@d1__fft_coefficient__attr_"imag"__coeff_9@d1__minimum@d1__fft_coefficient__attr_"imag"__coeff_6@d1__fft_coefficient__attr_"angle"__coeff_2@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"@d1__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0@d1__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"abs"__coeff_75@d1__number_crossing_m__m_-1@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0@d1__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min"@d1__last_location_of_minimum@d1__fft_coefficient__attr_"imag"__coeff_3@d1__first_location_of_minimum@d1__fft_coefficient__attr_"angle"__coeff_1@d1__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)@d1__friedrich_coefficients__coeff_0__m_3__r_30@d1__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0@d1__fft_coefficient__attr_"angle"__coeff_8@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8@d1__quantile__q_0.1@d1__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0@d1__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"abs"__coeff_96@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0@d1__count_above__t_0@d1__range_count__max_0__min_-1000000000000.0@d1__range_count__max_1000000000000.0__min_0@d1__variation_coefficient@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0@d1__fft_coefficient__attr_"imag"__coeff_11@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0@d1__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"abs"__coeff_84@d1__fft_coefficient__attr_"imag"__coeff_4@d1__fft_coefficient__attr_"imag"__coeff_63@d1__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"angle"__coeff_6@d1__fft_coefficient__attr_"angle"__coeff_62@d1__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"min"@d1__energy_ratio_by_chunks__num_segments_10__segment_focus_9@d1__median@d1__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"angle"__coeff_9@d1__index_mass_quantile__q_0.9@d1__mean_change@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0@d1__quantile__q_0.4@d1__fft_coefficient__attr_"abs"__coeff_0@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4@d1__fft_coefficient__attr_"angle"__coeff_3@d1__fft_coefficient__attr_"real"__coeff_50@d1__agg_autocorrelation__f_agg_"mean"__maxlag_40@d1__fft_coefficient__attr_"real"__coeff_66@d1__quantile__q_0.3@d1__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)@d1__time_reversal_asymmetry_statistic__lag_1@d1__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)'
    #for sec10 data_len300
    #selected_features = 'd1__large_standard_deviation__r_0.15000000000000002@d1__variance_larger_than_standard_deviation@d1__has_duplicate@d1__fft_coefficient__attr_"angle"__coeff_0@d1__large_standard_deviation__r_0.1@d1__linear_trend__attr_"slope"@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"@d1__linear_trend__attr_"rvalue"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"@d1__ar_coefficient__coeff_0__k_10@d1__fft_coefficient__attr_"real"__coeff_0@d1__mean@d1__sum_values@d1__fft_coefficient__attr_"imag"__coeff_50@d1__fft_coefficient__attr_"real"__coeff_1@d1__mean_change@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0@d1__skewness@d1__count_below_mean@d1__count_above_mean@d1__large_standard_deviation__r_0.05@d1__fft_coefficient__attr_"imag"__coeff_2@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"@d1__fft_coefficient__attr_"angle"__coeff_50@d1__median@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"@d1__fft_coefficient__attr_"imag"__coeff_1@d1__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"@d1__fft_coefficient__attr_"imag"__coeff_7@d1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"@d1__time_reversal_asymmetry_statistic__lag_2@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"@d1__linear_trend__attr_"intercept"@d1__fft_coefficient__attr_"real"__coeff_10@d1__range_count__max_0__min_-1000000000000.0@d1__range_count__max_1000000000000.0__min_0@d1__count_above__t_0@d1__time_reversal_asymmetry_statistic__lag_1@d1__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2@d1__fft_coefficient__attr_"real"__coeff_2@d1__friedrich_coefficients__coeff_3__m_3__r_30@d1__fft_coefficient__attr_"imag"__coeff_13@d1__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"@d1__longest_strike_below_mean@d1__fft_coefficient__attr_"imag"__coeff_4@d1__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min"@d1__fft_coefficient__attr_"angle"__coeff_13@d1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"@d1__fft_coefficient__attr_"imag"__coeff_14@d1__fft_coefficient__attr_"imag"__coeff_3@d1__time_reversal_asymmetry_statistic__lag_3@d1__fft_coefficient__attr_"real"__coeff_4@d1__fft_coefficient__attr_"imag"__coeff_39@d1__number_peaks__n_3@d1__longest_strike_above_mean@d1__fft_coefficient__attr_"imag"__coeff_19@d1__fft_coefficient__attr_"angle"__coeff_2@d1__fft_coefficient__attr_"imag"__coeff_31@d1__fft_coefficient__attr_"real"__coeff_3@d1__fft_coefficient__attr_"real"__coeff_9@d1__fft_coefficient__attr_"imag"__coeff_6@d1__fft_coefficient__attr_"angle"__coeff_1@d1__fft_coefficient__attr_"real"__coeff_54@d1__fft_coefficient__attr_"real"__coeff_57@d1__fft_coefficient__attr_"imag"__coeff_10@d1__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"@d1__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"mean"@d1__fft_coefficient__attr_"imag"__coeff_16@d1__fft_coefficient__attr_"imag"__coeff_9@d1__fft_coefficient__attr_"angle"__coeff_7@d1__fft_coefficient__attr_"imag"__coeff_30@d1__fft_coefficient__attr_"angle"__coeff_10@d1__count_below__t_0@d1__fft_coefficient__attr_"real"__coeff_8@d1__fft_coefficient__attr_"imag"__coeff_21@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0@d1__fft_coefficient__attr_"angle"__coeff_39@d1__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8@d1__fft_coefficient__attr_"abs"__coeff_12@d1__large_standard_deviation__r_0.2@d1__fft_coefficient__attr_"real"__coeff_21@d1__last_location_of_minimum@d1__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"@d1__first_location_of_minimum@d1__maximum@d1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8@d1__quantile__q_0.4@d1__fft_coefficient__attr_"imag"__coeff_17@d1__fft_coefficient__attr_"real"__coeff_49@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8@d1__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"@d1__fft_coefficient__attr_"imag"__coeff_28@d1__fft_coefficient__attr_"real"__coeff_15@d1__fft_coefficient__attr_"angle"__coeff_4@d1__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2@d1__fft_coefficient__attr_"angle"__coeff_19@d1__fft_coefficient__attr_"abs"__coeff_98@d1__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"@d1__permutation_entropy__dimension_3__tau_1@d1__fft_coefficient__attr_"angle"__coeff_3@d1__augmented_dickey_fuller__attr_"usedlag"__autolag_"AIC"@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"@d1__fft_coefficient__attr_"real"__coeff_17'

    # for sec2 data_len15
    selected_features = 'd1__ratio_beyond_r_sigma__r_3@d1__variance_larger_than_standard_deviation@d1__has_duplicate_max@d1__has_duplicate_min@d1__has_duplicate@d1__symmetry_looking__r_0.2@d1__symmetry_looking__r_0.25@d1__symmetry_looking__r_0.30000000000000004@d1__symmetry_looking__r_0.35000000000000003@d1__symmetry_looking__r_0.4@d1__symmetry_looking__r_0.45@d1__symmetry_looking__r_0.5@d1__symmetry_looking__r_0.6000000000000001@d1__symmetry_looking__r_0.55@d1__symmetry_looking__r_0.7000000000000001@d1__symmetry_looking__r_0.75@d1__symmetry_looking__r_0.8@d1__symmetry_looking__r_0.8500000000000001@d1__symmetry_looking__r_0.9@d1__symmetry_looking__r_0.9500000000000001@d1__large_standard_deviation__r_0.05@d1__large_standard_deviation__r_0.1@d1__large_standard_deviation__r_0.15000000000000002@d1__large_standard_deviation__r_0.2@d1__large_standard_deviation__r_0.25@d1__symmetry_looking__r_0.65@d1__fft_coefficient__attr_"angle"__coeff_0@d1__symmetry_looking__r_0.15000000000000002@d1__fft_coefficient__attr_"real"__coeff_0@d1__sum_values@d1__mean@d1__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)@d1__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"@d1__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)@d1__large_standard_deviation__r_0.30000000000000004@d1__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)@d1__symmetry_looking__r_0.05@d1__median@d1__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)@d1__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"@d1__number_peaks__n_5@d1__minimum@d1__cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)@d1__friedrich_coefficients__coeff_3__m_3__r_30@d1__quantile__q_0.4@d1__quantile__q_0.1@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0@d1__number_crossing_m__m_-1@d1__skewness@d1__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0@d1__symmetry_looking__r_0.1@d1__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0@d1__large_standard_deviation__r_0.35000000000000003@d1__quantile__q_0.3@d1__cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0@d1__linear_trend__attr_"intercept"@d1__quantile__q_0.2@d1__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)@d1__quantile__q_0.6@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0@d1__cwt_coefficients__coeff_0__w_5__widths_(2, 5, 10, 20)@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"@d1__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0@d1__maximum@d1__time_reversal_asymmetry_statistic__lag_2@d1__count_above_mean@d1__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"@d1__quantile__q_0.7@d1__count_below_mean@d1__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"@d1__longest_strike_above_mean@d1__number_crossing_m__m_1@d1__range_count__max_0__min_-1000000000000.0@d1__range_count__max_1000000000000.0__min_0@d1__count_above__t_0@d1__large_standard_deviation__r_0.4@d1__quantile__q_0.9@d1__longest_strike_below_mean@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4@d1__fft_coefficient__attr_"abs"__coeff_4@d1__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6@d1__sum_of_reoccurring_data_points@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6@d1__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2@d1__sum_of_reoccurring_values@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6@d1__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2@d1__max_langevin_fixed_point__m_3__r_30@d1__absolute_maximum@d1__fft_coefficient__attr_"abs"__coeff_1@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0@d1__root_mean_square@d1__abs_energy@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8@d1__variance@d1__standard_deviation@d1__range_count__max_1__min_-1@d1__number_crossing_m__m_0@d1__quantile__q_0.8@d1__linear_trend__attr_"stderr"@d1__fft_coefficient__attr_"abs"__coeff_5@d1__fft_coefficient__attr_"abs"__coeff_6@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4@d1__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0@d1__mean_n_absolute_max__number_of_maxima_7@d1__fft_coefficient__attr_"abs"__coeff_2@d1__cid_ce__normalize_False@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"@d1__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0@d1__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)@d1__c3__lag_3@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2@d1__absolute_sum_of_changes@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0@d1__mean_abs_change@d1__fft_coefficient__attr_"abs"__coeff_7@d1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2@d1__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"@d1__autocorrelation__lag_6@d1__spkt_welch_density__coeff_5@d1__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)@d1__approximate_entropy__m_2__r_0.1@d1__benford_correlation@d1__percentage_of_reoccurring_values_to_all_values@d1__c3__lag_2@d1__number_peaks__n_1@d1__fft_coefficient__attr_"abs"__coeff_0@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6@d1__energy_ratio_by_chunks__num_segments_10__segment_focus_8'
    # for sec10 data_len3
    #selected_features = 'd1__lempel_ziv_complexity__bins_100@d1__variance_larger_than_standard_deviation@d1__has_duplicate_max@d1__has_duplicate_min@d1__has_duplicate@d1__symmetry_looking__r_0.30000000000000004@d1__symmetry_looking__r_0.35000000000000003@d1__symmetry_looking__r_0.4@d1__symmetry_looking__r_0.45@d1__symmetry_looking__r_0.5@d1__symmetry_looking__r_0.55@d1__symmetry_looking__r_0.6000000000000001@d1__symmetry_looking__r_0.65@d1__lempel_ziv_complexity__bins_10@d1__symmetry_looking__r_0.75@d1__symmetry_looking__r_0.7000000000000001@d1__symmetry_looking__r_0.8500000000000001@d1__large_standard_deviation__r_0.4@d1__symmetry_looking__r_0.8@d1__large_standard_deviation__r_0.30000000000000004@d1__large_standard_deviation__r_0.25@d1__large_standard_deviation__r_0.2@d1__large_standard_deviation__r_0.35000000000000003@d1__large_standard_deviation__r_0.1@d1__large_standard_deviation__r_0.05@d1__symmetry_looking__r_0.9500000000000001@d1__symmetry_looking__r_0.9@d1__large_standard_deviation__r_0.15000000000000002@d1__symmetry_looking__r_0.25@d1__fft_coefficient__attr_"angle"__coeff_0@d1__large_standard_deviation__r_0.45@d1__symmetry_looking__r_0.2@d1__lempel_ziv_complexity__bins_5@d1__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)@d1__fft_coefficient__attr_"real"__coeff_0@d1__sum_values@d1__mean@d1__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)@d1__quantile__q_0.3@d1__quantile__q_0.2@d1__quantile__q_0.4@d1__number_cwt_peaks__n_5@d1__quantile__q_0.1@d1__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)@d1__quantile__q_0.7@d1__symmetry_looking__r_0.15000000000000002@d1__quantile__q_0.6@d1__minimum@d1__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_5__widths_(2, 5, 10, 20)@d1__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)@d1__quantile__q_0.8@d1__median@d1__number_crossing_m__m_-1@d1__quantile__q_0.9@d1__symmetry_looking__r_0.1@d1__linear_trend__attr_"intercept"@d1__friedrich_coefficients__coeff_3__m_3__r_30@d1__maximum@d1__lempel_ziv_complexity__bins_3@d1__number_crossing_m__m_1@d1__range_count__max_0__min_-1000000000000.0@d1__range_count__max_1000000000000.0__min_0@d1__count_above__t_0@d1__number_cwt_peaks__n_1@d1__number_peaks__n_1@d1__count_below__t_0@d1__mean_abs_change@d1__absolute_sum_of_changes@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0@d1__max_langevin_fixed_point__m_3__r_30@d1__fft_coefficient__attr_"abs"__coeff_1@d1__standard_deviation@d1__variance@d1__cid_ce__normalize_False@d1__absolute_maximum@d1__abs_energy@d1__root_mean_square@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0@d1__range_count__max_1__min_-1@d1__value_count__value_0@d1__index_mass_quantile__q_0.7@d1__linear_trend__attr_"stderr"@d1__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0@d1__variation_coefficient@d1__fft_coefficient__attr_"abs"__coeff_0@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0@d1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0@d1__fft_coefficient__attr_"real"__coeff_1'

    if selected_features != '':
        selected_feature_list = selected_features.split("@")
        print("selected_features length:",len(selected_feature_list))
        fc_parameters = feature_extraction.settings.from_columns(selected_feature_list)
    else:
        fc_parameters = {}

    # start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, fc_parameters, targets, divs, subs, mode, data_chain をリストで保持させる
    # db_list:抽出元のDB名のリスト。空のリストの場合はbet_termとtermから自動で抽出元が決定される。空でない場合はbet_termは不要なのでNoneにする
    # data_len:特徴量算出のもとになる時系列データの数
    # targets:特徴量計算元の特徴量
    # div:divをとる特徴量
    # subs:subをとる特徴量
    # data_chain:データが続いていなければ対象としない。分足などは続いていなくとも(休日を挟んでいても)OKなのでFalseにする
    # fill_in: データ作成元がFOOTの場合などに、次の行のスコアまでの間をbet_term刻みでおなじ行を追加し他のデータとマージ出来るようにする
    dates = [
        [datetime(2023, 1, 1), datetime(2023, 11, 4), 2, 2, [], 2, 2, 30, 15, 'USDJPY', 'localhost','localhost', fc_parameters, ['d1'], [], [], True, False,],
        #[datetime(2023, 1, 1), datetime(2023, 3, 1), 2, 2, [], 2, 2, 30, 300, 'USDJPY', 'localhost', 'localhost',fc_parameters, ['d1'], [], [], True, False,],

    ]

    #datesに日付をループさせて追加していく
    l = []
    month_term = 6
    start_dt = datetime(2017, 1, 1)
    end = datetime(2023, 1, 1) #終了日付

    end_dt = start_dt + relativedelta(months=month_term)

    while True:
        if end_dt > end:
            break
        tmp_l = [start_dt, end_dt] + [3, 3, [], 2, 2, 30, 15, 'USDJPY', 'localhost', 'localhost',fc_parameters, ['d1'], [], [], True, False, ]
        l.append(tmp_l)
        start_dt = end_dt
        end_dt = end_dt + relativedelta(months=month_term)
    l.reverse()
    dates = dates + l


    #スタートとエンドだけ試しに表示させる
    for d in dates:
        start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, fc_parameters, targets, divs, subs, data_chain, fill_in = d
        print(start, end)


    for d in dates:
        start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, fc_parameters, targets, divs, subs, data_chain, fill_in = d
        make_tsfresh(start, end, org_db_no, new_db_no, db_list, bet_term, term, pred_sec, data_len, symbol, org_db, new_db, fc_parameters, targets, divs, subs, data_chain, fill_in)

