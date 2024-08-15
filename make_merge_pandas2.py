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
from util import *
import talib
import csv
import pandas as pd
import pickle
import socket

"""
lgbm用のpandasデータをmerge関数を使わずにマージする
"""
print("START")
start_time = time.perf_counter()

symbol = "USDJPY"

startDt = datetime(2017, 1, 1)
endDt = datetime(2023, 9, 3)
#startDt = datetime(2016, 1, 1)
#endDt = datetime(2023, 7, 29)
start_score = int(time.mktime(startDt.timetuple()))
end_score = int(time.mktime(endDt.timetuple()))

end_tmp = endDt + timedelta(days=-1)

mode = "pickle" #csv:csvだけ保存 pickle:pickleだけ保存
csv_file_name = "/db2/lgbm/" + symbol + "/csv_file/tmp.csv"

if mode == "csv" and csv_file_name == "":
    print("csv_file_name is needed")
    exit(1)


bet_term = 2
data_term = 2

#leftのファイルを基本とする
left_df_file = "CF46"
left_df_file_path = "/db2/lgbm/" + symbol + "/concat_file/" + left_df_file + ".pickle"

right_df_file = "CF54"
right_df_file_path = "/db2/lgbm/" + symbol + "/concat_file/" + right_df_file + ".pickle"

#使用する列名
left_input = ["score","o",]
left_tmp_input = [
    "2-d-1@2-d-10@2-d-100@2-d-101@2-d-102@2-d-103@2-d-104@2-d-105@2-d-106@2-d-107@2-d-108@2-d-109@2-d-11@2-d-110@2-d-111@2-d-112@2-d-113@2-d-114@2-d-115@2-d-116@2-d-117@2-d-118@2-d-119@2-d-12@2-d-120@2-d-121@2-d-122@2-d-123@2-d-124@2-d-125@2-d-126@2-d-127@2-d-128@2-d-129@2-d-13@2-d-130@2-d-131@2-d-132@2-d-133@2-d-134@2-d-135@2-d-136@2-d-137@2-d-138@2-d-139@2-d-14@2-d-140@2-d-141@2-d-142@2-d-143@2-d-144@2-d-145@2-d-146@2-d-147@2-d-148@2-d-149@2-d-15@2-d-150@2-d-151@2-d-152@2-d-153@2-d-154@2-d-155@2-d-156@2-d-157@2-d-158@2-d-159@2-d-16@2-d-160@2-d-161@2-d-162@2-d-163@2-d-164@2-d-165@2-d-166@2-d-167@2-d-168@2-d-169@2-d-17@2-d-170@2-d-171@2-d-172@2-d-173@2-d-174@2-d-175@2-d-176@2-d-177@2-d-178@2-d-179@2-d-18@2-d-180@2-d-181@2-d-182@2-d-183@2-d-184@2-d-185@2-d-186@2-d-187@2-d-188@2-d-189@2-d-19@2-d-190@2-d-191@2-d-192@2-d-193@2-d-194@2-d-195@2-d-196@2-d-197@2-d-198@2-d-199@2-d-2@2-d-20@2-d-200@2-d-201@2-d-202@2-d-203@2-d-204@2-d-205@2-d-206@2-d-207@2-d-208@2-d-209@2-d-21@2-d-210@2-d-211@2-d-212@2-d-213@2-d-214@2-d-215@2-d-216@2-d-217@2-d-218@2-d-219@2-d-22@2-d-220@2-d-221@2-d-222@2-d-223@2-d-224@2-d-225@2-d-226@2-d-227@2-d-228@2-d-229@2-d-23@2-d-230@2-d-231@2-d-232@2-d-233@2-d-234@2-d-235@2-d-236@2-d-237@2-d-238@2-d-239@2-d-24@2-d-240@2-d-241@2-d-242@2-d-243@2-d-244@2-d-245@2-d-246@2-d-247@2-d-248@2-d-249@2-d-25@2-d-250@2-d-251@2-d-252@2-d-253@2-d-254@2-d-255@2-d-256@2-d-257@2-d-258@2-d-259@2-d-26@2-d-260@2-d-261@2-d-262@2-d-263@2-d-264@2-d-265@2-d-266@2-d-267@2-d-268@2-d-269@2-d-27@2-d-270@2-d-271@2-d-272@2-d-273@2-d-274@2-d-275@2-d-276@2-d-277@2-d-278@2-d-279@2-d-28@2-d-280@2-d-281@2-d-282@2-d-283@2-d-284@2-d-285@2-d-286@2-d-287@2-d-288@2-d-289@2-d-29@2-d-290@2-d-291@2-d-292@2-d-293@2-d-294@2-d-295@2-d-296@2-d-297@2-d-298@2-d-299@2-d-3@2-d-30@2-d-300@2-d-31@2-d-32@2-d-33@2-d-34@2-d-35@2-d-36@2-d-37@2-d-38@2-d-39@2-d-4@2-d-40@2-d-41@2-d-42@2-d-43@2-d-44@2-d-45@2-d-46@2-d-47@2-d-48@2-d-49@2-d-5@2-d-50@2-d-51@2-d-52@2-d-53@2-d-54@2-d-55@2-d-56@2-d-57@2-d-58@2-d-59@2-d-6@2-d-60@2-d-61@2-d-62@2-d-63@2-d-64@2-d-65@2-d-66@2-d-67@2-d-68@2-d-69@2-d-7@2-d-70@2-d-71@2-d-72@2-d-73@2-d-74@2-d-75@2-d-76@2-d-77@2-d-78@2-d-79@2-d-8@2-d-80@2-d-81@2-d-82@2-d-83@2-d-84@2-d-85@2-d-86@2-d-87@2-d-88@2-d-89@2-d-9@2-d-90@2-d-91@2-d-92@2-d-93@2-d-94@2-d-95@2-d-96@2-d-97@2-d-98@2-d-99".split("@")
]
for t_l in left_tmp_input:
    left_input.extend(t_l)

right_input = ["score"]
right_tmp_input = [
    "d1__agg_autocorrelation__f_agg_mean__maxlag_40_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_max_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_min_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_50__f_agg_max_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_50__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_50__f_agg_min_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_5__f_agg_max_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_5__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_rvalue__chunk_len_5__f_agg_min_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_10__f_agg_max_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_10__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_10__f_agg_min_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_50__f_agg_max_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_50__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_50__f_agg_min_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_5__f_agg_max_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_5__f_agg_mean_2_30_300@d1__agg_linear_trend__attr_slope__chunk_len_5__f_agg_min_2_30_300@d1__agg_linear_trend__attr_stderr__chunk_len_10__f_agg_min_2_30_300@d1__agg_linear_trend__attr_stderr__chunk_len_50__f_agg_min_2_30_300@d1__agg_linear_trend__attr_stderr__chunk_len_5__f_agg_min_2_30_300@d1__ar_coefficient__coeff_0__k_10_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_0.2__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_0.6__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_1.0__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_1.0__ql_0.2_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_1.0__ql_0.4_2_30_300@d1__change_quantiles__f_agg_mean__isabs_False__qh_1.0__ql_0.8_2_30_300@d1__change_quantiles__f_agg_mean__isabs_True__qh_0.2__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_True__qh_0.4__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_True__qh_0.6__ql_0.0_2_30_300@d1__change_quantiles__f_agg_mean__isabs_True__qh_0.8__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_False__qh_0.2__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_False__qh_0.4__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_False__qh_0.6__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_False__qh_0.8__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_True__qh_0.4__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_True__qh_0.6__ql_0.0_2_30_300@d1__change_quantiles__f_agg_var__isabs_True__qh_0.8__ql_0.0_2_30_300@d1__count_above__t_0_2_30_300@d1__count_above_mean_2_30_300@d1__count_below_mean_2_30_300@d1__cwt_coefficients__coeff_0__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_0__w_20__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_10__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_11__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_1__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_1__w_20__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_2__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_3__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_4__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_5__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_6__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_7__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_8__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__cwt_coefficients__coeff_9__w_10__widths_2_ 5_ 10_ 20_2_30_300@d1__energy_ratio_by_chunks__num_segments_10__segment_focus_9_2_30_300@d1__fft_coefficient__attr_abs__coeff_0_2_30_300@d1__fft_coefficient__attr_abs__coeff_75_2_30_300@d1__fft_coefficient__attr_abs__coeff_84_2_30_300@d1__fft_coefficient__attr_abs__coeff_96_2_30_300@d1__fft_coefficient__attr_angle__coeff_0_2_30_300@d1__fft_coefficient__attr_angle__coeff_10_2_30_300@d1__fft_coefficient__attr_angle__coeff_1_2_30_300@d1__fft_coefficient__attr_angle__coeff_2_2_30_300@d1__fft_coefficient__attr_angle__coeff_3_2_30_300@d1__fft_coefficient__attr_angle__coeff_62_2_30_300@d1__fft_coefficient__attr_angle__coeff_6_2_30_300@d1__fft_coefficient__attr_angle__coeff_8_2_30_300@d1__fft_coefficient__attr_angle__coeff_9_2_30_300@d1__fft_coefficient__attr_imag__coeff_10_2_30_300@d1__fft_coefficient__attr_imag__coeff_11_2_30_300@d1__fft_coefficient__attr_imag__coeff_1_2_30_300@d1__fft_coefficient__attr_imag__coeff_2_2_30_300@d1__fft_coefficient__attr_imag__coeff_3_2_30_300@d1__fft_coefficient__attr_imag__coeff_4_2_30_300@d1__fft_coefficient__attr_imag__coeff_55_2_30_300@d1__fft_coefficient__attr_imag__coeff_63_2_30_300@d1__fft_coefficient__attr_imag__coeff_6_2_30_300@d1__fft_coefficient__attr_imag__coeff_8_2_30_300@d1__fft_coefficient__attr_imag__coeff_9_2_30_300@d1__fft_coefficient__attr_real__coeff_0_2_30_300@d1__fft_coefficient__attr_real__coeff_1_2_30_300@d1__fft_coefficient__attr_real__coeff_2_2_30_300@d1__fft_coefficient__attr_real__coeff_50_2_30_300@d1__fft_coefficient__attr_real__coeff_66_2_30_300@d1__first_location_of_maximum_2_30_300@d1__first_location_of_minimum_2_30_300@d1__friedrich_coefficients__coeff_0__m_3__r_30_2_30_300@d1__has_duplicate_2_30_300@d1__has_duplicate_max_2_30_300@d1__has_duplicate_min_2_30_300@d1__index_mass_quantile__q_0.9_2_30_300@d1__large_standard_deviation__r_0.05_2_30_300@d1__large_standard_deviation__r_0.15000000000000002_2_30_300@d1__large_standard_deviation__r_0.1_2_30_300@d1__last_location_of_maximum_2_30_300@d1__last_location_of_minimum_2_30_300@d1__linear_trend__attr_rvalue_2_30_300@d1__linear_trend__attr_slope_2_30_300@d1__longest_strike_above_mean_2_30_300@d1__longest_strike_below_mean_2_30_300@d1__mean_2_30_300@d1__mean_change_2_30_300@d1__median_2_30_300@d1__minimum_2_30_300@d1__number_crossing_m__m_-1_2_30_300@d1__quantile__q_0.1_2_30_300@d1__quantile__q_0.3_2_30_300@d1__quantile__q_0.4_2_30_300@d1__range_count__max_0__min_-1000000000000.0_2_30_300@d1__range_count__max_1000000000000.0__min_0_2_30_300@d1__skewness_2_30_300@d1__sum_values_2_30_300@d1__time_reversal_asymmetry_statistic__lag_1_2_30_300@d1__variance_larger_than_standard_deviation_2_30_300@d1__variation_coefficient_2_30_300".split("@")
]
for t_l in right_tmp_input:
    right_input.extend(t_l)

on_col = "score"

#ファイル読み込み
with open(left_df_file_path, 'rb') as f:
    left_df = pickle.load(f)
    print("left_df info")
    print(left_df.info())
    print(left_df[:100])
    # 開始、終了期間で絞る
    left_df.query('@start_score <= score < @end_score', inplace=True)
    print("memory1.1", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

print(left_df.info())

with open(right_df_file_path, 'rb') as f:
    right_df = pickle.load(f)
    print("right_df info")
    print(right_df.info())
    print(right_df[:100])
    # 開始、終了期間で絞る
    right_df.query('@start_score <= score < @end_score', inplace=True)
    print("memory1.2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")


print("left_df cnt ",len(left_df.index))
print("right_df cnt ",len(right_df.index))


#カラム抽出
left_df = left_df.loc[:, left_input]
print("memory1.3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
right_df = right_df.loc[:, right_input]
print("memory1.4", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")


left_df.set_index('score',drop=False, inplace=True)
print("memory1.5", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")
right_df.set_index('score',drop=False, inplace=True)
print("memory1.6", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

for col in right_tmp_input:
    left_df.loc[:, col] = None  # 初期化
print("memory1.7", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

#マージ実施
for index, row in left_df.iterrows():
    try:
        right_row = right_df.loc[index, :]
        for col in left_tmp_input:
            left_df.at[index, col] = right_row[col]
    except Exception as e:
        pass

print("merge finished")
print("memory1.8", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

del right_df
gc.collect()

print("gc finished")
print("memory1.9", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

left_df = left_df.sort_index(ascending=True,inplace=True)  # scoreの昇順　古い順にする
print("memory2", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

print("left_df info")
print(left_df.info())

print("memory3", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

print(left_df[:100])
print(left_df[-100:])
csv_regist_cols = left_df.columns.tolist()


csv_regist_cols.sort()  # カラムを名前順にする
# o,scoreは最後にしたいので削除後、追加
if 'o' in csv_regist_cols:
    csv_regist_cols.remove('o')
    csv_regist_cols.append('o')
if 'score' in csv_regist_cols:
    csv_regist_cols.remove('score')
    csv_regist_cols.append('score')

print("cols", csv_regist_cols)

if mode == "csv":

    if os.path.isfile(csv_file_name):
        # すでにファイルがあるなら追記
        left_df.to_csv(csv_file_name, index=False, mode="a", header=False)
    else:
        # 指定したパスが存在しない場合は新規作成、存在する場合は上書き
        left_df.to_csv(csv_file_name, index=False, mode="w")

    print("csv make finished")
    print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

elif mode == "pickle":

    input_name = list_to_str(csv_regist_cols, "@")
    merge_files = "_MFS-" + list_to_str([left_df_file, right_df_file])

    tmp_file_name = symbol + "_B" + str(bet_term) + "_D" + str(data_term) + "_IN-" + input_name + "_" + \
                    date_to_str(startDt, format='%Y%m%d') + "-" + date_to_str(end_tmp,
                                                                              format='%Y%m%d') + merge_files + "_" + socket.gethostname()

    db_name_file = "MERGE_FILE_NO_" + symbol
    # win2のDBを参照してモデルのナンバリングを行う
    r = redis.Redis(host='192.168.1.114', port=6379, db=1, decode_responses=True)
    result = r.zrevrange(db_name_file, 0, -1, withscores=True)  # 全件取得
    if len(result) == 0:
        print("CANNOT GET MERGE_FILE_NO")
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

    tmp_path = "/db2/lgbm/" + symbol + "/merge_file/" + "MF" + str(newest_no)
    pickle_file_name = tmp_path + ".pickle"

    print("newest_no", newest_no)
    print("input_name", tmp_file_name)
    tmp_dict = {}
    for col in csv_regist_cols:
        if col != 'score' and col != 'o' and ("hor" in col) == False:
            tmp_dict[col] = 'float32'
    # score以外をfloat32に型変換
    left_df = left_df.astype(tmp_dict, copy=False)

    left_df.to_pickle(pickle_file_name)
    print("save pickle finished")
    print("memory", psutil.virtual_memory().available / 1024 / 1024 / 1024, "GB")

else:
    print("mode is incorrect")
    exit(1)

print(datetime.now())
print("FINISH")
print("Processing Time(Sec)", time.perf_counter() - start_time)