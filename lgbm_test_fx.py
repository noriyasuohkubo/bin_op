import pickle
import time
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

from important_index import *

host = socket.gethostname()
output_log_name = "/home/reicou/tmp_" + host + "-lgbm.txt"
output = output_log(output_log_name)

png_dir = "/app/fx/png/"


def do_test(conf, test_conf, test_lmd, start_dt, end_dt):
    start_time = time.perf_counter()

    df = test_lmd.get_x()

    answer_rate_up_list = test_lmd.get_y_up_answer_rate()
    answer_score_up_list = test_lmd.get_y_up_answer_score()
    answer_rate_dw_list = test_lmd.get_y_dw_answer_rate()
    answer_score_dw_list = test_lmd.get_y_dw_answer_score()

    x = df.loc[:, conf.INPUT_DATA]

    # 全予想時レート(予想対象外含む)
    close_list = test_lmd.get_close_list()
    # 全決済時レート(予想対象外含む)
    answer_list = test_lmd.get_answer_list()
    # 全スコア(予想対象外含む)
    score_list = test_lmd.get_score_list()
    # 全spread値のリスト
    spread_list = test_lmd.get_spread_list()
    #予想対象リスト 予想対象の場合-1が入っている
    train_list_index = test_lmd.get_train_list_index()
    # 全jpy値のリスト
    jpy_list = test_lmd.get_jpy_list()

    #長さチェック
    if len(close_list) != len(answer_list) or len(close_list) != len(score_list) or len(close_list) != len(spread_list) or len(close_list) != len(train_list_index) \
        or len(close_list) != len(jpy_list):
        print("total list length is wrong!!!", len(close_list), len(answer_list), len(score_list), len(spread_list), len(train_list_index), len(jpy_list), )
        exit(1)

    ##以下、予想対象のみ
    # 予想対象のスコアのリスト
    target_score_list = np.array(df.index)
    # 予想対象の予想時レート
    pred_close_list = close_list[np.where(train_list_index != -1)[0]]
    # 予想対象の決済時レート
    real_close_list = answer_list[np.where(train_list_index != -1)[0]]

    sub_close_list = real_close_list - pred_close_list

    # 予想対象のspread値のリスト
    target_spread_list = spread_list[np.where(train_list_index != -1)[0]]

    # 予想対象のjpyのリスト
    target_jpy_list = jpy_list[np.where(train_list_index != -1)[0]]

    #長さチェック
    if len(target_score_list) != len(pred_close_list) or len(target_score_list) != len(real_close_list) or len(target_score_list) != len(target_spread_list) \
        or len(target_score_list) != len(target_jpy_list):
        print("list length is wrong!!!", len(target_score_list), len(pred_close_list), len(real_close_list), len(target_spread_list), len(target_jpy_list),  )
        exit(1)

    # 予想時に持てるポジション
    if conf.JPY_FLG == False:
        position_list = np.array(conf.get_fx_position_jpy(pred_close_list, target_jpy_list))
    else:
        position_list = np.array(conf.get_fx_position(pred_close_list))

    #参照したい列を取得
    target_ind_cols = []
    if len(target_ind_cols) != 0:
        target_ind_list = df[target_ind_cols].values
    else:
        target_ind_list = np.full(len(target_score_list), None) #Noneで他のリストと数を合わせる

    data_length = len(df.index)

    if data_length != 0:
        output("data_length:", data_length)
        output("UP: ", test_lmd.up/data_length)
        output("SAME: ", test_lmd.same/data_length)
        output("DOWN: ", test_lmd.dw/data_length)
        output("up_take_profit_rate:", test_lmd.up_take_profit_cnt/data_length)
        output("dw_take_profit_rate: ", test_lmd.dw_take_profit_cnt/data_length)

    # CATEGORY_BIN_BOTHの場合はupとdwのモデルをリストにする
    FILE_PREFIXS = [
        [
            "USDJPY_LT3_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN194",
            "USDJPY_LT4_M7_LSTM1_B2_T30_I2-10-60-300_IL300-300-240-48_LU30-30-24-5_DU96-48-24-12_BNL2_BDIV0.01_201001_202210_L-RATE0.002_LT1_ADAM_d1-M1_OT-d_OD-c_IDL1_BS15360_SD0_SHU1_EL20-21-22_ub1_MN196",
        ]
    ]

    FILE_PREFIXS = [
        "MN923",
    ]

    if len(FILE_PREFIXS) == 0:
        FILE_PREFIXS = [conf.FILE_PREFIX]

    border_list = [0.47,0.48,0.49,0.50, 0.51,  ]

    if conf.LEARNING_TYPE in ["CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_UP", "CATEGORY_BIN_DW",]:
        #border_list = [0.36,0.38,0.4,0.42,0.44,0.46,0.48, 0.49,0.5,  ]
        border_list = [0.5, 0.52,0.54,0.56,]
        #border_list = [0.62]
        #border_list = [ 0.56,0.57,0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64,  ]
        #border_list = [ 0.58, 0.59,0.6, 0.61, 0.62, 0.64,0.66,0.68,0.7, ]
        #border_list = [ 0.46,0.48, 0.5,] #for ADJUST_PIPS:0.0
        #border_list = [ 0.56, 0.58, 0.6, 0.62, 0.64,0.66,0.68,0.7, ]

    elif conf.LEARNING_TYPE in ["REGRESSION",]:
        #border_list = [2, 4, 6, 8, 10, 12, 14, 16 ]
        border_list = [0.001,0.002, 0.004, 0.006, 0.008, ]

    iteration_range_s = 141
    iteration_range_e = 141
    skip = 10
    output("iteration_range_s:",iteration_range_s)
    output("iteration_range_e:",iteration_range_e)
    output("iteration skip:",skip)
    output("border_list:", border_list)

    model_suffix = [str(i) for i in range(iteration_range_s, iteration_range_e+1, skip)] #start,end+1,skip
    #model_suffix = [str(i+1) for i in range(10)] + [str(i) for i in range(iteration_range_s, iteration_range_e+1, skip)] #start,end+1,skip
    #model_suffix = ["50000"]

    #model_suffix = [["35", [0, ]], ] #特定のsuffixとborder_listを組みでテストする場合
    #model_suffix = [[["33", "35"], [[0.55, 0.55],  ]], ]#特定のsuffixとborder_listを組みでテストする場合 both用
    show_profit_ind = False
    show_profit_ind_up_dw = False
    show_profit_time = False
    show_profit_per_spread = False

    show_plot = False
    show_total = True #borderやsuffix全体の成績順位を表示する場合 True. 特定のsuffixとborder_listを組みでテストする場合はFalseになる
    save_dir = None

    #重要指標の時間帯を除外してテストする
    #important_index_list = []
    important_index_list = ["雇用統計", "CPI", "ISM製造業景況指数","GDP", "ADP雇用統計", "ISM非製造業景況指数", "小売売上高", "新築住宅販売件数", "個人消費支出", "FOMC金利発表", "日銀政策金利発表",]
    important_index_range = 300 #除外する前後の時間秒

    if show_plot:
        # png保存用のディレクトリ作成
        save_dir = png_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
        makedirs(save_dir)
        output("PNG SAVE DIR:", save_dir)

    output("")

    result_per_suffix_border = {}

    total_accuacy_txt = []
    total_loss_txt = []
    output("TERM:", conf.TERM)
    output("FX_FUND:", conf.FX_FUND)
    output("FX_LEVERAGE:", conf.FX_LEVERAGE)
    output("FX_FIX_POSITION:", conf.FX_FIX_POSITION)
    output("FX_SINGLE_FLG:", conf.FX_SINGLE_FLG)
    output("TRADE_SHIFT:", conf.TRADE_SHIFT)
    if conf.SYMBOL == "BTCUSD":
        output("BTCUSD_SPREAD_PERCENT:", conf.BTCUSD_SPREAD_PERCENT)
    else:
        output("ADJUST_PIPS:", conf.ADJUST_PIPS)
    output("RESTRICT_FLG:", conf.RESTRICT_FLG)
    if conf.RESTRICT_FLG:
        output("RESTRICT_SEC:", conf.RESTRICT_SEC)
    output("EXCEPT_LIST_HOUR_TEST:", conf.EXCEPT_LIST_HOUR_TEST)

    output("important_index_list:", important_index_list)
    if len(important_index_list) != 0:
        output("important_index_range:", important_index_range)
    importantAnswer = ImportantIndex(index_set = important_index_list, range=important_index_range)

    for file in FILE_PREFIXS:
        if show_plot:
            # png保存用のディレクトリ作成
            save_dir = png_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
            makedirs(save_dir)
            output("PNG SAVE DIR:", save_dir)

        output("FILE_PREFIX:", file)

        for suffix in model_suffix:
            output("")
            if conf.LEARNING_TYPE == "REGRESSION":
                # suffixがリストなら２つめの値をborder_listとする
                if isinstance(suffix, list):
                    suffix, border_list = suffix
                    show_total = False
                output("suffix:", suffix)

                result_per_suffix_border[suffix] = {}

                bst = lgb.Booster(model_file=conf.MODEL_DIR + file)

                # 予想取得
                predict_list = bst.predict(x, num_iteration=int(suffix))

                if conf.OUTPUT_TYPE == "d":
                    # 現実のレートに換算し、予想時のレートと結果時のレートの差を求める
                    tmp_list = []
                    for t_c, t_p in zip(pred_close_list, predict_list):
                        #print(t_c,t_p)
                        tmp_list.append((t_c * ((t_p / 10000) + 1)) - t_c)

                    predict_list = np.array(tmp_list)

                elif conf.OUTPUT_TYPE == "sub":
                    # 現実のレートに換算する
                    tmp_list = []
                    for t_p in predict_list:
                        tmp_list.append(t_p)
                    predict_list = np.array(tmp_list)
            else:
                if conf.LEARNING_TYPE != "CATEGORY_BIN_BOTH":
                    # suffixがリストなら２つめの値をborder_listとする
                    if isinstance(suffix, list):
                        suffix, border_list = suffix
                        show_total = False
                    output("suffix:", suffix)

                    result_per_suffix_border[suffix] = {}

                    bst = lgb.Booster(model_file=conf.MODEL_DIR + file)

                    #予想取得
                    predict_list = bst.predict(x, num_iteration=int(suffix))
                    #print(predict_list)
                else:
                    # suffixがリストなら２つめの値をborder_listとする
                    if isinstance(suffix, list):
                        tmp_suffix, tmp_border_list = suffix
                        show_total = False
                        suffix = str(tmp_suffix[0]) + "-" + str(tmp_suffix[1])  # upとdwのsuffixを合わせる

                        border_list = []
                        for t_b in tmp_border_list:
                            border_list.append(str(t_b[0]) + "-" + str(t_b[1]))  # upとdwのborderを合わせる

                    output("suffix:", suffix)

                    result_per_suffix_border[suffix] = {}

                    # CATEGORY_BIN_UP とDWで予想した結果を合わせる
                    bst_up = lgb.Booster(model_file=conf.MODEL_DIR + file[0])
                    bst_dw = lgb.Booster(model_file=conf.MODEL_DIR + file[1])

                    # ndarrayで返って来る
                    predict_list_up = bst_up.predict(x, num_iteration=int(tmp_suffix[0]))
                    predict_list_dw = bst_dw.predict(x, num_iteration=int(tmp_suffix[1]))


                    # SAMEの予想結果は0とする
                    predict_list_zero = np.zeros((len(predict_list_up), 1))

                    predict_list_up = predict_list_up.reshape(len(predict_list_up), -1)#一次元から二次元配列に反感
                    predict_list_dw = predict_list_dw.reshape(len(predict_list_dw), -1)#一次元から二次元配列に反感

                    # UP,SAME,DWの予想結果を合算する
                    predict_list = np.concatenate([predict_list_up, predict_list_zero, predict_list_dw], 1)
                    #predict_list = all[:, [0, 1, 2]]

            for border_ind, border in enumerate(border_list):

                # 予想結果表示用テキストを保持
                result_txt = []

                # 成績の詳細を表示
                Acc, total_num, profit, correct_num, correct_pips, wrong_num, wrong_pips, earned_money_max_drawdown, max_drawdown, \
                bet_cnt_up, earned_money_up, correct_num_up, correct_pips_up, wrong_num_up, wrong_pips_up, \
                bet_cnt_dw, earned_money_dw, correct_num_dw, correct_pips_dw, wrong_num_dw, wrong_pips_dw, \
                answer_rate_list, answer_rate_list_up, answer_rate_list_dw, idx, idx_up, idx_dw, close_correct_num, spr_pred_pips_list = showDetail(
                    save_dir, suffix, conf, predict_list, sub_close_list, border, answer_rate_up_list,
                    answer_rate_dw_list,
                    position_list, target_score_list, close_list, score_list, answer_score_up_list,
                    answer_score_dw_list, target_spread_list, target_jpy_list, pred_close_list, importantAnswer, show_plot=show_plot, lgbm_flg=True, start_dt=start_dt, end_dt=end_dt)

                result_per_suffix_border[suffix][border] = get_result_rgr_both(Acc, total_num, profit, correct_num,
                                                                               correct_pips, wrong_num, wrong_pips,
                                                                               bet_cnt_up, earned_money_up,
                                                                               correct_num_up,
                                                                               bet_cnt_dw, earned_money_dw,
                                                                               correct_num_dw, close_correct_num)
                result_per_suffix_border[suffix][border][
                    "EanredMoney/(MAXDrawDown-FUND)"] = earned_money_max_drawdown
                result_per_suffix_border[suffix][border]["MAXDrawDown"] = max_drawdown

                profit_pips = correct_pips + wrong_pips

                # 全体の予想結果を表示 ※UP or DOWNのみ SAMEの予想結果は無視
                result_txt.append("Accuracy over " + str(border) + ":" + str(Acc))
                result_txt.append(
                    "Bet_cnt:" + str(total_num) + " Correct:" + str(correct_num) + " Wrong:" + str(wrong_num))
                if total_num != 0:
                    result_txt.append("Profit:" + str(profit) + " Pips:" + str(profit_pips) + " /Total_num:" + str(
                        profit_pips / total_num))
                    result_txt.append(
                        "EanredMoney/(MAXDrawDown-FUND):" + str(earned_money_max_drawdown) + " MAXDrawDown:" + str(
                            max_drawdown))
                    result_txt.append("C-W:" + str(correct_num - wrong_num))
                    result_txt.append("C-C-W:" + str(close_correct_num - (total_num - close_correct_num)))
                    result_txt.append("C-C-W-R:" + str(close_correct_num / total_num))

                    result_per_suffix_border[suffix][border]["Pips"] = profit_pips
                    result_per_suffix_border[suffix][border]["Pips/Total_num"] = profit_pips / total_num

                if correct_num != 0:
                    result_txt.append(
                        "Correct_Pips:" + str(correct_pips) + " /Correct_num:" + str(correct_pips / correct_num))
                if wrong_num != 0:
                    result_txt.append(
                        "Wrong_Pips:" + str(wrong_pips) + " /Wrong_num:" + str(wrong_pips / wrong_num))

                if conf.LEARNING_TYPE in ["REGRESSION", "CATEGORY", "CATEGORY_BIN_BOTH", "CATEGORY_BIN_UP",
                                          "CATEGORY_BIN_DW", "CATEGORY_BIN", "REGRESSION_OCOPS", "CATEGORY_OCOPS"]:
                    if bet_cnt_up != 0:
                        result_txt.append("Profit_UP:" + str(earned_money_up) + " Acc_UP:" + str(
                            result_per_suffix_border[suffix][border]["Acc_UP"]) + " Bet_cnt_UP:" + str(bet_cnt_up))
                    if bet_cnt_dw != 0:
                        result_txt.append("Profit_DW:" + str(earned_money_dw) + " Acc_DW:" + str(
                            result_per_suffix_border[suffix][border]["Acc_DW"]) + " Bet_cnt_DW:" + str(bet_cnt_dw))

                output("")
                for i in result_txt:
                    output(i)

                if show_profit_ind:
                    showProfitInd(border, idx, answer_rate_list, target_ind_list, show_plot, save_dir, target_ind_cols)

                if show_profit_ind_up_dw:
                    showProfitIndUpDown(border, idx_up, idx_dw, answer_rate_list_up, answer_rate_list_dw, target_ind_list, show_plot, save_dir, target_ind_cols)

                if show_profit_time:
                    showProfitTime(conf, idx, answer_rate_list, target_score_list)

                if len(spr_pred_pips_list) != 0:
                    if show_profit_per_spread:
                        showPipsPerSpread(spr_pred_pips_list, border)

        if show_total == True and conf.LEARNING_TYPE in ["REGRESSION", "CATEGORY", "CATEGORY_BIN_BOTH",
                                                         "CATEGORY_BIN_UP", "CATEGORY_BIN_DW", "CATEGORY_BIN",
                                                         "REGRESSION_OCOPS", "CATEGORY_OCOPS"]:
            output("利益が多い順")
            total_result = {}
            for b in border_list:
                output("border:", b)
                sorted_d = sorted(result_per_suffix_border.items(), key=lambda x: x[1][b]["Profit"], reverse=True)
                cnt_t = 0
                for k, v in sorted_d:
                    if cnt_t > 20:
                        break

                    total_profit = v[b]["Profit"]
                    total_len = v[b]["Bet_cnt"]

                    total_result[str(k) + "-" + str(b)] = v[b]
                    if total_len != 0:
                        output("suffix:", k, "EanredMoney/(MAXDrawDown-FUND):",
                               v[b]["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                               "MAXDrawDown:", v[b]["MAXDrawDown"], "Acc:", v[b]["Acc"], "Bet_cnt:", total_len,
                               "C-W:", v[b]["Correct-Wrong"], "C-C-W:", v[b]["Close-Correct-Wrong"],
                               "C-C-W-R:", v[b]["Close-Correct-Wrong-Rate"],
                               "Pips:", v[b]["Pips"], "Pips/Total_num:", v[b]["Pips/Total_num"],
                               )
                    cnt_t += 1
                output("")
                output("")

            output("EanredMoney/(MAXDrawDown-FUND)が多い順")
            for b in border_list:
                output("border:", b)
                sorted_d = sorted(result_per_suffix_border.items(),
                                  key=lambda x: x[1][b]["EanredMoney/(MAXDrawDown-FUND)"], reverse=False)
                cnt_t = 0
                for k, v in sorted_d:
                    if cnt_t > 20:
                        break
                    total_profit = v[b]["Profit"]
                    total_len = v[b]["Bet_cnt"]
                    total_result[str(k) + "-" + str(b)] = v[b]
                    if total_len != 0:
                        output("suffix:", k, "EanredMoney/(MAXDrawDown-FUND):",
                               v[b]["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                               "MAXDrawDown:", v[b]["MAXDrawDown"], "Acc:", v[b]["Acc"], "Bet_cnt:", total_len,
                               "C-W:", v[b]["Correct-Wrong"], "C-C-W:", v[b]["Close-Correct-Wrong"],
                               "C-C-W-R:", v[b]["Close-Correct-Wrong-Rate"],
                               "Pips:", v[b]["Pips"], "Pips/Total_num:", v[b]["Pips/Total_num"],
                               )
                    cnt_t += 1
                output("")

            output("正解が多い順(スプレッドあり)")
            for b in border_list:
                output("border:", b)
                sorted_d = sorted(result_per_suffix_border.items(), key=lambda x: x[1][b]["Correct-Wrong"],
                                  reverse=True)
                cnt_t = 0
                for k, v in sorted_d:
                    if cnt_t > 20:
                        break
                    total_profit = v[b]["Profit"]
                    total_len = v[b]["Bet_cnt"]
                    total_result[str(k) + "-" + str(b)] = v[b]
                    if total_len != 0:
                        output("suffix:", k, "EanredMoney/(MAXDrawDown-FUND):",
                               v[b]["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                               "MAXDrawDown:", v[b]["MAXDrawDown"], "Acc:", v[b]["Acc"], "Bet_cnt:", total_len,
                               "C-W:", v[b]["Correct-Wrong"], "C-C-W:", v[b]["Close-Correct-Wrong"],
                               "C-C-W-R:", v[b]["Close-Correct-Wrong-Rate"],
                               "Pips:", v[b]["Pips"], "Pips/Total_num:", v[b]["Pips/Total_num"],
                               )
                    cnt_t += 1
                output("")

            output("正解が多い順(スプレッドなし)")
            for b in border_list:
                output("border:", b)
                sorted_d = sorted(result_per_suffix_border.items(), key=lambda x: x[1][b]["Close-Correct-Wrong"],
                                  reverse=True)
                cnt_t = 0
                for k, v in sorted_d:
                    if cnt_t > 20:
                        break
                    total_profit = v[b]["Profit"]
                    total_len = v[b]["Bet_cnt"]
                    total_result[str(k) + "-" + str(b)] = v[b]
                    if total_len != 0:
                        output("suffix:", k, "EanredMoney/(MAXDrawDown-FUND):",
                               v[b]["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                               "MAXDrawDown:", v[b]["MAXDrawDown"], "Acc:", v[b]["Acc"], "Bet_cnt:", total_len,
                               "C-W:", v[b]["Correct-Wrong"], "C-C-W:", v[b]["Close-Correct-Wrong"],
                               "C-C-W-R:", v[b]["Close-Correct-Wrong-Rate"],
                               "Pips:", v[b]["Pips"], "Pips/Total_num:", v[b]["Pips/Total_num"],
                               )
                    cnt_t += 1
                output("")

            output("全体の利益の多い順")
            sorted_d_total = sorted(total_result.items(), key=lambda x: x[1]["Profit"], reverse=True)
            cnt_t = 0
            for k, v in sorted_d_total:
                if cnt_t > 20:
                    break
                total_profit = v["Profit"]
                total_len = v["Bet_cnt"]
                if total_len != 0:
                    output("suffix-border:", k, "Profit:", total_profit, "EanredMoney/(MAXDrawDown-FUND):",
                           v["EanredMoney/(MAXDrawDown-FUND)"],
                           "MAXDrawDown:", v["MAXDrawDown"], "Acc:", v["Acc"], "Bet_cnt:", total_len, "C-W:",
                           v["Correct-Wrong"], "C-C-W:", v["Close-Correct-Wrong"],
                           "C-C-W-R:", v["Close-Correct-Wrong-Rate"],
                           "Pips:", v["Pips"], "Pips/Total_num:", v["Pips/Total_num"],
                           )

                cnt_t += 1
            output("")

            output("全体のEanredMoney/(MAXDrawDown-FUND)の多い順")
            sorted_d_total = sorted(total_result.items(), key=lambda x: x[1]["EanredMoney/(MAXDrawDown-FUND)"],
                                    reverse=False)
            cnt_t = 0
            for k, v in sorted_d_total:
                if cnt_t > 20:
                    break
                total_profit = v["Profit"]
                total_len = v["Bet_cnt"]
                if total_len != 0:
                    output("suffix-border:", k, "EanredMoney/(MAXDrawDown-FUND):",
                           v["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                           "MAXDrawDown:", v["MAXDrawDown"], "Acc:", v["Acc"], "Bet_cnt:", total_len, "C-W:",
                           v["Correct-Wrong"], "C-C-W:", v["Close-Correct-Wrong"],
                           "C-C-W-R:", v["Close-Correct-Wrong-Rate"],
                           "Pips:", v["Pips"], "Pips/Total_num:", v["Pips/Total_num"],
                           )

                cnt_t += 1
            output("")

            output("全体の正解の多い順(スプレッドあり)")
            sorted_d_total = sorted(total_result.items(), key=lambda x: x[1]["Correct-Wrong"], reverse=True)
            cnt_t = 0
            for k, v in sorted_d_total:
                if cnt_t > 20:
                    break
                total_profit = v["Profit"]
                total_len = v["Bet_cnt"]
                if total_len != 0:
                    output("suffix-border:", k, "EanredMoney/(MAXDrawDown-FUND):",
                           v["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                           "MAXDrawDown:", v["MAXDrawDown"], "Acc:", v["Acc"], "Bet_cnt:", total_len, "C-W:",
                           v["Correct-Wrong"], "C-C-W:", v["Close-Correct-Wrong"],
                           "C-C-W-R:", v["Close-Correct-Wrong-Rate"],
                           "Pips:", v["Pips"], "Pips/Total_num:", v["Pips/Total_num"],
                           )

                cnt_t += 1
            output("")

            output("全体の正解の多い順(スプレッドなし)")
            sorted_d_total = sorted(total_result.items(), key=lambda x: x[1]["Close-Correct-Wrong"], reverse=True)
            cnt_t = 0
            for k, v in sorted_d_total:
                if cnt_t > 20:
                    break
                total_profit = v["Profit"]
                total_len = v["Bet_cnt"]
                if total_len != 0:
                    output("suffix-border:", k, "EanredMoney/(MAXDrawDown-FUND):",
                           v["EanredMoney/(MAXDrawDown-FUND)"], "Profit:", total_profit,
                           "MAXDrawDown:", v["MAXDrawDown"], "Acc:", v["Acc"], "Bet_cnt:", total_len, "C-W:",
                           v["Correct-Wrong"], "C-C-W:", v["Close-Correct-Wrong"],
                           "C-C-W-R:", v["Close-Correct-Wrong-Rate"],
                           "Pips:", v["Pips"], "Pips/Total_num:", v["Pips/Total_num"],
                           )

                cnt_t += 1
            output("")

    output("predict end", time.perf_counter() - start_time)

if __name__ == "__main__":

    start_time = time.perf_counter()

    conf = conf_class_lgbm.ConfClassLgbm()
    test_data_load_path = "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF17.pickle"
    conf_load_path = "/db2/lgbm/" + conf.SYMBOL + "/test_file/TESF17-conf.pickle"

    start_dt = datetime(2023, 4, 1, )
    end_dt = datetime(2024, 5, 4, ) #この時間を含めない
    output("test_data_load_path:", test_data_load_path)
    output("conf_load_path:", conf_load_path)

    output("start end:", start_dt, end_dt)

    # テストデータロード
    with open(test_data_load_path, 'rb') as f:
        test_lmd = pickle.load(f)

    # テストデータ作成時のconfをロード
    with open(conf_load_path, 'rb') as cf:
        test_conf = pickle.load(cf)

    do_test(conf, test_conf, test_lmd, start_dt, end_dt )

    print("Processing Time(Sec)", time.perf_counter() - start_time)
    # 終わったらメールで知らせる
    mail.send_message(host, ": testLstmFX2_answer finished!!!")