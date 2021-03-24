import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
import redis
import json
import pandas as pd
import os
from scipy.ndimage.interpolation import shift
from datetime import datetime
import time
from decimal import Decimal
import readConf
import psutil
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle

readConf.logging.config.fileConfig( os.path.join(readConf.current_dir,"config","logging.conf"))
logger = readConf.logging.getLogger("app")
myLogger = readConf.printLog(logger)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
symbol = "GBPJPY"
db_no = 3
train = False

#過去データ数
maxlen = 350
pred_term = 15
s = "2"
#特徴量の種類数(1ならcloseのみ)
in_num = 1
#startからendへ戻ってrec_num分のデータを学習用とする
rec = 20000000
rec_num = rec + maxlen + pred_term + 1
estimators =5000

start = datetime(2018, 1, 1)
end = datetime(2018, 12, 25)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))


except_highlow = True
except_list = [20, 21, 22]

payout = 1000
payoff = 1000

np.random.seed(0)

spread = 1

file_prefix = symbol + "_lgbm_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_rec_" + str(rec) + "_est_" + str(estimators)

model_file_path = os.path.join(readConf.model_dir, file_prefix + ".pcl")
myLogger("Model is " , model_file_path)

def get_redis_data(symbol, rec_num, maxlen, pred_term, db_no):
    print("Train:", str(train))
    r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)

    if train:
        result = r.zrevrangebyscore(symbol, start_score, end_score, start=0, num=rec_num + 1,withscores=True)
        result.reverse()
    else:
        result = r.zrangebyscore(symbol, start_score, end_score,withscores=True)
    print("got redis data!!! stop redis to save memory");


    close_tmp, time_tmp = [], []
    score_tmp = []
    for line in result:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)
        # askとbid(close)の仲値にする。ハイローに合わせる
        #mid = float((Decimal(str(tmps.get("close"))) + Decimal(str(tmps.get("ask")))) / Decimal("2"))
        close_tmp.append(tmps.get("close"))
        time_tmp.append(tmps.get("time"))
        #score_tmp.append(score)

    #一つ後の終値との変化率を特徴量とする
    tmp_array = []
    for i,v in enumerate(close_tmp):
        if i == 0:
            continue
        divide = close_tmp[i]/close_tmp[i-1]
        if close_tmp[i] == close_tmp[i-1]:
            divide = 1
        tmp_array.append(divide)
    #close = np.array(10000 * np.log(close_tmp / shift(close_tmp, 1, cval=np.NaN))[1:])
    close = np.array(10000 * np.log(tmp_array))
    #メモリ解放
    del result
    gc.collect()
    mem = psutil.virtual_memory()
    print("メモリ残量")
    print(mem.available)
    #print(type(close))
    print("学習始め")
    print(time_tmp[0:5])
    print("学習終わり")
    print(time_tmp[-5:])

    close_data, label_data = [], []

    up = 0
    same = 0
    data_length = len(close) - maxlen - pred_term - 1
    print("data_length: " + str(data_length))

    #学習対象データのインデックスを保持
    target_list = []

    for i in range(data_length):
        # ハイローオーストラリアの取引時間外を学習対象からはずす
        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen - 1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue;

        # maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
        tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen - 1], '%Y-%m-%d %H:%M:%S')
        delta = tmp_time_aft - tmp_time_bef

        if delta.total_seconds() > ((maxlen - 1) * int(s)):
            continue;

        target_list.append(i);
        if i % 1000000 == 0:
            print("処理件数1:" + str(i))


    retX = np.zeros((len(target_list), maxlen))
    retY = np.zeros(len(target_list))
    time_data = np.array([ "yyyy-mm-dd 00:00:00" for i in range(len(target_list)) ])
    #score_data = np.zeros(len(target_list))
    #close_data = np.zeros((len(target_list), 10))
    for i, idx in enumerate(target_list):
        retX[i] = close[idx:(idx + maxlen)]
        #score_data[i] = score_tmp[1 + idx + maxlen - 1]
        bef = close_tmp[1 + idx + maxlen - 1]
        aft = close_tmp[1 + idx + maxlen + pred_term - 1]
        time_data[i] = time_tmp[1 + idx + maxlen - 1]
        #close_data[i] = close_tmp[idx-1:(idx + 9)]
        #正解をいれる
        if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
            # 上がった場合
            retY[i] = 0
            up = up + 1
        elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal(str("0.00001")) * Decimal(str(spread))):
            retY[i] = 2
        else:
            retY[i] = 1
            same = same + 1

        if i % 1000000 == 0:
            print("処理件数2:" + str(i))

    time_tmp_np = np.array(time_tmp[1:])
    close_tmp_np = np.array(close_tmp[1:])
    del time_tmp
    gc.collect()
    mem = psutil.virtual_memory()
    print("メモリ残量")
    print(mem.available)
    """
    for i in score_data[:10]:
        print(str(i))
    print(close_data[:10])
    """
    #print("TYPE:", type(retX))
    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ", up / len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))

    return retX, retY, time_data,time_tmp_np,close, close_tmp_np

'''
モデル学習
'''

def do_train():
    # 処理時間計測
    t1 = time.time()
    X_train, Y_train, time_data, time_tmp_np, close, close_tmp_np = get_redis_data(symbol, rec_num, maxlen, pred_term, db_no)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(  X, Y, test_size=0.2, random_state=42)

    #lgb_train = lgbm.Dataset(X_train, Y_train)
    #lgb_eval = lgbm.Dataset(X_valid, Y_valid)

    # LightGBM parameters

    evaluation_results = {} #評価結果格納用
    clf = LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass', #分類
        min_data_in_leaf=1,
        num_leaves=31, #木に存在する分岐の個数
        device="gpu",
        n_estimators=estimators,#木の数
        min_child_weight=1,
        learning_rate=0.1,
        nthread=2,
        gpu_platform_id=1,gpu_device_id=0,#1080 Ti
        n_jobs=4)

    clf.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train),],
        eval_names=['Train',],
        early_stopping_rounds=10,
        eval_metric="multi_logloss",
        callbacks=[lgbm.record_evaluation(evaluation_results),],
        verbose=10,
    )
    print('Saving model...')
    # save model to file
    with open(model_file_path, mode='wb') as fp:
        pickle.dump(clf, fp)

    #Y_pred = clf.predict_proba(X_valid)
    #print(Y_pred[:10])
    #print(Y_valid[:10])
    # 返り値は確率になっているので最尤に寄せる
    #Y_pred_max = np.argmax(Y_pred, axis=1)

    #myLogger("best_iteration: " + str(clf.best_iteration))

    #パフォーマンスを可視化
    #参考:http://knknkn.hatenablog.com/entry/2019/02/02/164829
    #Accuracy score: 正解率。1のものは1として分類(予測)し、0のものは0として分類した割合
    #Precision score: 精度。1に分類したものが実際に1だった割合
    #Recall score: 検出率。1のものを1として分類(予測)した割合
    #F1 score: PrecisionとRecallを複合した0~1のスコア。数字が大きいほど良い評価。
    #myLogger('Accuracy score = \t {}'.format(accuracy_score(Y_valid, Y_pred_max)))

    t2 = time.time()
    elapsed_time = t2 - t1
    myLogger("経過時間：" + str(elapsed_time))
    mem = psutil.virtual_memory()
    print("メモリ残量")
    print(mem.available)
    # Plot the log loss during training

    fig, axs = plt.subplots(1, 2, figsize=[15, 4])
    axs[0].plot(evaluation_results['Train']['multi_logloss'], label='Train')
    #axs[0].plot(evaluation_results['Valid']['multi_logloss'], label='Valid')
    axs[0].set_ylabel('Log loss')
    axs[0].set_xlabel('Boosting round')
    axs[0].set_title('Training performance')
    axs[0].legend()

    # Plot feature importance
    importances = pd.DataFrame({'features': clf.n_features_,
                                'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    axs[1].bar(x=np.arange(len(importances)), height=importances['importance'])
    axs[1].set_xticks(np.arange(len(importances)))
    axs[1].set_xticklabels(importances['features'])
    axs[1].set_ylabel('Feature importance (# times used to split)')
    axs[1].set_title('Feature importance')
    plt.show()


def getAcc(pred, border, correct):
    up = pred[:, 0]
    down = pred[:, 2]
    up_ind5 = np.where(up >= border)[0]
    down_ind5 = np.where(down >= border)[0]
    pred_up = pred[up_ind5,:]
    correct_up= correct[up_ind5]
    pred_down = pred[down_ind5,:]
    correct_down= correct[down_ind5]

    up_eq = np.equal(pred_up.argmax(axis=1), correct_up)
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    down_eq = np.equal(pred_down.argmax(axis=1), correct_down)
    down_cor_length = int(len(np.where(down_eq == True)[0]))

    if (len(up_ind5) + len(down_ind5)) ==0:
        Acc =0
    else:
        Acc = (up_cor_length + down_cor_length) / (len(up_ind5) + len(down_ind5))
    total = len(up_ind5) + len(down_ind5)
    correct = int(total * Acc)

    return Acc, total, correct

def countDrawdoan(max_drawdowns, max_drawdown, drawdown, money):
    drawdown = drawdown + money
    if max_drawdown > drawdown:
        #最大ドローダウンを更新してしまった場合
        max_drawdown = drawdown

    if drawdown > 0:
        if max_drawdown != 0:
            max_drawdowns.append(max_drawdown)
        drawdown = 0
        max_drawdown = 0

    return max_drawdown, drawdown

class TimeRate():
    def __init__(self):
        self.all_cnt = 0
        self.correct_cnt = 0

def do_predict():
    X, Y, time_data, time_tmp_np, close, close_tmp_np = get_redis_data(symbol, rec_num, maxlen, pred_term, db_no)
    # 処理時間計測
    t1 = time.time()
    border_list=[0.51,0.52,0.53,0.54,0.55,0.56]
    border_list_show=[0.51,0.52,0.53]
    result_txt = []

    # モデルロード
    with open(model_file_path, mode='rb') as fp:
        clf = pickle.load(fp)
    Y_pred = clf.predict_proba(X)

    # 返り値は確率になっているので最尤に寄せる
    Y_pred_max = np.argmax(Y_pred, axis=1)
    #myLogger(X[:10])
    myLogger(Y_pred[:10])
    myLogger(Y_pred_max[:10])
    myLogger(Y[:10])


    print('Accuracy score = \t {}'.format(accuracy_score(Y, Y_pred_max)))

    for b in border_list:
        spread_trade = {}
        spread_win = {}
        max_drawdown = 0
        drawdown = 0
        max_drawdowns = []

        border = b
        Acc5 = getAcc(Y_pred,border,Y)

        result_txt.append("Accuracy over " + str(border) + ":" + str(Acc5[0]))
        result_txt.append("Total:" + str(Acc5[1]) + " Correct:" + str(Acc5[2]))
        win_money = (payout * Acc5[2]) - ((Acc5[1] - Acc5[2]) * payoff)
        result_txt.append("Money:" + str(win_money))
        if border not in border_list_show:
            continue

        perTimeRes = {}
        for j in range(24):
            perTimeRes[str(j)] = {"count": 0, "win_count": 0}

        ind5 = np.where(Y_pred >=border)[0]
        x5 = Y_pred[ind5,:]
        y5= Y[ind5]
        #z5= dataZ[ind5]
        #p5 = price_data[ind5]
        t5 = time_data[ind5]
        #ep5 = end_price_data[ind5]
        #sp5 = spread_data[ind5]
        #ca5 = close_abs_data[ind5]


        #cor_list_abs, wrong_list_abs = np.ones(up_cor_length + down_cor_length, dtype=np.float64), np.ones(up_wrong_length + down_wrong_length, dtype=np.float64)

        money_x, money_y = np.array([ "00:00:00" for i in range(len(time_tmp_np)) ]), np.ones(len(time_tmp_np), dtype=np.float64)
        money_tmp = {}

        money = 1000000
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

        cnt_up_cor = 0
        cnt_up_wrong = 0
        cnt_down_cor = 0
        cnt_down_wrong = 0
        loop_cnt = 0
        cnt_cor_abs = 0
        cnt_wrong_abs = 0

        time_rate_list = {}
        for i in range(0, 24):
            time_rate_list[i] = (TimeRate())

        ind_tmp = -1
        for x,y,t in zip(x5,y5,t5):
            hourT = str(datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour)
            ind_tmp = ind_tmp +1

            max = x.argmax()
            trade_flg = False
            win_flg = False
            # Up predict
            if max == 0:
                perTimeRes[hourT]["count"] += 1

                if max == y:
                    money = money + payout
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payout)
                    cnt_up_cor = cnt_up_cor + 1
                    win_flg = True
                    perTimeRes[hourT]["win_count"] += 1
                else :
                    money = money - payoff
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payoff * -1)
                    cnt_up_wrong = cnt_up_wrong + 1

            elif max == 2:
                perTimeRes[hourT]["count"] += 1

                if max == y:
                    money = money + payout
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payout)
                    cnt_down_cor = cnt_down_cor + 1
                    win_flg = True
                    perTimeRes[hourT]["win_count"] += 1
                else:
                    money = money - payoff
                    max_drawdown, drawdown = countDrawdoan(max_drawdowns, max_drawdown, drawdown, payoff * -1)
                    cnt_down_wrong = cnt_down_wrong + 1

            money_tmp[t] = money
            loop_cnt = loop_cnt + 1

        prev_money = 1000000
        #T = time[0]
        #myLogger("T:" + T[11:])
        for i, ti in enumerate(time_tmp_np):
            if ti in money_tmp.keys():
                prev_money = money_tmp[ti]

            money_x[i] = ti[11:13]
            money_y[i] = prev_money

        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
        fig = plt.figure()
        #価格の遷移
        ax1 = fig.add_subplot(111)
        #ax1.plot(time,close)
        ax1.plot(close_tmp_np, 'g')
        ax2 = ax1.twinx()
        ax2.plot(money_y)

        plt.title('border:' + str(border) + " payout:" + str(payout) + " spread:" + str(spread) + " money:" + str(money))
        plt.show()

        for k in range(24):
            cnt = perTimeRes[str(k)]["count"]
            winCnt = perTimeRes[str(k)]["win_count"]
            if cnt != 0:
                myLogger(str(k) + ": win rate " + str(winCnt / cnt) + " ,count " + str(cnt) + " ,win count " + str(winCnt))

        max_drawdowns.sort()
        myLogger(max_drawdowns[0:10])

        drawdown_cnt = {}
        for i in max_drawdowns:
            for k, v in readConf.drawdown_list.items():
                if i < v[0] and i >= v[1]:
                    drawdown_cnt[k] = drawdown_cnt.get(k,0) + 1
                    break
        for k, v in sorted(readConf.drawdown_list.items()):
            myLogger(k, drawdown_cnt.get(k,0))

        max_drawdowns_np = np.array(max_drawdowns)
        df = pd.DataFrame(pd.Series(max_drawdowns_np.ravel()).describe()).transpose()
        myLogger(df)

    for i in result_txt:
        myLogger(i)

    t2 = time.time()
    elapsed_time = t2 - t1
    myLogger("経過時間：" + str(elapsed_time))

if __name__ == '__main__':
    if train:
        do_train()
    else:
        # モデルロードをjava化
        """
        with open(model_file_path, mode='rb') as fp:
            clf = pickle.load(fp)
        code = m2c.export_to_java(clf)
        with open("/app/bin_op/model/lgbm/" + file_prefix + ".java", mode='w') as fp:
            fp.write(code)
        print("convert to java finish!!")
        """
        do_predict()
