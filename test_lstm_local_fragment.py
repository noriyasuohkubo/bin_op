import sys
import send_mail as mail

from util_predict import *

"""
close間隔が1秒未満に対応済み
"""

class Conf():
    def __init__(self):
        self.PAIR = "USDJPY"

        self.AI_MODEL_TERM = 1 #AIモデルの最小データ間隔秒(closeデータの間隔)
        self.LOOP_TERM = self.AI_MODEL_TERM #mainのループ間隔

        self.MODEL_PREDICT_TERM = 30
        self.MODEL_NO = "2057-184"
        self.MODEL_NAME = "MN" + self.MODEL_NO
        self.MODEL_LEARN_TYPE = "lstm"
        self.MODEL_CODE = "predict" + str(self.MODEL_PREDICT_TERM) + "_" + self.MODEL_NO + "_" + self.MODEL_LEARN_TYPE

        self.MODEL_TYPE = "CATEGORY"

        self.FRAGMENT_NUM = 6 # 参照するレート情報のインデックス.win2のDB1に登録してあるFRAGMENTSから取得する.None:設定なし
        self.FRAGMENT_INDEX_LIST = [-10738,-10226,-9714,-9202,-8690,-8178,-7666,-7410,-7154,-6898,-6642,-6386,-6130,-5874,
                                    -5618,-5362,-5106,-4850,-4594,-4338,-4082,-3826,-3698,-3570,-3442,-3314,-3186,-3058,-2930,
                                    -2802,-2674,-2546,-2418,-2290,-2162,-2034,-1906,-1842,-1778,-1714,-1650,-1586,-1522,-1458,-1394,
                                    -1330,-1266,-1202,-1138,-1074,-1010,-946,-914,-882,-850,-818,-786,-754,-722,-690,-658,-626,-594,
                                    -562,-530,-498,-466,-450,-434,-418,-402,-386,-370,-354,-338,-322,-306,-290,-274,-258,-242,-226,-218,
                                    -210,-202,-194,-186,-178,-170,-162,-154,-146,-138,-130,-122,-114,-106,-102,-98,-94,-90,-86,-82,-78,-74,
                                    -70,-66,-62,-58,-54,-50,-46,-44,-42,-40,-38,-36,-34,-32,-30,-28,-26,-24,-22,-20,-18,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1
                                    ]

        self.MAX_CLOSE_LEN = self.FRAGMENT_INDEX_LIST[0] * -1 #渡されるcloseの長さ
        self.MAX_LEN = self.MAX_CLOSE_LEN - 1
        self.MAX_LEN_SEC = self.MAX_LEN * self.AI_MODEL_TERM

        self.FRAGMENTS_INPUT_TYPE = "div-10000"  # 特徴量のタイプ div or sub とmultiの値をハイフンでつなげる

        self.MODEL_DIR_PATH = "/app/model/bin_op/"

        current_dir = os.path.dirname(__file__)
        logging.config.fileConfig(os.path.join(current_dir, "config", "logging_" + self.MODEL_CODE) + ".conf")
        loggerConf = logging.getLogger(self.MODEL_CODE)

        self.LOGGER = printLog(loggerConf)
        self.SERVER_NAME = "win3_" + self.MODEL_CODE

        self.HOST = 'localhost'
        self.DB_NO = 8
        self.DB_KEY = self.PAIR + "_" + self.MODEL_CODE

        self.FX_DATA_MACHINE = "192.168.1.114"  # win2
        self.FX_DB_NO = 0
        self.DB_FX_DATA_KEY = "VANTAGE_USDJPY.p_S1"
        self.FX_DATA_TERM = 1
        self.DIV_SEC = 300  # 指定秒数前と現在レートのdivを求める為に設定
        self.SUB_SEC = 300  # 指定秒数前と現在レートのsubを求める為に設定


        self.RATE_FORMAT = "{:.3f}"

        self.END_DATETIME = None

        self.DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S'

        self.SPREAD = None

        tmp_dt = datetime.datetime.now()
        if tmp_dt.hour in [22,23]:
            tmp_dt = tmp_dt + timedelta(days=1)

        self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                              hour=20, minute=59, second=0, microsecond=0)
        if tmp_dt.weekday() == 4:
            #金曜日のみ取引所が閉まるので早めに終了
            self.END_DATETIME = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                                  hour=19, minute=59, second=0, microsecond=0)


def get_remote_close(conf, base_t_just):
    cnt = 0
    return_close = []
    while True:
        cnt += 1
        if cnt > 50:
            break

        result = redis_fx_db.zrangebyscore(conf.DB_FX_DATA_KEY, get_decimal_sub(base_t_just, get_decimal_sub(conf.LOOP_TERM, conf.AI_MODEL_TERM)),
                                           base_t_just, withscores=True)
        # conf.LOGGER(result)
        tmp_multi = get_decimal_divide(1, conf.FX_DATA_TERM)
        if len(result) == int(get_decimal_multi(get_decimal_sub(conf.LOOP_TERM, conf.AI_MODEL_TERM), tmp_multi) + 1):

            for i in range(0, len(result), int(get_decimal_multi(conf.AI_MODEL_TERM, tmp_multi))):
                line = result[i]
                body = line[0]

                tmps = json.loads(body)
                ask = tmps["ask"]
                bid = tmps["bid"]

                return_close.append(float(get_decimal_divide(get_decimal_add(ask, bid), "2")))
                conf.SPREAD = int(get_decimal_multi(get_decimal_sub(ask, bid), 1000))

            # conf.LOGGER(return_close)
            return return_close
        time.sleep(0.01)
    return return_close

def registRedis(conf, redis_db, score, child, key):
    # 既存レコードがなければ追加
    tmp_val = redis_db.zrangebyscore(key, score, score)
    if len(tmp_val) == 0:
        redis_db.zadd(key, json.dumps(child), score)
    #redis_db.zadd(key, json.dumps(child), score)

def main_loop(conf,):
    return_code = 1
    err_flg = 0

    try:
        tmp_dt = datetime.datetime.now()
        base_dt = datetime.datetime(year=tmp_dt.year, month=tmp_dt.month, day=tmp_dt.day,
                                    hour=tmp_dt.hour, minute=tmp_dt.minute, second=tmp_dt.second, microsecond=0)

        #開始時刻設定 数秒待つ
        base_dt = base_dt + timedelta(seconds=(conf.LOOP_TERM * 2 + (conf.LOOP_TERM - tmp_dt.second % conf.LOOP_TERM)))
        base_t = get_decimal_add(time.mktime(base_dt.timetuple()), 0.01)
        conf.LOGGER("base_t", base_t)

        while True:
            time.sleep(0.0001)
            # print(datetime.now().microsecond)
            if (base_t - time.time()) < 0.0005:  # time.timeの誤差を考慮して0.5ミリ秒早く起きる
                break
            # もし追い越してしまったらエラーとする
            if (base_t - time.time()) < -0.01:
                conf.LOGGER("TIME START FAILED!!", base_t, time.time())
                err_flg = True

        if err_flg:
            # ここまででエラーあったら終了
            mail.send_message(subject=conf.SERVER_NAME, msg="SYSTEM ERROR OCCURED! EXIT!!")
            return 3

        first_loop = True
        closes = []
        closes_org = []

        close_take = 0.0
        predict_take = 0.0
        order_take = 0.0
        deal_take = 0.0
        db_take = 0.0
        loop_take = 0.0

        sleep_time = 0.0

        time_over_flg = False

        while (True):
            base_t_just = get_decimal_sub(base_t, 0.01)  # base_tは0.01秒遅くなっているため

            start = time.perf_counter()

            offset = base_t - time.time()  # 起動すべき時間と起動した時間の差
            tmp_offset = offset
            if tmp_offset < 0:
                tmp_offset = tmp_offset * -1
            # offsetが1000ミリ秒以上の場合メール送信 早くても遅くても駄目
            if tmp_offset > 1:
                conf.LOGGER("offset over 1000milces", offset, close_take, predict_take, order_take, deal_take, db_take,
                            loop_take, sleep_time)
                conf.TIMEOVER_CNT += 1
                err_flg = True
                time_over_flg = True
                break

            tdt = datetime.datetime.now()

            # 終了時間になったらポジションあれば決済し、抜ける
            if tdt >= conf.END_DATETIME:
                conf.LOGGER("main loop end!!")
                return_code = 2
                break

            start_close = time.perf_counter()

            #リモートPCのレート取得
            close = get_remote_close(conf, base_t_just)
            if len(close) == 0:
                conf.LOGGER("get remote close failed!!!")
                err_flg = True
                break

            close_take = time.perf_counter() - start_close

            # 過去分レート取得
            if first_loop:
                conf.LOGGER("first_loop get remote rate start")
                end_t = get_decimal_sub(base_t_just, conf.LOOP_TERM)
                start_t = get_decimal_sub(end_t, get_decimal_sub(conf.MAX_LEN_SEC, conf.LOOP_TERM))
                result = redis_fx_db.zrangebyscore(conf.DB_FX_DATA_KEY, start_t, end_t, withscores=True)

                # print(db ,len(result))
                for i in range(0, len(result), conf.AI_MODEL_TERM):  # AI_MODEL_TERM秒おきのデータのみ必要なのでLOOP_TERMおきに取得
                    line = result[i]
                    body = line[0]

                    tmps = json.loads(body)
                    ask = tmps["ask"]
                    bid = tmps["bid"]
                    tmp_close = float(get_decimal_divide(get_decimal_add(ask, bid), "2"))
                    closes.append(tmp_close)

                if len(closes) != conf.MAX_LEN - (int(get_decimal_divide(conf.LOOP_TERM, conf.AI_MODEL_TERM)) - 1):
                    conf.LOGGER("Data Short! length:", len(closes))
                    err_flg = True
                    break
                conf.LOGGER("first_loop get remote rate end")

            for tmp_c in close:
                closes.append(tmp_c)  # 最初に取得したレートを追加
                closes_org.append(tmp_c)

            while True:
                if len(closes_org) > conf.MAX_LEN + 1:
                    closes_org.pop(0)
                else:
                    break

            while True:
                if len(closes) > conf.MAX_LEN + 1:
                    closes.pop(0)
                else:
                    break

            now_rate = close[-1]

            # 3分間レートが変わっていなかったら異常発生としメール送信！
            if len(closes_org) >= 180:
                startInd = len(closes_org) - 179
                rate_err = True
                for j in range(179):
                    if closes_org[startInd - 1] != closes_org[startInd + j]:
                        # 変化あったらエラーなし
                        rate_err = False
                        break

                if rate_err:
                    conf.LOGGER("rate has not Changed for 3 min !")
                    err_flg = True
                    break

            # 予想取得
            start_predict = time.perf_counter()
            try:
                if len(closes) != conf.MAX_CLOSE_LEN:
                    print("error!!! data length not correct:", len(closes), conf.MAX_CLOSE_LEN)
                    exit(1)

                #x = get_x(conf.AI_MODEL_TERM, False, base_t_just, conf.MODEL_DATA_LENGTH, conf.MODEL_INPUT_DATAS, conf.MODEL_INPUT_SEPARATE_FLG,
                #               conf.MODEL_METHOD, closes=closes)

                x = get_x_fragments(conf, closes)

                predict = predict_class.predict_on_batch(x)

                if conf.MODEL_TYPE == "CATEGORY":
                    response_text = str(predict[0][0]) + "_" + str(predict[0][1]) + "_" + str(predict[0][2])
                elif conf.MODEL_TYPE == "REGRESSION":
                    response_text = str(predict[0][0])


            except Exception as request_e:
                err_flg = True
                conf.LOGGER("response", response_text)
                conf.LOGGER(tracebackPrint(request_e))
                break

            predict_take = time.perf_counter() - start_predict

            print("score", base_t_just, datetime.datetime.fromtimestamp(base_t_just), "predict_take", predict_take, "res", response_text)

            start_db = time.perf_counter()

            div = get_divide(closes[-300], closes[-1]) if (conf.DIV_SEC != None and conf.DIV_SEC != "") else None
            sub = get_decimal_sub(closes[-300], closes[-1]) if (conf.SUB_SEC != None and conf.SUB_SEC != "") else None

            regist_score = base_t_just
            regist_time_str = datetime.datetime.fromtimestamp(regist_score)
            child = {
                'response': response_text,
                'now_rate': now_rate,
                'now_spread': conf.SPREAD,
                'predict_take': '{:.3f}'.format(predict_take),
                'time': str(regist_time_str),
                'loop_take': time.perf_counter() - start,
                'db_take': db_take,
                'close_take': close_take,
                'div' + str(conf.DIV_SEC): div,
                'sub' + str(conf.SUB_SEC): sub,
            }

            registRedis(conf, redis_db, regist_score, child, conf.DB_KEY)
            db_take = time.perf_counter() - start_db

            # 処理時間表示
            end = time.perf_counter()
            # 処理時間がconf.LOOP_TERM + 0.9以上の場合
            process_t = end - start
            if process_t > (conf.LOOP_TERM + 0.9):
                conf.LOGGER("time over:", process_t, close_take, predict_take, db_take, loop_take, sleep_time, )

            if first_loop:
                first_loop = False

            # 次に起動すべき時間
            base_t += conf.LOOP_TERM

            # 次のターンまでスリープする
            start_loop = time.perf_counter()

            sleep_time = base_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_take = time.perf_counter() - start_loop

        if err_flg:
            return_code = 3

    except Exception as e:
        conf.LOGGER("Error Occured!!:", tracebackPrint(e))

        return_code = 3

    return return_code


if __name__ == '__main__':
    conf = Conf()

    redis_db = redis.Redis(host='win2', port=6379, db=0, decode_responses=True)
    redis_fx_db = redis.Redis(host=conf.FX_DATA_MACHINE, port=6379, db=conf.FX_DB_NO, decode_responses=True,
                              socket_keepalive=True)

    result_data = redis_db.zrangebyscore("VANTAGE_USDJPY.p_S1", 1779671617 - conf.MAX_LEN, 1779671617, withscores=True)
    #result_data = redis_db.zrangebyscore("USDJPY_1_0", 1733154551 - conf.MAX_LEN, 1733154551, withscores=True)

    print(len(result_data))
    closes = []

    for i, line in enumerate(result_data):
        body = line[0]
        score = float(line[1])
        tmps = json.loads(body)
        ask = float(tmps["ask"])
        bid = float(tmps["bid"])
        closes.append(get_decimal_divide(get_decimal_add(ask, bid), 2))
        #closes.append(tmps["c"])

    retX = get_x_fragments(conf, closes)


    predict_class = load_model(conf.MODEL_DIR_PATH + conf.MODEL_NAME,
                           custom_objects={"root_mean_squared_error": root_mean_squared_error, })


    predict = predict_class.predict_on_batch(retX)
    print(predict)
    conf.LOGGER("end init predict class")
