import os
import logging.config
from decimal import Decimal


def makedirs(path):#dirなければつくる
    if not os.path.isdir(path):
        os.makedirs(path)

#定数ファイル
current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging.conf"))
loggerConf = logging.getLogger("app")

symbol = "GBPJPY"

# betする間隔(秒)
s = 2

# 予測する間隔:Sが2でpred_termが15なら30秒後の予測をする
pred_term = 15

# inputするデータの秒間隔
db1_term = 2
# inputするデータを秒をどれだけシフトして作成しているか
# 例:termが60でshiftが0,30なら60秒間隔でデータ作成していることは変わらないが
# 位相が"01:00,02:00,03:00..." と"01:30,02:30,03:30..."で異なっている
db1_shifts = []

for i in range(int(Decimal(str(db1_term)) / Decimal(str(s)))):
    db1_shifts.append(term - ((i + 1) * org_term) )

print(shift_list)

symbols = [symbol]


# 学習方法 Bidirectionalならby
# LSTMならlstm
#method = "lstm"
method = "lstm"

maxlen = 600
maxlen_min = 0

maxlen_min_str = ""
if maxlen_min != 0:
    maxlen_min_str = "(" + str(maxlen_min) + ")"



# db内データの秒間隔と、学習データの間隔が異る場合のcloseデータをずらす間隔
# 例えば,30秒予想でDBが2秒間隔で学習データの間隔を10秒とするなら
# s=10,merg=2でDBデータを5個ずらしてその変化率を学習データとする。
# だが、DBが1秒間隔で学習データの間隔も1秒だが、トレードタイミングであるmerg=2で,mergの方が大きい数字の場合、
# ずらす必要はないため、close_shiftは1とし、検証時(testLstm.py)は秒をmergで割った余りが0のデータだけを使って結果をみる
if s!=db:
    if int(s) >= int(merg):
        close_shift = int(Decimal(s) / Decimal(merg))
    else:
        close_shift = 1
else:
    close_shift = 1
print("close_shift:" + str(close_shift))

# 学習時、close_shiftが1より大きいならデータ作成時間の秒を学習データの間隔sで割った余りがdata_setの値のものだけつかうようにする
# またclose_shiftがiならデータ作成時間の秒をトレード間隔mergで割った余りがdata_setの値のものだけつかうようにする
# 何も設定されていなければ全てのセットを使用する
#
data_set = []
data_set_str = "_set_ALL"
if len(data_set) != 0:
    data_set_str = "_set"
    for set in data_set:
        data_set_str = data_set_str + "_" + str(set)

n_hidden ={
    1: 60,
    2: 0,
    3: 0,
    4: 0,
}
dense_hidden ={
    1: 0,
    2: 0,
}

min_hidden = ""

min_hidden_str = ""
if min_hidden != "":
    min_hidden_str = "_mhid_" + str(min_hidden)

hidden = ""
d_hidden = ""

for k, v in sorted(n_hidden.items()):
    hidden = hidden + "_hid" + str(k) + "_" + str(v)

for k, v in sorted(dense_hidden.items()):
    if v !=0:
        d_hidden = d_hidden + "_hid" + str(k) + "_" + str(v)

drop = 0.0
#特徴量の種類 close_divide:closeの変化率
in_features = ["close_divide",]
#in_features = ["open_divide",]


in_features_str = ""
for feature in in_features:
    if in_features_str == "":
        in_features_str = feature
    else:
        in_features_str = in_features_str + "-" + feature

#使用するより大きい足のデータ
#例:2秒足データに加えて1分足のデータも使用する
#in_longers = ["min_score",]
in_longers = []

in_longers_str = ""
for longer in in_longers:
    if in_longers_str == "":
        in_longers_str = "_" + longer
    else:
        in_longers_str = in_longers_str + "-" + longer

in_longers_db = {"min_score":"GBPJPY_M1",}

#インプットの特徴量の種類数
x_length = len(in_features) + len(in_longers)

if functional_flg:
    x_length = len(in_features)

spread = 1
#spread = 3

suffix = ".90*17"
db_suffix = ""

payout = 1000
payoff = 1000

fx = False
fx_position = 10000
db_no = 3
#学習対象外スプレッドを設けるか
except_low_spread = False
limit_spread = 0.00008

except_index = False

#学習対象外時間を設けるか
except_highlow = True

#学習対象外時間(ハイローがやっていない時間)
except_list = [20,21,22]

spread_list = {"spread0":(-1,0.00000),"spread2":(0.00000,0.00002), "spread4":(0.00002,0.00004),"spread6":(0.00004,0.00006),"spread8":(0.00006,0.00008)
    , "spread10": (0.00008, 0.00010), "spread12": (0.00010, 0.00012), "spread14": (0.00012, 0.00014), "spread16": (0.00014, 0.00016),"spread16Over":(0.00016,1),}

drawdown_list = {"drawdown1":(0,-10000),"drawdown2":(-10000,-20000),"drawdown3":(-20000,-30000),"drawdown4":(-30000,-40000),"drawdown5":(-40000,-50000),"drawdown6":(-50000,-60000),
                 "drawdown7": (-60000, -70000),"drawdown8": (-70000, -80000),"drawdown9": (-80000, -90000),"drawdown9over": (-90000, -1000000),}

model_dir = "/app/bin_op/model"
gpu_count = 2
batch_size = 1024 * 8 * gpu_count
#process_count = multiprocessing.cpu_count() - 1
process_count = 1
type = "category"

file_prefix = symbol+ "_" + method + functional_str + "_" + in_features_str + in_longers_str + "_" + s + "_m" + str(maxlen) + maxlen_min_str + "_term_" + str(pred_term * int(s)) + hidden + d_hidden + min_hidden_str + "_drop_" + str(drop)  + askbid + merg_file + data_set_str



# 0:新規作成
# 1:modelからロード
# 2:chekpointからロード
LOAD_TYPE = 1
LOADING_NUM = 1

# 1つのモデルに対して実行した学習回数
# モデルを引き続きロードして学習する場合1を足す
LEARNING_NUM = 1
if LOAD_TYPE == 0:
    LEARNING_NUM = 1

# 保存用のディレクトリ
MODEL_DIR = "/app/model/bin_op/" + file_prefix + "-" + str(LEARNING_NUM)
HISTORY_DIR = "/app/history/bin_op/" + file_prefix + "-" + str(LEARNING_NUM)
CHK_DIR = "/app/chk/bin_op/" + file_prefix + "-" + str(LEARNING_NUM)

# Load用のディレクトリ
MODEL_DIR_LOAD = "/app/model/bin_op/" + file_prefix + "-" + str(LOADING_NUM)
CHK_DIR_LOAD = "/app/chk/bin_op/" + file_prefix + "-" + str(LOADING_NUM)

makedirs(HISTORY_DIR)
makedirs(CHK_DIR)
makedirs(CHK_DIR_LOAD)

HISTORY_PATH = os.path.join(HISTORY_DIR, file_prefix)

LOAD_CHK_PATH = os.path.join(CHK_DIR_LOAD, "0180")



#ロガー関数を返す(標準出力と/app/bin_op/log/app.logに出力 )
def printLog(logger):
    def f(*args):
        print(*args)
        fmt =""
        for i, j in enumerate(args):
            fmt = fmt + "{" + str(i) + "} "
        logger.info(fmt.format(*args))

    return f

myLogger = printLog(loggerConf)

myLogger("Model is " , model_file)

#print("Model is ", model_file)

# モデル作成
def create_model():
    # FunctionalAPIで組み立てる
    # https://www.tensorflow.org/guide/keras/functional#manipulate_complex_graph_topologies
    # close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1 ))

    l2 = tf.keras.regularizers.l2(0.01)  # 正則化： L2、 正則化率： 0.01

    close_input = keras.Input(shape=(rnn_conf.CLOSE_STATE_SIZE, 1))
    #now_reward_input = keras.Input(shape=(1,))  # 現在の仮の報酬 決済したら得られる報酬
    #bet_state_input = keras.Input(shape=(1,))  # ベット中ならそのインデックスをいれる buy:1,stay=0,sell=-1
    bet_cnt_input = keras.Input(shape=(1,))  # ベットしてからのステップ数

    close_lstm = keras.layers.LSTM(rnn_conf.UNIT_CLOSE,
                                  return_sequences=False)(close_input)

    concate = keras.layers.Concatenate()([close_lstm, bet_cnt_input])

    main1 = Dense(rnn_conf.UNIT_MAIN1, activation="relu",
                  kernel_regularizer=l2,)(concate) # 正則化： L2、
    main2 = Dense(rnn_conf.UNIT_MAIN2, activation="relu",
                  kernel_regularizer=l2,)(main1) # 正則化： L2、
    main3 = Dense(rnn_conf.UNIT_MAIN3, activation="relu",
                  kernel_regularizer=l2,)(main2) # 正則化： L2、
    output = Dense(rnn_conf.OUTPUTS, )(main3)
    model = keras.Model(inputs=[close_input, bet_cnt_input], outputs=[output])

    model.compile(loss=huber, optimizer=Adam(lr=rnn_conf.LERNING_RATE), metrics=['mean_absolute_error'])
    #model.summary()

    return model

#行動価値関数定義
def get_model(single_flg):
    if single_flg:
        #複数GPUを使用しない CPU用
        if rnn_conf.LOAD_TYPE == 1:
            self.model = tf.keras.models.load_model(rnn_conf.MODEL_DIR_LOAD)
        elif rnn_conf.LOAD_TYPE == 2:
            self.model = create_model()
        else:
            self.model = create_model()
    else:
        #モデル作成
        with tf.distribute.MirroredStrategy().scope():
            # 複数GPU使用する
            # https://qiita.com/ytkj/items/18b2910c3363b938cde4
            if rnn_conf.LOAD_TYPE == 1:
                self.model = tf.keras.models.load_model(rnn_conf.MODEL_DIR_LOAD)
            elif rnn_conf.LOAD_TYPE == 2:
                self.model = create_model()
                self.model.load_weights(rnn_conf.LOAD_CHK_PATH)
            else:
                self.model = create_model()
