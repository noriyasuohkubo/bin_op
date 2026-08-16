import numpy as np

# tensorflowのライブラリを読み込む
import tensorflow as tf
from tensorflow import keras

# 乱数シードを固定する
tf.random.set_seed(1234)

# kerasのライブラリを読み込む
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

# RMSEを計算するためにsklearnのライブラリを読み込む
from sklearn.metrics import r2_score, mean_squared_error

# 訓練データを用意する
x_train = np.array([
    [1, 4, 7, 10, 13, 16, 19],
    [2, 5, 8, 11, 14, 17, 20],
    [3, 6, 9, 12, 15, 18, 21],
    [4, 7, 10, 13, 16, 19, 22],
    [5, 8, 11, 14, 17, 20, 23],
    [6, 9, 12, 15, 18, 21, 24],
    [7, 10, 13, 16, 19, 22, 25],
    [8, 11, 14, 17, 20, 23, 26]])

# 教師データを用意する
y_train = np.array([
    [22, 25, 28],
    [23, 26, 29],
    [24, 27, 30],
    [25, 28, 31],
    [26, 29, 32],
    [27, 30, 33],
    [28, 31, 34],
    [29, 32, 35]])

y_train = np.array([
    [28],
    [29],
    [30],
    [31],
    [32],
    [33],
    [34],
    [35]])

# 訓練データと教師データをLSTMで読み込める形式に変換する
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

# モデルを定義する
#model = Sequential()

# 隠れ層の数は、10とする
# 活性化関数はreluに設定する
# input_shapeは、1回に入力するデータ＝1行分のデータ、となる
#model.add(LSTM(10, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
# 予測数の指定は、RepeatVectorで設定する。1回あたりの教師データの数が予測数となる。
#model.add(RepeatVector(y_train.shape[1]))
# RepeatVectorを設定しているので、return_sequences=TrueとTimeDistributedを設定する
#model.add(LSTM(10, activation='relu', return_sequences=True))
#model.add(TimeDistributed(Dense(1)))

input = keras.Input(shape=(x_train.shape[1], 1))
input2 = TimeDistributed(Dense(1))(input)

lstm = LSTM(10, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]))(input2)
#rv = RepeatVector(3)(lstm)
#lstm2 = LSTM(10, activation='relu', return_sequences=True)(rv)
#td = TimeDistributed(Dense(1))(lstm2)
#td = Dense(1)(rv[:,-1,:])
td = Dense(1)(lstm)
model = keras.Model(inputs=(input), outputs=td)

# モデルの情報を表示する
model.summary()

# モデルを作成する
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルを訓練する
model.fit(x_train, y_train, epochs=1000, verbose=1)

# 検証データを用意する
x_test = np.array([12, 15, 18, 21, 24, 27, 30])
x_test = x_test.reshape((1, x_test.shape[0], 1))

# 正解データを用意する
#y_test = np.array([33, 36, 39])
y_test = np.array([39])

# 検証データを用いて予測する
y_predict = model.predict(x_test)

# 予測結果を1次元化する
y_predict = np.ravel(y_predict)

# 予測データを表示する
print(f'予測結果： {y_predict}')

# モデルの精度を評価する
# R2、RMSEを計算する。R2は1.0に、RMSEは0.0に近いほど、モデルの精度は高い
r2_score = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print(f'r2_score: {r2_score:.4f}')
print(f'rmse: {rmse:.4f}')