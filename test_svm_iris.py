import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# データ用意
iris = datasets.load_iris()    # データロード
X = iris.data                  # 説明変数セット
Y = iris.target                # 目的変数セット
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0) # random_stateはseed値。

print(Y)

# SVM実行
from sklearn.svm import SVC # SVM用
model = SVC()               # インスタンス生成
model.fit(X_train, Y_train) # SVM実行

# 予測実行
from sklearn import metrics       # 精度検証用
predicted = model.predict(X_test) # テストデーテへの予測実行
print(predicted[:100])