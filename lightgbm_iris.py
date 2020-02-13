import numpy as np
import lightgbm as lgbm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.model_selection import StratifiedKFold

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=0)
print(type(X_train))
print(X_train[:10])
print(type(y_train))
print(y_train[:10])
lgb_train = lgbm.Dataset(X_train, y_train)
lgb_eval = lgbm.Dataset(X_valid, y_valid)

# LightGBM parameters
params = {
        'task': 'train',
        'device': 'gpu',
        'gpu_platform_id': 1, 'gpu_device_id' : 0,#1080 Ti
        'boosting_type': 'gbdt',
        'objective': 'multiclass', #多値分類
        'metric': {'multi_logloss'},
        'num_class': 3, #分類数
        'learning_rate': 0.1, #学習率 default:0.1
        'num_leaves': 23, #木にある分岐の個数 defalut:31
        'min_data_in_leaf': 1,
        'num_boost_round': 1000, #学習回数
        'verbose': -1 # 0:学習経過を表示 -1:非表示
}

#交差検証
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
kf = kfold.split(X, y)
for i, (train_fold, validate) in enumerate(kf):
    #print("train_fold")
    #print(train_fold)
    #print("validate")
    #print(validate)
    X_train, X_valid, y_train, y_valid = \
        X[train_fold, :], X[validate, :], y[train_fold], y[validate]
    #print(X_train[:10])
    #print(y_valid[:10])
    lgb_train = lgbm.Dataset(X_train, y_train)
    lgb_eval = lgbm.Dataset(X_valid, y_valid)

# train
evaluation_results = {} #評価結果格納用
clf = lgbm.train(params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],#評価用のデータセット
            valid_names=['Train', 'Valid'],
            early_stopping_rounds=10, #この回数やっても評価が改善しないなら学習を早めに切り上げる
            evals_result=evaluation_results,
            verbose_eval=20, #20回ごとに評価値を表示
            )
print('Saving model...')
# save model to file
clf.save_model('model.txt')

y_pred = clf.predict(X_valid,
                     num_iteration=clf.best_iteration, #一番よかったiterationで予想する(default)
                     )
print(y_pred[:10])
# 返り値は確率になっているので最尤に寄せる
y_pred_max = np.argmax(y_pred, axis=1)
print(y_pred_max[:10])
optimum_boost_rounds = clf.best_iteration

print('Loading model to predict...')
# load model to predict
clf2 = lgbm.Booster(model_file='model.txt')

#パフォーマンスを可視化
#参考:http://knknkn.hatenablog.com/entry/2019/02/02/164829
#Accuracy score: 正解率。1のものは1として分類(予測)し、0のものは0として分類した割合
#Precision score: 精度。1に分類したものが実際に1だった割合
#Recall score: 検出率。1のものを1として分類(予測)した割合
#F1 score: PrecisionとRecallを複合した0~1のスコア。数字が大きいほど良い評価。
print('Accuracy score = \t {}'.format(accuracy_score(y_valid, y_pred_max)))
print('Precision score = \t {}'.format(precision_score(y_valid, y_pred_max,average=None)))
print('Recall score =   \t {}'.format(recall_score(y_valid, y_pred_max,average=None)))
print('F1 score =      \t {}'.format(f1_score(y_valid, y_pred_max,average=None)))

# Plot the log loss during training
fig, axs = plt.subplots(1, 2, figsize=[15, 4])
axs[0].plot(evaluation_results['Train']['multi_logloss'], label='Train')
axs[0].plot(evaluation_results['Valid']['multi_logloss'], label='Valid')
axs[0].set_ylabel('Log loss')
axs[0].set_xlabel('Boosting round')
axs[0].set_title('Training performance')
axs[0].legend()

# Plot feature importance
importances = pd.DataFrame({'features': clf.feature_name(),
                            'importance': clf.feature_importance()}).sort_values('importance', ascending=False)
axs[1].bar(x=np.arange(len(importances)), height=importances['importance'])
axs[1].set_xticks(np.arange(len(importances)))
axs[1].set_xticklabels(importances['features'])
axs[1].set_ylabel('Feature importance (# times used to split)')
axs[1].set_title('Feature importance')
plt.show()

