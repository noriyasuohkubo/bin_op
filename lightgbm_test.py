import numpy as np
from sklearn import datasets, metrics, cross_validation
from lightgbm.sklearn import LGBMRegressor
import os
import optuna
import time
import m2cgen as m2c

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
clf = LGBMRegressor(max_depth=50,
                       num_leaves=21,
                    device="cpu",
                       n_estimators=100,
                       min_child_weight=1,
                       learning_rate=0.001,
                       nthread=24,
                       subsample=0.80,
#gpu_platform_id=1,gpu_device_id=0,#1080 Ti
                       colsample_bytree=0.80,
                       seed=42)

x_t, x_test, y_t, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
#処理時間計測
t1 = time.time()

clf.fit(x_t, y_t, eval_set=[(x_test, y_test)])
print("Mean Square Error: ", metrics.mean_squared_error(y_test, clf.predict(x_test)))
code = m2c.export_to_java(clf)
print(code)
t2 = time.time()
elapsed_time = t2-t1
print("経過時間：" + str(elapsed_time))