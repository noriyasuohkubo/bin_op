import statsmodels.api as sm
import redis
from datetime import datetime
import time
import json
import pandas as pd
from matplotlib import pylab as plt

import numpy as np

"""
自己相関係数を調べる
"""
lag = 500
db_no = 0
db = "GBPJPY_2_0"
start = datetime(2021, 1, 1, 0)
end = datetime(2021, 1, 30, 22)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))

r = redis.Redis(host='localhost', port=6379, db=db_no, decode_responses=True)
result = r.zrevrangebyscore(db, end_score, start_score, withscores=True)

print("result_length:", len(result))
#print(result[:10])

close_lists = []

for line in result:
    body = line[0]
    score = int(line[1])
    tmps = json.loads(body)
    close_lists.append(tmps.get("d"))

#print(close_lists[:10])
nums = []
data_length = 0
for i, close in enumerate(close_lists):

    if i % 100000 == 0:
        print(datetime.now(),i)

    if len(close_lists[i:i+lag]) == lag:
        data_length += 1
        acfs,alphas = sm.tsa.stattools.acf(close_lists[i:i+lag],nlags=lag,alpha=.05) #95%信頼区間
        cnt = 0
        num = 0
        for acf,alpha in zip(acfs, alphas):
            if cnt != 0:
                if acf > alpha[1]:
                    num = cnt
                    print("acf:", acf, "alpha:", alpha[1])
                else:
                    nums.append(num)
                    break
            cnt += 1

print("data_length:", data_length)
print("nums_length:", len(nums))

nums_dict = {}

for num in nums:
    if num in nums_dict:
        nums_dict[num] = nums_dict[num] + 1
    else:
        nums_dict[num] = 1

for key in nums_dict.keys():
    print(key, nums_dict[key])

#print(acf[:100])
#print(alpha[:100])"

"""
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(111)
fig = sm.graphics.tsa.plot_acf(np.array(close_lists),lags=1000,ax=ax1,alpha=.005)
plt.show()
"""