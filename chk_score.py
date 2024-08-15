import numpy as np
#import pandas as pd
#import tensorflow as tf
from matplotlib import pyplot as plt
from decimal import Decimal
from datetime import datetime
import time


a = [1,2]
print(type(a) == 'list')

print('percent {:.2f}'.format(123.456))



start = datetime(2007, 1, 1, )
end = datetime(2023, 4, 1,)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))
print(end_score - end_score%60)

print(start_score, end_score)
print(datetime.fromtimestamp(1706745600))
print(int(time.mktime(datetime(2024, 3, 1, 0,0,11).timetuple())))
