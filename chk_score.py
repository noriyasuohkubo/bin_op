import numpy as np
import pandas as pd
#import tensorflow as tf
from matplotlib import pyplot as plt
from decimal import Decimal
from datetime import datetime
import time




print('percent {:.2f}'.format(123.456))



start = datetime(2009, 1, 1)
end = datetime(2000, 1, 1)

start_score = int(time.mktime(start.timetuple()))
end_score = int(time.mktime(end.timetuple()))

print(start_score)
print(datetime.fromtimestamp(start_score))