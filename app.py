from flask import Flask, request
import numpy as np
import keras.models
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
import logging.config
from keras.models import load_model

app = Flask(__name__)

#symbol = "EURUSD"
symbol = "GBPJPY"
db_no = 7

maxlen = 200
drop = 0.1
in_num=1
pred_term = 3
s = "10"
np.random.seed(0)
n_hidden =  30
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0
border = 0.54

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)
MODEL_DIR = config['lstm']['MODEL_DIR']

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop)

model_file = os.path.join(MODEL_DIR, file_prefix +".hdf5")

"""
model_file = os.path.join(MODEL_DIR, "bydrop_in1_" + s + "_m" + str(maxlen) + "_hid1_" + str(n_hidden)
                          + "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) +".hdf5")
"""
signal = ['UP','SAME','DOWN','SHORT']

# model and backend graph must be created on global
global model, graph
if os.path.isfile(model_file):
    model = load_model(model_file)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    #print("Load Model")
else:
    logger.warning("no model exists")

graph = tf.get_default_graph()

def get_redis_data():
    print("DB_NO:", db_no)
    r = redis.Redis(host='localhost', port=6379, db=db_no)
    result = r.zrevrange(symbol, 0  , maxlen
                      , withscores=False)
    if len(result) < maxlen:
        return None
    close_tmp= []
    result.reverse()
    for line in result:
        tmps = json.loads(line.decode('utf-8'))
        close_tmp.append(tmps.get("close"))
    #logger.info(close_tmp[len(result)-3:])
    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]

    dataX = np.zeros((1,maxlen, 1))
    dataX[:, :, 0] = close[:]
    #print("X SHAPE:", dataX.shape)

    return dataX

@app.route('/', methods=['GET'])
def root():
    dataX = get_redis_data()
    #data sort
    if dataX is None:
        return signal[3]

    with graph.as_default():  # use the global graph
        res = model.predict(dataX, verbose=0)[0]
        #print(res)
        pred = res.argmax()
        prob = res[pred]
        #logger.info("predicted:" + signal[pred] + " probup:" + str(res[0])+ " probsame:" + str(res[1])+ " probdown:" + str(res[2]))
        ret = signal[pred]
        if prob < border:
            ret = signal[1] # SAME
        return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
