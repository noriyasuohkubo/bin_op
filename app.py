from flask import Flask, render_template, request, jsonify, abort, send_from_directory
import os
import logging.config
import redis
import json
from util import *
import send_mail as mail

current_dir = os.path.dirname(__file__)
logging.config.fileConfig( os.path.join(current_dir, "config", "logging_flask_app.conf"))
loggerConf = logging.getLogger("flask_app")

SIGNAL_HOST = '192.168.1.15' #win5
SIGNAL_DB_NO = 8
SIGNAL_DB_NAME = 'USDJPY_BINARY_TRADE_SIGNAL'

def printLog(logger):
    def f(*args):
        #print(*args)
        tmp_str = ""
        for i, a in enumerate(args):
            tmp_str = tmp_str + " " + str(a) if i != 0 else str(a)
        logger.info(tmp_str)
        print(tmp_str)

    return f

LOGGER = printLog(loggerConf)

app = Flask(__name__)

host_list = {}

redis_db = redis.Redis(host=SIGNAL_HOST, port=6379, db=SIGNAL_DB_NO, decode_responses=True,socket_keepalive=True)

# ブロックしたいIPアドレスのプレフィックスを指定
BLOCKED_IP_PREFIXES = ['85.208.96.', '185.191.171.']

@app.before_request
def block_ip():
    # アクセス元IPアドレスを取得
    remote_ip = request.remote_addr

    # ブロックされたIPアドレスの場合は403 Forbiddenを返す
    for prefix in BLOCKED_IP_PREFIXES:
        if remote_ip.startswith(prefix):
            abort(403)

@app.route("/sound/UP.mp3")
def sound_up():
    return send_from_directory("sound", "UP.mp3")

@app.route("/sound/DW.mp3")
def sound_dw():
    return send_from_directory("sound", "DW.mp3")

@app.route("/sound/ERROR.mp3")
def sound_error():
    return send_from_directory("sound", "ERROR.mp3")

@app.route("/predict", methods=['GET'])
def predict():
    data = request.get_json()

    return_str = ""

    try:
        result = redis_db.zrevrange(SIGNAL_DB_NAME, 0, 0, withscores=True)
        print(result)
        if len(result) == 1:
            line = result[0]
            body = line[0]

            tmps = json.loads(body)
            sign = tmps["sign"]
            score = str(int(tmps["score"]))
            probe_up = str(tmps["probe_up"])
            probe_same = str(tmps["probe_same"])
            probe_dw = str(tmps["probe_dw"])
            return_str = sign + "-" + score + "-" + probe_up[:4] + "-" + probe_same[:4] + "-" + probe_dw[:4]
        else:
            raise Exception("cannot get predict")

    except Exception as e:
        LOGGER("Error Occured!!:", tracebackPrint(e))
        mail.send_message("predict_app", "Error Occured!! see log!!!")
        return_str = "ERROR"

    return return_str

@app.route('/')
def index():
    #アクセス元のIPを記録しておく
    access_host = request.remote_addr
    if access_host in host_list.keys():
        host_list[access_host] += 1
    else:
        host_list[access_host] = 1

    LOGGER(host_list)
    #print(request.headers)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='192.168.1.15', debug=False)