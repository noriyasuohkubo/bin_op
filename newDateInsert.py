import redis
from datetime import datetime
import time

symbol = "GBP_JPY_CLOSE"

def import_data():

    r = redis.Redis(host= "127.0.0.1", port=6379, db=7)

    for i in range(300):
        body = 100.1 +i
        score = i
        r.zadd(symbol, body, score)

if __name__ == "__main__":
    import_data()
