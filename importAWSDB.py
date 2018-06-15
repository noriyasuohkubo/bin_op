import redis
from datetime import datetime
import time

import_db_nos = {"wordpress":1}
symbols = ["AUD_USD","EUR_USD","GBP_JPY","GBP_USD","NZD_USD","USD_CAD","USD_CHF","USD_JPY",]

export_db_no = 1

export_host = "wordpress"
import_host = "127.0.0.1"

start = datetime(2018, 2, 1)
start_stp = int(time.mktime(start.timetuple()))


end = datetime(2018, 5, 20)
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    import_db_no = import_db_nos.get(export_host)
    export_r = redis.Redis(host= export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host= import_host, port=6379, db=import_db_no)

    print("start:" + str(start_stp))

    for symbol in symbols:
        result_data = export_r.zrangebyscore(symbol, start_stp, end_stp, withscores=True)
        print("symbol:" + symbol)

        for line in result_data:
            body = line[0]
            score = line[1]
            imp = import_r.zrangebyscore(symbol , score, score)
            if len(imp) == 0:
                import_r.zadd(symbol, body, score)



if __name__ == "__main__":
    import_data()
