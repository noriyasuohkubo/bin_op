
from datetime import datetime

import time
from indices import index
from decimal import Decimal
import redis


export_db_no = 8
import_db_no = 8

export_db_name = "GBPJPY_30_SPR_TRADE"
import_db_name = "GBPJPY_30_SPR_TRADE"

export_host = "amd6"
import_host = "127.0.0.1"

start = datetime(2020, 12, 19, 23)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2021, 1, 15, 22)
end_stp = int(time.mktime(end.timetuple()))


def import_data():
    export_r = redis.Redis(host=export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host=import_host, port=6379, db=import_db_no)
    result_data = export_r.zrangebyscore(export_db_name, start_stp, end_stp, withscores=True)
    print(len(result_data))

    cnt = 0
    for line in result_data:
        body = line[0]
        score = line[1]
        imp = import_r.zrangebyscore(import_db_name, score, score)
        if len(imp) == 0:
            cnt += 1
            import_r.zadd(import_db_name, body, score)

    print(cnt)
if __name__ == "__main__":
    import_data()