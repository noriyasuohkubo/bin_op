
from datetime import datetime

import time
from indices import index
from decimal import Decimal
import redis
import sys

export_host = "192.168.1.111"
import_host = "localhost"

export_db_no = 8
import_db_no = 8

print(datetime.now(), "import start!!!")

db_list = ["BTCUSD_30_SPR_DATA","GBPJPY_30_SPR","GBPJPY_30_SPR_TRADE","GBPJPY_60_SPR","GBPJPY_60_SPR_TRADE",]
#db_list = DB1_LIST + DB2_LIST + DB3_LIST + DB4_LIST + DB5_LIST

print(db_list)

start = datetime(2000, 1, 1,  )
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2022, 5, 1, )
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    export_r = redis.Redis(host=export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host=import_host, port=6379, db=import_db_no)

    for db_name in db_list:
        result_data = export_r.zrangebyscore(db_name, start_stp, end_stp, withscores=True)
        print(datetime.now(), db_name, len(result_data))

        cnt = 0
        for line in result_data:
            body = line[0]
            score = line[1]
            #imp = import_r.zrangebyscore(db_name, score, score)

            #if len(imp) == 0:
                #import_r.zadd(db_name, body, score)

            import_r.zadd(db_name, body, score)

            cnt = cnt + 1

        if cnt != len(result_data):
            print("import failed!!", len(result_data), cnt)
            sys.exit(1)


if __name__ == "__main__":
    import_data()