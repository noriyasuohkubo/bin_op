import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

"""
SEE:https://github.com/hootnot/oandapyV20-examples
dukasのデータは日本時間の土曜朝7時から月曜朝7時までデータがないが
オアンダは土曜朝7時から月曜朝2時までデータがなく、オアンダの方がデータが豊富
しかし、オアンダの取引時間はdukasと同じなので、本番のjavaでオアンダのデータを取得する場合は、
月曜2時から6時台までのデータは消す必要がある
"""

access_token = "972dced2b5a8a2ae5beca555d9912cc8-4a27352a52c26d7091bda6b6f6b772de"
account_id = "6975470"
account_id_v20= "001-009-1776566-001"

api = API(access_token=access_token, environment="live")

params = {
    "granularity": "H1",  # 取得する足
    "count": 20,         # 取得する足数
    "price": "M",        # 仲値
    "from": "2009-01-02T20:00:00Z"
}

instrument = "GBP_JPY"   # 通貨ペア

instruments_candles = instruments.InstrumentsCandles(instrument=instrument, params=params)

api.request(instruments_candles)
response = instruments_candles.response

for k,v in response.items():
    print("k")
    print(k)
    print("v")
    print(v)

