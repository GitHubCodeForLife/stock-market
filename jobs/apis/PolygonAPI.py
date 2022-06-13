from polygon import RESTClient


api_key = "eL1LXRaWbyscSO72T_MJxu6bwp9ttzIM"
client = RESTClient(api_key)


class PolygonAPI:
    def __init__(self):

        pass

    def getData(self, symbol):
        trades = []

        for t in client.list_trades("TSLA", "2022-04-04", limit=5):
            trades.append(t)
        print(trades)

        return trades
