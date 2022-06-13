import websocket
import requests
import json
# DOCUMENTATION: https://websocket-python.readthedocs.io/en/stable/websocket.html
# https://api.binance.com/api/v3/depth?symbol=BNBBTC&limit=1000
# https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
# https://stackoverflow.com/questions/70579991/websocket-client-python-no-data-from-the-binance-api
# https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md
# https://github.com/binance/binance-spot-api-docs


class BinanceWebSocket:
    cc = 'btcusdt'
    interval = '1m'

    on_message = None
    on_error = None
    on_close = None
    on_open = None

    def __init__(self):
        pass

    def run(self):
        socket = f'wss://stream.binance.com:9443/ws/{self.cc}@kline_{self.interval}'
        ws = websocket.WebSocketApp(socket,
                                    on_open=self.on_open,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        # websocket.enableTrace(True)
        ws.run_forever()

    def getAllSymbolTickets(self, callback=None):
        def on_message(ws, message):
            # save to file Txt
            file = './static/data/tickets.txt'
            with open(file, 'a') as f:
                f.write(message)
            ws.close()
        socket = f'wss://stream.binance.com:9443/ws/!ticker@arr'
        ws = websocket.WebSocketApp(socket,
                                    on_message=on_message)
        websocket.enableTrace(True)
        ws.run_forever()

    def getDataFromAPI(self, symbol, interval='1m'):
        # UPperCase symbol
        symbol = symbol.upper()
        url = "https://api.binance.com/api/v3/klines?symbol=" + \
            symbol + "&interval=" + interval + "&limit=1000"

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers)
        # convert to json
        data = json.loads(response.text)
        return data
