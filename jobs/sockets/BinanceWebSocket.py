import websocket
import requests
import json
# DOCUMENTATION:
# https://websocket-python.readthedocs.io/en/stable/websocket.html
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
        self.cc = self.cc.lower()
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

            # message
            # [{"e":"24hrTicker","E":1655112059452,"s":"ETHBTC","p":"-0.00237800","P":"-4.446","w":"0.05295085","x":"0.05349000","c":"0.05111200","Q":"1.98260000","b":"0.05111200","B":"29.60290000","a":"0.05111300","A":"30.63620000","o":"0.05349000","h":"0.05476800","l":"0.04908100","v":"798822.91180000","q":"42298.35082848","O":1655025659434,"C":1655112059434,"F":346375460,"L":347047808,"n":672349},{"e":"24hrTicker","E":1655112059429,"s":"LTCBTC","p":"-0.00006200","P":"-3.405","w":"0.00177986","x":"0.00182100","c":"0.00175900","Q":"10.56600000","b":"0.00175800","B":"41.15800000","a":"0.00175900","A":"49.49800000","o":"0.00182100","h":"0.00187300","l":"0.00171600","v":"482086.41400000","q":"858.04489692","O":1655025658943,"C":1655112058943,"F":80652319,"L":80732648,"n":80330},{"e":"24hrTicker","E":1655112059432,"s":"BNBBTC","p":"-0.00009200","P":"-0.979","w":"0.00939411","x":"0.00939300","c":"0.00930100","Q":"6.50000000","b":"0.00930300","B":"5.47900000","a":"0.00930400","A":"7.50000000","o":"0.00939300","h":"0.00960400","l":"0.00910000","v":"150864.83700000","q":"1417.24045680","O":1655025659380,"C":1655112059380,"F":187719519,"L":187832525,"n":113007}
            message = json.loads(message)
            symbols = []
            for i in message:
                symbols.append(i['s'])
            if callback:
                callback(symbols)

        socket = f'wss://stream.binance.com:9443/ws/!ticker@arr'
        ws = websocket.WebSocketApp(socket,
                                    on_message=on_message)
        # websocket.enableTrace(True)
        ws.run_forever()

    def getDataFromAPI(self, symbol, interval='1m'):

        url = "https://api.binance.com/api/v3/klines?symbol=" + \
            symbol + "&interval=" + interval + "&limit=1000"

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers)
        # convert to json
        data = json.loads(response.text)
        return data
