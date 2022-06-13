import websocket

# DOCUMENTATION: https://websocket-python.readthedocs.io/en/stable/websocket.html
# https://api.binance.com/api/v3/depth?symbol=BNBBTC&limit=1000
# https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
# https://stackoverflow.com/questions/70579991/websocket-client-python-no-data-from-the-binance-api
# https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md


class BinanceWebSocket:
    cc = 'btcusdt'
    interval = '1m'

    on_message = None
    on_error = None
    on_close = None

    def __init__(self):
        pass

    def run(self):
        socket = f'wss://stream.binance.com:9443/ws/{self.cc}@kline_{self.interval}'
        ws = websocket.WebSocketApp(socket,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        websocket.enableTrace(True)
        ws.run_forever()
