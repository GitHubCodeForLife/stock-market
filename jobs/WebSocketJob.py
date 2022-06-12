from jobs.BaseJob import BaseJob

from binance import ThreadedWebsocketManager

api_key = 'api_key'
api_secret = 'api_secret'
time = 10


class WebSocketJob(BaseJob):
    twm = None
    key_word = "kline"

    def __init__(self):
        super().__init__(time=time)
        # self.twm = ThreadedWebsocketManager(
        #     api_key=api_key, api_secret=api_secret)

    # override doJob
    def doJob(self):
        print("WebsocketJob doJob")
        # self.twm.start()
        # self.twm.start_kline_socket(
        #     callback=self.handle_socket_message, symbol="NSE-TATAGLOBAL")

        # self.twm.join()
        return "WebsocketJob doJob"

    def handle_socket_message(msg):
        print(f"message type: {msg['e']}")
        print(msg)
