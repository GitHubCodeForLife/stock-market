from jobs.sockets.BinanceWebSocket import BinanceWebSocket


class SocketFactory:
    def __init__(self):
        pass

    @staticmethod
    def getSocket(socketType="BinanceWebSocket"):
        return BinanceWebSocket()
