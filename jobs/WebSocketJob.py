from jobs.BaseJob import BaseJob
from jobs.utils.FileWaiter import FileWaiter
from jobs.sockets.SocketFactory import SocketFactory
from helper.log.LogService import LogService
# import threading

fileWaiter = FileWaiter()


class WebSocketJob(BaseJob):
    criterias = None
    websocket = None

    def __init__(self, criterias):
        super().__init__()
        self.criterias = criterias

    def run(self):
        print("WebSocketJob doJob")
        websocket = SocketFactory().getSocket()
        # get 1000 rows until now and insert to the file
        datas = websocket.getDataFromAPI(self.criterias['symbol'])
        fileWaiter.saveToFile(
            datas, self.criterias['symbol'], self.criterias['algorithm'])

        # use socket to insert realtime data to the file
        websocket.on_message = self.on_message
        websocket.on_error = self.on_error
        websocket.on_open = self.on_open
        websocket.cc = self.criterias['symbol']
        websocket.run()
        return None

    def on_open(self, ws):
        self.websocket = ws

    def on_message(self, ws, data):
        # print("Web socket Binnance: " + threading.current_thread().name)
        fileWaiter.saveAppendToFile(
            data, self.criterias['symbol'], self.criterias['algorithm'])

    def on_error(self, ws, error):
        print("WebsocketJob on_error")
        print(str(error))
        LogService().logAppendToFile("Error: "+str(error))

    def runAgain(self, criterias):
        self.criterias = criterias
        self.websocket.close()
        self.run()
