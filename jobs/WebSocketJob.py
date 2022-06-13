from jobs.BaseJob import BaseJob
from jobs.ScheduleJob import ScheduleJob
from jobs.utils.FileWaiter import FileWaiter
from modeltrainer.FactoryTrainer import FactoryTrainer
from predictor.FactoryPredictor import FactoryPredictor
from jobs.sockets.SocketFactory import SocketFactory
import threading
from helper.log.LogService import LogService

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
        LogService().logAppendToFile("WebSocketJob" + "run")
        LogService().logAppendToFile(self.websocket)

    def on_message(self, ws, data):
        print("Web socket Binnance: " + threading.current_thread().name)
        fileWaiter.saveAppendToFile(
            data, self.criterias['symbol'], self.criterias['algorithm'])

    def on_error(self, ws, error):
        print("WebsocketJob on_error")
        print(str(error))

    def runAgain(self, criterias):
        self.criterias = criterias
        self.websocket.close()
        self.run()

    def getMockDataFromTxt(self):
        file = "./static/data/data.txt"
        with open(file, "r") as f:
            return f.read()
