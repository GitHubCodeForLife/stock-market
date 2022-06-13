import webbrowser

import websockets
from jobs.BaseJob import BaseJob
from jobs.utils.FileWaiter import FileWaiter
from modeltrainer.FactoryTrainer import FactoryTrainer
from predictor.FactoryPredictor import FactoryPredictor
from jobs.sockets.SocketFactory import SocketFactory
import threading

fileWaiter = FileWaiter()


class WebSocketJob(BaseJob):
    symbol = "MSFT"
    algorithm = "LSTM"
    feature = "Close"

    websocket = None
    isChange = True

    def __init__(self):
        super().__init__()

    def run(self):
        if(self.isChange):
            websocket = SocketFactory().getSocket()
            websocket.on_message = self.on_message
            websocket.on_error = self.on_error
            websocket.run()

        self.websocket = websocket

    def on_message(self, ws, data):
        # print current thread
        print("Web socket Binnance: " + threading.current_thread().name)
        print("on_message: " + str(ws))
        print(str(data))

        fileWaiter.saveAppendToFile(data, self.symbol, self.algorithm)

        # fileName, fileTrainModel = fileWaiter.saveToTrainFile(
        #     data, self.symbol, self.algorithm)

        # trainer = FactoryTrainer().getTrainer(self.algorithm, self.feature)
        # trainer.run(fileName, fileTrainModel)

        # predictor = FactoryPredictor().getPredictor(self.algorithm, self.feature)
        # train, valid, dataset = predictor.run(fileName, fileTrainModel)

        # self.emit({"valid": valid, "train": train, "dataset": dataset})

        self.emit({"valid": "valid", "train": "train", "dataset": "dataset"})

    def on_error(self, ws, error):
        print("WebsocketJob on_error")
        print(str(error))

    def setSymbol(self, symbol):
        self.symbol = symbol

    def setAlgorithm(self, algorithm):
        self.algorithm = algorithm

    def getMockDataFromTxt(self):
        file = "./static/data/data.txt"
        with open(file, "r") as f:
            return f.read()
