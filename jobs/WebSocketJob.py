from datetime import datetime

from algorithms.AlogorithmFactory import AlgorithmFactory
from helper.log.LogService import LogService

from jobs.BaseJob import BaseJob
from jobs.sockets.SocketFactory import SocketFactory
from jobs.utils.FileWaiter import FileWaiter

fileWaiter = FileWaiter()


class WebSocketJob(BaseJob):
    criterias = None
    websocket = None

    # ATTRIBUTES
    current_time = datetime.now()
    STEPS_CONST = 10
    steps = 0

    def __init__(self, criterias):
        super().__init__()
        self.criterias = criterias

    def run(self):
        criterias_copy = self.criterias.copy()
        print("WebSocketJob doJob")
        websocket = SocketFactory().getSocket()
        # get 1000 rows until now and insert to the file
        datas = websocket.getDataFromAPI(self.criterias['symbol'])
        fileWaiter.saveToFile(
            datas, self.criterias['symbol'], self.criterias['algorithm'])
        self.trainModel()
        product = {'criterias': criterias_copy}
        self.emit(product=product)
        # use socket to insert realtime data to the file
        websocket.on_message = self.on_message
        websocket.on_error = self.on_error
        websocket.on_open = self.on_open
        websocket.cc = self.criterias['symbol']
        websocket.run()

    def on_open(self, ws):
        self.websocket = ws

    def on_message(self, ws, data):
        data = fileWaiter.convertDataToStandard(data)
        print(self.current_time)
        print(data)
        # check the same minute
        date_str = data['Date']
        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        if self.current_time.minute != date.minute:
            if self.steps == 0:
                self.steps = self.STEPS_CONST
                self.trainModel()
            self.steps = self.steps - 1
            fileWaiter.saveAppendToFile(
                data, self.criterias['symbol'], self.criterias['algorithm'])

        self.current_time = date

    def on_error(self, ws, error):
        print("WebsocketJob on_error")
        print(str(error))

    def runAgain(self, criterias):
        self.criterias = criterias
        self.websocket.close()
        self.restart()

    def restart(self):
        self.run()

    def trainModel(self):
        print("WebSocketJob trainModel")
        train_file = FileWaiter().getTrainFile(
            self.criterias['symbol'], self.criterias['algorithm'])
        model_file = FileWaiter().getModelFile(
            self.criterias['symbol'], self.criterias['algorithm'])
        algorithm = AlgorithmFactory().getAlgorithm(
            self.criterias['algorithm'])
        algorithm.run_train(train_file, model_file)
        LogService().logAppendToFile(
            "Train model for " + self.criterias['symbol'] + " " + self.criterias['algorithm'])
        print("WebSocketJob trainModel done: " + model_file)
