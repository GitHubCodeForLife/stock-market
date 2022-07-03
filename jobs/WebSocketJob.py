from datetime import datetime, timedelta

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
        train_file = FileWaiter().getDataFile(self.criterias['symbol'])

        fileWaiter.saveToFile(datas, train_file)
        self.trainModel()
        self.emit({
            "criterias": criterias_copy,
            "type": "Init_data"
        })

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
            train_file = FileWaiter().getDataFile(self.criterias['symbol'])
            fileWaiter.saveAppendToFile(
                data, train_file)
            # print("Log on file")
            self.emit({"criterias": self.criterias,
                       "data": data, "type": "Append_data"})
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
        train_file = FileWaiter().getDataFile(self.criterias['symbol'])
        model_file = FileWaiter().getModelFile(
            self.criterias['symbol'], self.criterias['algorithm'], self.criterias['features'])

        # should train model here
        if (shouldTrain(model_file) == True):
            algorithm = AlgorithmFactory().getAlgorithm(
                self.criterias['algorithm'])
            algorithm.run_train(train_file, model_file)
            # LogService().logAppendToFile(
            #     "Train model for " + self.criterias['symbol'] + " " + self.criterias['algorithm'])
            print("WebSocketJob trainModel done: " + model_file)
        else:
            print("WebSocketJob trainModel skip: " + model_file)


# if file is not exist, then train model
# if file is exist, then check the last modified time
# if last modified time is less than 1 hour, then train model
def shouldTrain(model_file):
    if (FileWaiter.isFileExist(model_file) == False):
        return True
    else:
        last_modified_time = FileWaiter.getLastModifiedTime(model_file)

        if (last_modified_time < datetime.now() - timedelta(hours=1)):
            return True
        else:
            return False
