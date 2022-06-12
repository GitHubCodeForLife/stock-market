import requests
from jobs.BaseJob import BaseJob
from jobs.utils.FileWaiter import FileWaiter
from modeltrainer.LSTMTrainer import LSTMTrainer
from predictor.NSEpredictor import NSEpredictor


time = 60/5

fileWaiter = FileWaiter()


class WebSocketJob(BaseJob):
    socket = None
    symbol = "MSFT"

    def __init__(self):
        super().__init__(time=time)

    def doJob(self):
        print("WebsocketJob doJob")

        # data = self.getMockDataFromTxt()
        data = self.getDataFromBinance()

        fileName, fileTrainModel = fileWaiter.saveToTrainFile(data)
        print("fileName: " + fileName)
        print("fileTrainModel: " + fileTrainModel)
        lstmTrainer = LSTMTrainer()
        lstmTrainer.run(fileName, fileTrainModel)

        predictor = NSEpredictor()
        predictor.run(fileName, fileTrainModel)

        valid, train = predictor.valid, predictor.train
        return {"valid": valid, "train": train}

    def setSymbol(self, symbol):
        self.symbol = symbol

    def getDataFromBinance(self):
        url = "https://alpha-vantage.p.rapidapi.com/query"

        querystring = {"interval": "5min", "function": "TIME_SERIES_INTRADAY",
                       "symbol": self.symbol, "datatype": "json", "output_size": "compact"}

        headers = {
            "X-RapidAPI-Key": "1c1a28d3a9mshc137ad75fd3c883p1c559cjsnd8c0303ebb90",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }

        response = requests.request(
            "GET", url, headers=headers, params=querystring)

        return response.text

    def getMockDataFromTxt(self):
        file = "./static/data/data.txt"
        with open(file, "r") as f:
            return f.read()
