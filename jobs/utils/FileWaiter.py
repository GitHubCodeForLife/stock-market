
from helper.Constant import Constant
import json
import os
from datetime import datetime


class FileWaiter:
    def __init__(self):
        pass

    @staticmethod
    def isFileExist(fileName):
        return os.path.exists(fileName)

    @staticmethod
    def getLastModifiedTime(file):
        time = os.path.getmtime(file)
        datetime_object = datetime.fromtimestamp(time)
        return datetime_object

    @staticmethod
    def getTrainFile(symbol, algorithm, features):
        return Constant.TRAIN_FOLDER + "/" + symbol + "_" + algorithm + "_" + "".join(features) + ".csv"

    @staticmethod
    def getModelFile(symbol, algorithm, features):
        return Constant.TRAIN_FOLDER + "/" + symbol + "_" + algorithm + "_" + "".join(features) + ".h5"

    def saveToTrainFile(self, data, symbol, algorithm):

        data = json.loads(data)
        fileName = Constant.TRAIN_FOLDER + "/" + \
            data["Meta Data"]["2. Symbol"] + ".csv"
        print("Saving to file: " + fileName)

        # get Time Series
        timeSeries = data["Time Series (5min)"]

        # Save To File
        with open(fileName, 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            for key in timeSeries:
                f.write(key + "," + str(timeSeries[key]["1. open"]) + "," + str(timeSeries[key]["2. high"]) + "," + str(
                    timeSeries[key]["3. low"]) + "," + str(timeSeries[key]["4. close"]) + "," + str(timeSeries[key]["5. volume"]) + "\n")

        print("Saved to file: " + fileName + "successfully")

        fileTrainModel = Constant.TRAIN_FOLDER + "/" + \
            data["Meta Data"]["2. Symbol"] + ".h5"

        # create filetrainmodel if not exists
        print("Creating filetrainmodel: " + fileTrainModel)
        if not os.path.exists(fileTrainModel):
            open(fileTrainModel, 'w').close()

        self.saveToTxtFile(data, symbol)

        return fileName, fileTrainModel

    def saveAppendToFile(self, data, fileName):
       # create first row # Date , Open , High , Low , Close , Volume if not exists
        if not os.path.exists(fileName):
            with open(fileName, 'w') as f:
                f.write("Date,Open,High,Low,Close,Volume\n")

        # Save To File
        with open(fileName, 'a') as f:
            # Date , Open , High , Low , Close , Volume
            f.write(data["Date"] + "," + str(data["Open"]) + "," + str(data["High"]) + "," +
                    str(data["Low"]) + "," + str(data["Close"]) + "," + str(data["Volume"]) + "\n")

        # print("Saved to file: " + fileName + " successfully")

    def saveToFile(self, datas, fileName):
        # create first row # Date , Open , High , Low , Close , Volume if not exists
        with open(fileName, 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
        #
        #   [
        #     1499040000000,      // Open time
        #     "0.01634790",       // Open
        #     "0.80000000",       // High
        #     "0.01575800",       // Low
        #     "0.01577100",       // Close
        #     "148976.11427815",  // Volume
        #     1499644799999,      // Close time
        #     "2434.19055334",    // Quote asset volume
        #     308,                // Number of trades
        #     "1756.87402397",    // Taker buy base asset volume
        #     "28.46694368",      // Taker buy quote asset volume
        #     "17928899.62484339" // Ignore.
        #   ]
        # ]
        # loop to datas
        for data in datas:
            # convert Date from millisecond to Date in format "HH:MM:SS"
            data[0] = self.convertMillisecondToDate(data[0])
            with open(fileName, "a") as f:
                f.write(data[0] + "," + str(data[1]) + "," + str(data[2]) + "," +
                        str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "\n")
        print("Saved to file: " + fileName + " ssuccessfully")

    def saveToTxtFile(self, datas, symbol, algorithm):
        file = Constant.TRAIN_FOLDER + "/" + symbol + ".txt"
        data = self.convertDataToStandard(data)
        with open(file, "w") as f:
            f.write(str(data))

    def convertDataToStandard(self, data):
        # {
        #   "e": "kline",     // Event type
        #   "E": 123456789,   // Event time
        #   "s": "BNBBTC",    // Symbol
        #   "k": {
        #     "t": 123400000, // Kline start time
        #     "T": 123460000, // Kline close time
        #     "s": "BNBBTC",  // Symbol
        #     "i": "1m",      // Interval
        #     "f": 100,       // First trade ID
        #     "L": 200,       // Last trade ID
        #     "o": "0.0010",  // Open price
        #     "c": "0.0020",  // Close price
        #     "h": "0.0025",  // High price
        #     "l": "0.0015",  // Low price
        #     "v": "1000",    // Base asset volume
        #     "n": 100,       // Number of trades
        #     "x": false,     // Is this kline closed?
        #     "q": "1.0000",  // Quote asset volume
        #     "V": "500",     // Taker buy base asset volume
        #     "Q": "0.500",   // Taker buy quote asset volume
        #     "B": "123456"   // Ignore
        #   }
        # }
        data = json.loads(data)
        result = {}
        # result["Date"] = data["k"]["t"]
        result["Date"] = self.convertMillisecondToDate(data["k"]["t"])
        result["Open"] = data["k"]["o"]
        result["High"] = data["k"]["h"]
        result["Low"] = data["k"]["l"]
        result["Close"] = data["k"]["c"]
        result["Volume"] = data["k"]["v"]
        result["Symbol"] = data["k"]["s"]
        return result

    def convertMillisecondToDate(self, millisecond):
        return datetime.fromtimestamp(millisecond / 1000).strftime('%Y-%m-%d %H:%M:%S')
