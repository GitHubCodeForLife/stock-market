
from helper.Constant import Constant
import json
import os


class FileWaiter:
    def __init__(self):
        pass

    def saveToTrainFile(self, data):

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

        return fileName, fileTrainModel
