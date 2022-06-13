from re import T
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import numpy as np
from helper.Constant import Constant

scaler = MinMaxScaler(feature_range=(0, 1))


class NSEpredictor:
    train = ""
    valid = ""

    def __init__(self):
        pass

    def run(self, trainFile=Constant.TRAIN_FILE, filename_model=Constant.MODEL_FILE):
        print("Stock Predictor")

        df = self.loadData(trainFile)
        print("Data loaded")

        dataset = self.cleanData(df)
        print("Data cleaned")

        train, valid = self.normalizeData(dataset)
        print("Data normalized")

        model = self.loadModel(filename_model)
        print("Model loaded")

        closing_price = self.predict(model, dataset, valid)
        print("Prediction done")

        train, valid = self.visualize(closing_price, dataset)
        print("Visualization done")

        self.train = train
        self.valid = valid

        return train, valid, dataset

    def loadData(self, filename):
        return pd.read_csv(filename)

    def cleanData(self, df):
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(
            0, len(df)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]

        return new_dataset

    def normalizeData(self, new_data):
        new_data.index = new_data.Date
        new_data.drop("Date", axis=1, inplace=True)
        dataset = new_data.values
        ex = len(new_data)*0.8 - 1
        ex = int(ex)

        train = dataset[0:ex, :]
        valid = dataset[ex:, :]

        scaled_data = scaler.fit_transform(dataset)

        return train, valid

    def loadModel(self, filename):
        return load_model(filename)

    def predict(self, model, new_data, valid):
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        return closing_price

    def visualize(self, closing_price, new_data):
        ex = int(len(new_data)*0.8 - 1)
        train = new_data[:ex]
        valid = new_data[ex:]
        valid['Predictions'] = closing_price
        return train, valid
