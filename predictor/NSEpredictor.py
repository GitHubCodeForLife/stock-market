import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import numpy as np


class NSEpredictor:
    train = ""
    valid = ""

    def __init__(self):
        print("Stock Predictor")

        df = self.loadData("./static/data/NSE-TATA.csv")
        print("Data loaded")

        dataset = self.cleanData(df)
        print("Data cleaned")

        scaler, train, valid = self.normalizeData(dataset)
        print("Data normalized")

        model = self.loadModel("./static/data/saved_model.h5")
        print("Model loaded")

        closing_price = self.predict(model, scaler, dataset, valid)
        print("Prediction done")

        train, valid = self.visualize(closing_price, dataset)
        print("Visualization done")

        self.train = train
        self.valid = valid

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
        train = dataset[0:987, :]
        valid = dataset[987:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        return scaler, train, valid

    def loadModel(self, filename):
        return load_model(filename)

    def predict(self, model, scaler, new_data, valid):
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
        train = new_data[:987]
        valid = new_data[987:]
        valid['Predictions'] = closing_price
        return train, valid
