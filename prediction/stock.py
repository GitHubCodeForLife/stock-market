# STEP 1: import necessary libraries
from hashlib import new
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler


class StockPrediction:
    def __init__(self):
        print("Stock Prediction")
        self.loadData()
        print("Data Loaded successfully")
        self.cleanData()
        print("Data Cleaned successfully")
        self.normalizeData()
        print("Data Normalized successfully")
        self.trainedModel()
        print("Model Trained successfully")
        self.predict()
        print("Prediction done successfully")
        self.saveModel()
        print("Model Saved successfully")
        pass

    def loadData(self):
        df = pd.read_csv("./static/data/NSE-TATA.csv")
        self.df = df

    def cleanData(self):
        df = self.df
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']
        plt.figure(figsize=(16, 8))
        plt.plot(df["Close"], label='Close Price history')
        data = df.sort_index(ascending=True, axis=0)
        new_dataset = pd.DataFrame(index=range(
            0, len(df)), columns=['Date', 'Close'])
        for i in range(0, len(data)):
            new_dataset["Date"][i] = data['Date'][i]
            new_dataset["Close"][i] = data["Close"][i]
        self.new_dataset = new_dataset

    def normalizeData(self):
        new_dataset = self.new_dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        final_dataset = new_dataset.values
        train_data = final_dataset[0:987, :]
        valid_data = final_dataset[987:, :]

        new_dataset.reset_index()

        new_dataset.drop('Date', axis=1, inplace=True)
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(new_dataset)
        x_train_data, y_train_data = [], []
        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i-60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(
            x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(
            x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        self.x_train_data = x_train_data
        self.y_train_data = y_train_data
        self.valid_data = valid_data
        self.scaler = scaler

    def trainedModel(self):
        x_train_data = self.x_train_data
        y_train_data = self.y_train_data
        valid_data = self.valid_data
        scaler = self.scaler
        new_dataset = self.new_dataset

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(x_train_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))
        inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data,
                       epochs=1, batch_size=1, verbose=2)

        print(lstm_model.summary())
        self.lstm_model = lstm_model
        self.inputs_data = inputs_data

    def predict(self):
        lstm_model = self.lstm_model
        inputs_data = self.inputs_data
        scaler = self.scaler
        valid_data = self.valid_data
        new_dataset = self.new_dataset
        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(
            predicted_closing_price)
        train_data = new_dataset[:987]
        valid_data = new_dataset[987:]
        valid_data['Predictions'] = predicted_closing_price

        self.inputs_data = inputs_data
        self.train_data = train_data

        # return valid_data.loc[:, ['Predictions']]

    def saveModel(self):
        lstm_model = self.lstm_model
        inputs_data = self.inputs_data
        scaler = self.scaler
        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(
            predicted_closing_price)

        lstm_model.save('lstm_model.h5')

    def adjustModel(self, data):
        lstm_model = self.lstm_model
        scaler = self.scaler
        data = data.reshape(-1, 1)
        data = scaler.transform(data)
        x_test_data = []
        for i in range(60, data.shape[0]):
            x_test_data.append(data[i-60:i, 0])
        x_test_data = np.array(x_test_data)
        x_test_data = np.reshape(
            x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))
        lstm_model.fit(x_test_data, data, epochs=1, batch_size=1, verbose=2)
        return lstm_model
