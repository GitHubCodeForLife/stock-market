# STEP 1: import necessary libraries
from hashlib import new
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


class LSTMTrainer:
    def __init__(self):
        print("Stock Trainee")

        df = self.loadData("./static/data/NSE-TATA.csv")
        print("Data Loaded successfully")

        new_dataset = self.cleanData(df)
        print("Data Cleaned successfully")

        x_train, y_train, valid_data, scaler = self.normalizeData(new_dataset)
        print("Data Normalized successfully")

        lstm_model, inputs_data, scaler = self.trainedModel(
            x_train, y_train, valid_data, scaler, new_dataset)
        print("Model Trained successfully")

        self.saveModel(lstm_model, inputs_data, scaler,
                       './static/data/saved_model.h5')
        print("Model Saved successfully")
        pass

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

    def normalizeData(self, new_dataset):
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

        return x_train_data, y_train_data, valid_data, scaler

    def trainedModel(self, x_train_data, y_train_data, valid_data, scaler, new_dataset):

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

        return lstm_model, inputs_data, scaler

    def saveModel(self, lstm_model, inputs_data, scaler, filename):
        X_test = []
        for i in range(60, inputs_data.shape[0]):
            X_test.append(inputs_data[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(
            predicted_closing_price)

        lstm_model.save(filename)
