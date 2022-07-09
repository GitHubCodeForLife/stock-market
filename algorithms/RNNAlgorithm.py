import datetime
import numpy as np
import pandas as pd
from helper.log.LogService import LogService
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from algorithms.utils.PriceROC import PriceROC

scaler = MinMaxScaler(feature_range=(0, 1))


class RNNAlgorithm:
    features = ['Close']

    def __init__(self):
        pass

    def setFeatures(self, features):
        self.features = features

    def shouldUsePROC(self):
        if 'PROC' in self.features:
            return True
        return False

    def addPROCColumn(self,  dataset):
        PROC = PriceROC.caculateROC_list(dataset['Close'].values)
        dataset['PROC'] = PROC
        return dataset

    def run_train(self, train_file, filename_model):
        print("Stock Trainee LSTM: " + train_file)

        df = self.loadData(train_file)
        print("Data Loaded successfully")

        new_dataset = self.cleanData(df)
        print("Data Cleaned successfully")

        if self.shouldUsePROC():
            self.addPROCColumn(new_dataset)

        x_train, y_train = self.normalizeData(new_dataset)
        print("Data Normalized successfully")

        lstm_model = self.trainedModel(x_train, y_train)
        print("Model Trained successfully")

        self.saveModel(lstm_model, filename_model)

    def run_predict(self, train_file, filename_model):
        print("Stock Predictor")

        df = self.loadData(train_file)
        print("Data loaded")

        new_dataset = self.cleanData(df.copy())
        self.normalizeData(new_dataset)

        dataset = pd.DataFrame()
        dataset['Date'] = df['Date']
        dataset['Close'] = df['Close']

        if self.shouldUsePROC():
            self.addPROCColumn(dataset)

        lstm_model = self.loadModel(filename_model)

        predictions = self.predictFuture(lstm_model, dataset)
        print("Prediction done")

        return predictions, dataset

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

    def normalizeData(self, new_dataset: pd.DataFrame):
        final_dataset = new_dataset.values
        train_data = final_dataset

        new_dataset.index = new_dataset['Date']
        new_dataset.drop('Date', axis=1, inplace=True)

        scaled_data = scaler.fit_transform(new_dataset)
        x_train_data, y_train_data = [], []

        for i in range(60, len(train_data)):
            x_train_data.append(scaled_data[i-60:i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(
            x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(
            x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        return x_train_data, y_train_data

    def trainedModel(self, x_train_data, y_train_data):
        lstm_model = Sequential()
        lstm_model.add(
            SimpleRNN(units=50, return_sequences=True, input_shape=(60, 1)))
        lstm_model.add(SimpleRNN(units=50))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error',
                           metrics='accuracy', optimizer='adam')
        lstm_model.fit(x_train_data, y_train_data,
                       epochs=1, batch_size=1, verbose=2)
        return lstm_model

    def saveModel(self, lstm_model, filename):
        lstm_model.save(filename)

    def loadModel(self, filename):
        return load_model(filename)

    def predictFuture(self, model, dataset):
        steps = 60
        predictions = None

        current = datetime.datetime.fromisoformat(
            dataset['Date'].values[-1]) + datetime.timedelta(minutes=1)
        for i in range(steps):
            # print(dataset)
            inputs = dataset['Close'].values[-60:]
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_test = np.array(inputs)
            X_test = np.reshape(X_test, (1, 60))
            closing_price = model.predict(X_test)
            closing_price = scaler.inverse_transform(closing_price)

            temp_df = pd.DataFrame(
                {"Close": closing_price[0], "Date": current})
            # print(temp_df)
            if predictions is None:
                predictions = temp_df
            else:
                predictions = pd.concat(
                    [predictions, temp_df], ignore_index=True)
            current = current + datetime.timedelta(minutes=1)
            dataset = pd.concat([dataset, temp_df], ignore_index=True)

        return predictions
