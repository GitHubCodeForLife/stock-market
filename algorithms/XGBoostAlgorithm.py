import datetime
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import plot_importance, plot_tree
from helper.log.LogService import LogService


class XGBoostAlgorithm:
    def __init__(self):
        pass

    def run_train(self, train_file, filename_model):
        print("This model don't support train")

    def run_predict(self, train_file, filename_model):
        print("Stock Predictor XGboost")
        df = self.loadData(train_file)
        print("Data loaded")
        steps = 60

        dataset = pd.DataFrame()
        dataset['Date'] = df['Date']
        dataset['Close'] = df['Close']

       # multi steps forcasting
        steps = 100

        le = int(len(dataset)*0.8 - 1)
        train_df = dataset[:le]
        valid_df = dataset[le:]

        predictions = None
        current = datetime.datetime.fromisoformat(
            dataset["Date"].iloc[-1]) + datetime.timedelta(minutes=1)

        for i in range(steps):
            prediction = self.xgboost_predict_and_forcast(train_df, valid_df)
            temp_df = pd.DataFrame({"Close": prediction, "Date": current})
            if predictions is None:
                predictions = temp_df
            else:
                #  print(temp_df)
                predictions = predictions.append(temp_df, ignore_index=True)

            dataset = dataset.append(temp_df, ignore_index=True)
            le = int(len(dataset)*0.8 - 1)
            train_df = dataset[:le]
            valid_df = dataset[le:]
            current = current + datetime.timedelta(minutes=1)

        print("Predictions done")
        return predictions, df

    def loadData(self, filename):
        return pd.read_csv(filename)

    def create_features(self, df, label=None):
        X = df[label]
        if label:
            y = df[label]
            return X, y
        return X

    def xgboost_predict_and_forcast(self, train_df, valid_df):
        # build modle
        X_train, y_train = self.create_features(train_df, ['Close'])
        X_valid, y_valid = self.create_features(valid_df, ['Close'])
        reg = xgb.XGBRegressor(n_estimators=1000)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=50,
                verbose=False)

        # forcast
        columns = ["Close"]
        data = np.arange(1, 2, 1)
        dft = pd.DataFrame(data=data,  columns=columns)
        result = reg.predict(dft)
        return result
