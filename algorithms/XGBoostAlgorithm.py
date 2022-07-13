import datetime
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import plot_importance, plot_tree
from helper.log.LogService import LogService
from algorithms.utils.PriceROC import PriceROC
from algorithms.utils.ExpMovingAverage import ExpMovingAverage


class XGBoostAlgorithm:
    features = ['Close','EMA']

    def __init__(self):
        pass

    def setFeatures(self, features_lst):
        features_set = set(['Close','EMA'])
        if isinstance(features_lst, str):
            features_set.add(features_lst)
        else:
            features_set.update(features_lst)
        self.features = list(features_set)

    def shouldUsePROC(self):
        if 'PROC' in self.features:
            return True
        return False

    def shouldUseEMA(self):
        if 'EMA' in self.features:
            return True
        return False

    def addPROCColumn(self,  dataset):
        PROC = PriceROC.caculateROC_list(dataset['Close'].values)
        dataset['PROC'] = PROC
        return dataset

    def addEMAColumn(self,  dataset):
        EMA = ExpMovingAverage.calculateEMA_list(dataset['Close'].values)
        dataset['EMA'] = EMA
        return dataset

    def run_train(self, train_file, filename_model):
        print("This model don't support train")

    def run_predict(self, train_file, filename_model):
        print("Stock Predictor XGboost")
        df = self.loadData(train_file)
        print("Data loaded")

        dataset = pd.DataFrame()
        dataset['Date'] = df['Date']
        dataset['Close'] = df['Close']

        if self.shouldUsePROC():
            self.addPROCColumn(dataset)
        if self.shouldUseEMA():
            self.addEMAColumn(dataset)

        # multi steps forcasting
        steps = 60

        predictions = None
        current = datetime.datetime.fromisoformat(
            dataset["Date"].values[-1]) + datetime.timedelta(minutes=1)
        for i in range(steps):
            train_df, valid_df = self.devideData(dataset)
            prediction = self.xgboost_predict_and_forcast(train_df, valid_df)
            print("prediction: ")
            print(prediction)
            temp_df = pd.DataFrame(columns=dataset.columns, index=[0])

            temp_df["Close"][0] = prediction[0]
            temp_df["Date"][0] = current
            if self.shouldUsePROC():
                temp_df["PROC"] = PriceROC.caculateROC(
                    prediction[0], dataset['Close'][dataset.index[-1]])
            if self.shouldUseEMA():
                temp_df["EMA"] = ExpMovingAverage.calculateEMA(
                    prediction[0], dataset['EMA'][dataset.index[-1]])
            print(temp_df)
            if predictions is None:
                predictions = temp_df
            else:
                predictions = pd.concat(
                    [predictions, temp_df], ignore_index=True)
            current = current + datetime.timedelta(minutes=1)
            dataset = pd.concat([dataset, temp_df], ignore_index=True)
        if self.shouldUsePROC():
            predictions.drop("PROC", axis=1, inplace=True)
        if self.shouldUseEMA():
            predictions.drop("EMA", axis=1, inplace=True)
        return predictions, df

    def devideData(self, dataset):
        le = int(len(dataset)*0.9 - 1)
        train_df = dataset[:le]
        valid_df = dataset[le:]

        return train_df, valid_df

    def loadData(self, filename):
        return pd.read_csv(filename)

    def get_columns(self):
        res = self.features.copy()
        res.remove("Close")
        return res

    def create_features(self, df, label=None):
        label = self.get_columns()
        X = df[label]
        if label:
            y = df['Close']
            return X, y
        return X

    def xgboost_predict_and_forcast(self, train_df, valid_df):
        # build modle
        X_train, y_train = self.create_features(train_df, self.features)
        X_valid, y_valid = self.create_features(valid_df, self.features)

        reg = xgb.XGBRegressor(n_estimators=1000)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=50,
                verbose=False)

        dft = pd.DataFrame(data=[X_valid.iloc[-1]])
        result = reg.predict(dft)
        return result
