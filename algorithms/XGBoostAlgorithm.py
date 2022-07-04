import datetime
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import plot_importance, plot_tree
from helper.log.LogService import LogService
from algorithms.utils.PriceROC import PriceROC


class XGBoostAlgorithm:
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

        # multi steps forcasting
        steps = 100

        train_df, valid_df = self.devideData(dataset)

        predictions = None
        current = datetime.datetime.fromisoformat(
            dataset["Date"].values[-1]) + datetime.timedelta(minutes=1)
        for i in range(steps):
            prediction = self.xgboost_predict_and_forcast(train_df, valid_df)

            temp_df = pd.DataFrame({"Close": prediction, "Date": current})
            if self.shouldUsePROC():
                self.caculatePROCForPredition(predictions, temp_df)

            if predictions is None:
                predictions = temp_df
            else:
                predictions = predictions.append(temp_df, ignore_index=True)

            dataset = dataset.append(temp_df, ignore_index=True)
            train_df, valid_df = self.devideData(dataset)
            current = current + datetime.timedelta(minutes=1)

        return predictions, df

    def caculatePROCForPredition(self, predictions, temp_df):
        if predictions is None:
            temp_df['PROC'] = 0
            return
        PROC = PriceROC.caculateROC_list(predictions['Close'].values)
        predictions['PROC'] = PROC
        temp_df['PROC'] = PROC[-1]

    def devideData(self, dataset):
        le = int(len(dataset)*0.9 - 1)
        train_df = dataset[:le]
        valid_df = dataset[le:]

        return train_df, valid_df

    def loadData(self, filename):
        return pd.read_csv(filename)

    def create_features(self, df, label=None):
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

        # forcast
        data = self.createInput(X_valid)

        dft = pd.DataFrame(data=data, columns=self.getColumns())
        result = reg.predict(dft)
        return result

    def createInput(self, X_valid):

        data = None
        if self.shouldUsePROC():
            # create array has shape (1,2)
            data = np.array([[X_valid.values[-1][0], X_valid.values[-1][1]]])

        else:
            data = np.array([[X_valid.values[-1][0]]])
        return data

    def getColumns(self):
        if self.shouldUsePROC():
            return ['PROC', 'Close']
        return ['Close']
