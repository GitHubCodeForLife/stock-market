import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import *
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from algorithms.utils.PriceROC import PriceROC
from algorithms.utils.ExpMovingAverage import ExpMovingAverage

scaler = MinMaxScaler(feature_range=(0, 1))

batch_size = 32
seq_len = 128
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        # Add dimension (batch, seq_len, 1)
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(
            x, self.weights_periodic) + self.bias_periodic)
        # Add dimension (batch, seq_len, 1)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        # shape = (batch, seq_len, 2)
        return tf.concat([time_linear, time_periodic], axis=-1)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

#############################################################################


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.linear = Dense(input_shape[0][-1],
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

#############################################################################


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(
            filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config


class TfmAndTedAlgorithm:
    features = ['Close']

    def __init__(self):
        pass

    def setFeatures(self, features_lst):
        features_set = set(['Close'])
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
        print("Stock Trainee TfmAndTed: " + train_file)

        df = self.loadData(train_file)
        print("Data Loaded successfully")

        new_dataset = self.cleanData(df)
        print("Data Cleaned successfully")

        if self.shouldUsePROC():
            self.addPROCColumn(new_dataset)
        if self.shouldUseEMA():
            self.addEMAColumn(new_dataset)

        x_train, y_train = self.normalizeData(new_dataset)
        print("X train, Y train: ")
        print(x_train.shape, y_train.shape)
        print("Data Normalized successfully")

        tfm_model = self.trainedModel(x_train, y_train, filename_model)
        print("Model Trained successfully")
        return tfm_model

    def run_predict(self, train_file, filename_model):
        print("Stock Predictor")

        df = self.loadData(train_file)
        print("Data loaded")

        new_dataset = self.cleanData(df.copy())
        self.normalizeData(new_dataset)

        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[[
            'Open', 'High', 'Low', 'Close', 'Volume']].rolling(10).mean()
        df.dropna(how='any', axis=0, inplace=True)

        dataset = pd.DataFrame()
        dataset['Date'] = df['Date']
        dataset['Close'] = df['Close']
        if self.shouldUsePROC():
            self.addPROCColumn(dataset)
        if self.shouldUseEMA():
            self.addEMAColumn(dataset)

        TfmAndTed_model = self.loadModel(filename_model)

        predictions = self.predictFuture(TfmAndTed_model, dataset)
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

        for i in range(seq_len, len(train_data)):
            x_train_data.append(scaled_data[i-seq_len:i])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(
            x_train_data), np.array(y_train_data)
        # x_train_data = np.reshape(
        #     x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        return x_train_data, y_train_data

    def trainedModel(self, x_train_data, y_train_data, filename):
        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(seq_len)
        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

        '''Construct model'''
        print("self feature len: ")
        print(self.features)
        print(self.features.__len__())
        in_seq = Input(shape=(seq_len, self.features.__len__()))
        x = time_embedding(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

        model.summary()

        callback = tf.keras.callbacks.ModelCheckpoint(filename,
                                                      monitor='val_loss',
                                                      verbose=1)
        history = model.fit(x_train_data, y_train_data,
                            batch_size=batch_size,
                            epochs=1,
                            callbacks=[callback])
        return history

    def loadModel(self, filename):
        return load_model(filename, custom_objects={'Time2Vector': Time2Vector,
                                                    'SingleAttention': SingleAttention,
                                                    'MultiAttention': MultiAttention,
                                                    'TransformerEncoder': TransformerEncoder})

    def predictFuture(self, model, dataset):
        steps = 60
        predictions = None

        current = datetime.datetime.fromisoformat(
            dataset['Date'].values[-1]) + datetime.timedelta(minutes=1)
        for i in range(steps):
            inputs = dataset.loc[:, dataset.columns != 'Date']
            inputs = scaler.fit_transform(inputs)
            inputs = np.array(inputs)[-seq_len:]
            X_test = inputs
            X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))

            # print("X_test shape: ")
            # print(X_test.shape)
            closing_price = model.predict(X_test)
            scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))
            closing_price = scaler.inverse_transform(
                closing_price.reshape(-1, 1))

            temp_df = pd.DataFrame(columns=dataset.columns, index=[0])
            temp_df["Close"][0] = closing_price[0][0]
            temp_df["Date"][0] = current
            if self.shouldUsePROC():
                temp_df["PROC"][0] = PriceROC.caculateROC(
                    closing_price[0][0], dataset['Close'][dataset.index[-1]])
            if self.shouldUseEMA():
                temp_df["EMA"][0] = ExpMovingAverage.calculateEMA(
                    closing_price[0][0], dataset['EMA'][dataset.index[-1]])
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
        return predictions
