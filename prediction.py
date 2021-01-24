import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras import backend
import keras
from glob import glob
import os

plt.style.use('fivethirtyeight')
features = ['High', 'Low', 'Open', 'Close']


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def plot_stock(train, valid, predictions):
    plt.figure(figsize=(16, 8))
    for i, key in enumerate(features, 1):
        plt.subplot(2, 2, i)
        plt.title(f'{key} value')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price USD ($)', fontsize=18)
        plt.plot(train[key])
        plt.plot(pd.concat([valid[key], predictions[key]], axis=1))
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def run():
    rdf = pd.read_csv('individual_stocks_5yr.csv')
    rdf['Date'] = pd.to_datetime(rdf['Date'])
    window_size = 80
    predicted = [s.split('.')[0][6:] for s in glob(os.path.join('model', '*.h5'))]
    tech_list = [x for x in rdf.Name.unique() if x not in predicted]

    for corp in tech_list:
        df = rdf[rdf['Name'] == corp][features]
        today = df[-1:].values
        df = df[:-1]
        dataset = df.dropna().values
        if len(dataset) < window_size:
            continue
        print(f"Last Date is: {list(rdf[rdf['Name'] == corp]['Date'][-1:])[0]}")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        training_data_len = int(np.ceil(len(dataset) * .8))
        train_data = scaled_data[:training_data_len, :]

        x_train = []
        y_train = []
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size: i, :])
            y_train.append(train_data[i, :])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dataset.shape[1]))
        # print(x_train.shape)

        model = Sequential()
        model.add(LSTM(window_size, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(window_size, return_sequences=True))
        model.add(LSTM(window_size, return_sequences=False))
        model.add(Dense(512))
        model.add(Dense(512))
        model.add(Dense(4))

        mc = keras.callbacks.ModelCheckpoint(filepath=f'model/{corp}.h5', verbose=True, save_best_only=True)
        model.compile(optimizer='adam', loss='mean_squared_error')

        rmse_mean = np.inf
        for epoch in [10, 20, 40, 100]:
            if rmse_mean < 5:
                break

            history = model.fit(x_train, y_train, batch_size=40, epochs=epoch,
                                callbacks=[mc], verbose=0, validation_split=0.3)

            test_data = scaled_data[training_data_len - window_size:, :]

            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(window_size, len(test_data)):
                x_test.append(test_data[i - window_size:i, :])

            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], dataset.shape[1]))

            predictions = model.predict(x_test)
            predictions = pd.DataFrame(scaler.inverse_transform(predictions))
            predictions.columns = features

            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            rmse_mean = np.mean(rmse)
            print(rmse_mean)

        train = df[:training_data_len]
        valid = df[training_data_len:]
        # plot_stock(train, valid, predictions)

        last_window_size_days = df[-window_size:].filter(features).values
        last_window_size_days_scaled = scaler.transform(last_window_size_days)

        X_test = [last_window_size_days_scaled]

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], last_window_size_days.shape[1]))

        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(f"Predicted {features}: {pred_price[0]}, Real: {today[0]}")


run()
