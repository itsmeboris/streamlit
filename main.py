import os
from datetime import timedelta
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

st.write("""
# Stock Price Display
This app Displays the **Stocks features**
""")


def user_input_features(df):
    stocks_name = st.sidebar.multiselect('Name', list(df['Name'].unique()), default=['AAPL'])
    stock_profit_range = st.sidebar.selectbox('Type', ('Day', 'Month', 'Quarter', 'Year'))
    key = st.sidebar.multiselect('Key', ('High', 'Low', 'Open', 'Close', 'Adj Close'), default='Close')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2011-01-21'))
    end_date = st.sidebar.date_input('End Date')
    predict = st.sidebar.radio('Predict', ['No', 'Yes'])
    future_prediction = st.sidebar.slider('Days to predict', 1, 200, 10)
    return stocks_name, key, pd.to_datetime(start_date), pd.to_datetime(
        end_date), stock_profit_range, predict, future_prediction


@st.cache
def get_data():
    df = pd.read_csv('individual_stocks_5yr.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


stocks = get_data()
models = [s.split('.')[0][6:] for s in glob(os.path.join('model', '*.h5'))]
window_size = 80

st.sidebar.header('User Input Features')
stocks_name, key, start_date, end_date, stock_profit_range, predict, future_prediction = user_input_features(stocks)
df = stocks.copy()
df = df[(df['Name'].isin(stocks_name)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]

if len(df) == 0 or len(key) == 0:
    st.write(f'No Information to display for stocks {" ".join(stocks_name)} from {start_date} to {end_date}')
else:
    with st.echo(code_location='below'):
        fig = px.line(data_frame=df, x='Date', y=key, animation_frame='Name')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=", ".join(key),
        )

        st.write(fig)


    def get_profit(arr):
        return arr[0]


    st.write(f"""Since we always care about the **profit** let's get the mean profit per {stock_profit_range}""")
    profit = pd.concat(
        [pd.DataFrame(pd.DataFrame(sdf['Open']).set_index(sdf['Date']).resample(stock_profit_range[0]).first()['Open']
                      - pd.DataFrame(sdf['Close']).set_index(sdf['Date']).resample(stock_profit_range[0]).last()[
                          'Close'], columns=[name]) for name, sdf in df.groupby('Name')], axis=1)
    with st.echo(code_location='below'):
        fig = px.line(profit)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title='Profit',
        )

        st.write(fig)

    with st.echo(code_location='below'):
        fig = px.scatter_3d(data_frame=df, x='Date', y='Volume', z='Close', color='Name')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title='Volume',
        )

        st.write(fig)

    if predict == 'Yes':
        st.write(f"""Now we want to predict for {future_prediction} days""")
        for stock_name in stocks_name:
            if stock_name in models:
                mdl = keras.models.load_model(f'model/{stock_name}.h5')
                scaler = MinMaxScaler(feature_range=(0, 1))
                features = ['High', 'Low', 'Open', 'Close']
                scaled_data = scaler.fit_transform(df[df['Name'] == stock_name][features].values)
                last_window_size_days = df[df['Name'] == stock_name][-window_size:].filter(['High', 'Low', 'Open', 'Close']).values
                future = []
                for i in tqdm(range(future_prediction)):
                    last_window_size_days_scaled = scaler.transform(last_window_size_days)
                    X_test = [last_window_size_days_scaled[i:]][0]
                    X_test = np.array(X_test)
                    X_test = np.reshape(X_test, (1, window_size, 4))
                    pred_price = mdl.predict(X_test)
                    future.extend(scaler.inverse_transform(pred_price))
                    last_window_size_days = np.reshape(np.append([last_window_size_days], scaler.inverse_transform(pred_price)),
                                               (window_size + i + 1, 4))

                future = pd.DataFrame(future, columns=['High', 'Low', 'Open', 'Close'])
                future.index = [list(df[df['Name'] == stock_name]['Date'][-1:])[0] + timedelta(days=i + 1)
                                for i in range(future_prediction)]
                st.write(f"""Showing Plots for {stock_name}""")
                fig = plt.figure(figsize=(30, 15))
                for i, key in enumerate(['High', 'Low', 'Open', 'Close'], 1):
                    plt.subplot(2, 2, i)
                    plt.title(f'{key} value')
                    plt.xlabel('Date', fontsize=18)
                    plt.ylabel('Price USD ($)', fontsize=18)
                    plt.plot(df[df['Name'] == stock_name]['Date'], df[df['Name'] == stock_name][key])
                    plt.plot(future[key])
                    plt.legend(['Train', 'Future Prediction'], loc='lower right')
                st.write(fig)
