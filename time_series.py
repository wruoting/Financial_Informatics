import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error


import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def open_data(path=None):
    time_series = []
    text_file = open(path, "r")
    lines = text_file.read().split('\n')
    for element in lines:
        row_item = element.split(',')
        time_series.append(row_item)
    return time_series


def parse_into_dataframe(data):
    # btdubs whitespace at parts of your file fuck you just sayin
    data_frame = pd.DataFrame(np.array(data[1:]), columns=[data[0][0], data[0][1]])
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%Y-%m-%d')
    data_frame['Value'] = pd.to_numeric(data_frame['Value'])
    data_frame.index = pd.DatetimeIndex(data=data_frame['Date'])
    data_frame = data_frame.sort_index(axis=0, ascending=True)
    # data_frame = data_frame.drop(columns='Date')
    return data_frame


def plot_raw_data(time_series_raw, name=None):
    plt.plot(time_series_raw['Date'],  time_series_raw['Value'])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(name)
    plt.close()


def plot_acf_data(time_series_raw, nlags=50, name=None):
    plt.bar(np.arange(0, nlags+1), acf(time_series_raw['Value'], nlags=nlags), width=0.5)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.savefig(name)
    plt.close()


def plot_seasonal_decompose(time_series_raw):
    time_series_raw = time_series_raw.drop(columns='Date')
    decomposition = sm.tsa.seasonal_decompose(time_series_raw, model='additive')
    decomposition.plot()
    plt.show()


# https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
def ar_model(time_series_raw, time_lag=10, max_lag=1):
    train_length = len(time_series_raw['Value']) - time_lag
    y_hat = pd.DataFrame([], columns=['Value'])
    for train_index in range(0, train_length):
        train, test = time_series_raw['Value'].iloc[train_index:train_index+time_lag], time_series_raw['Value'].iloc[train_index+time_lag]
        start_date_train = time_series_raw['Date'].iloc[train_index]
        end_date_train = time_series_raw['Date'].iloc[train_index+time_lag-1]
        predict_test = time_series_raw['Date'].iloc[train_index+time_lag]
        model = AR(train, dates=pd.date_range(start=start_date_train, end=end_date_train, freq='M'))
        model_fit = model.fit(maxlag=max_lag)
        predictions = model_fit.predict(start=predict_test, end=predict_test, dynamic=True)
        predictions = pd.DataFrame(predictions[0], columns=['Value'],
                                   index=pd.DatetimeIndex(data=predictions.index.date))
        y_hat = y_hat.append(predictions)
    # Drop the first time_lag+1 rows
    time_series_raw = time_series_raw[time_lag:]
    # MSE
    diff_score = time_series_raw['Value'].subtract(y_hat['Value'], axis=0)
    diff_score = diff_score.dropna()**2
    mse = diff_score.sum()
    print("MSE: {}".format(mse))
    # plt.plot(y_hat.index, y_hat['Value'], label='Predicted Values')
    # plt.plot(time_series_raw.index, time_series_raw['Value'], label='Real Values')
    # plt.legend(loc='upper left')
    # plt.title("AR Model")
    # plt.xlabel("Date")
    # plt.ylabel("Offset 1 Diff")
    # plt.show()


# Dickey fuller test to test if time series is stationary
# https://machinelearningmastery.com/time-series-data-stationary-python/
# https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
# Null hypothesis is that this series is not stationary
# p <= 0.05 indicates that this series is stationary
def dickey(time_series_raw, max_lag=None):
    result = adfuller(time_series_raw['Value'], maxlag=max_lag)
    dickey_output = pd.Series(result[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    print('Results of Dickey Test')
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print (dickey_output)


# Null hypothesis is that the time series is stationary
# p <= 0.5 indicates that the series is not stationary
def kpss_test(time_series_raw, max_lag=None):
    print ('Results of KPSS Test:')
    kpsstest = kpss(time_series_raw['Value'], lags=max_lag)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print (kpss_output)


def diff_series(time_series_raw, diff=1):
    values = time_series_raw['Value']
    for d in range(1, diff+1):
        values = values - values.shift(1)
        values = values.dropna()
    values = values.to_frame()
    values.iloc[1].name = 'Value'
    return values

time_series_hog = open_data(path="{}".format('./ODA-PPORK_USD_LEAN_HOG_1980_2017.csv'))
time_series_soybean = open_data(path="{}".format('./ODA-PSOYB_USD_SOYBEAN_PRICE_1980_2017.csv'))

time_series_hog = parse_into_dataframe(time_series_hog)
time_series_soybean = parse_into_dataframe(time_series_soybean)


# plot_seasonal_decompose(time_series_hog)
def no_diff_series(time_series_raw, dickey_toggle=False, kpss_toggle=False, name=None,  max_lag=None):
    if dickey_toggle:
        dickey(time_series_raw, max_lag=max_lag)
    if kpss_toggle:
        kpss_test(time_series_raw, max_lag=max_lag)
    # ar_model(time_series_raw,  max_lag=max_lag)
    plot_acf_data(time_series_raw, name=name)


def diff_one_series(time_series_raw, dickey_toggle=False, kpss_toggle=False, name=None, max_lag=None):
    shift_one = diff_series(time_series_raw)
    if dickey_toggle:
        dickey(shift_one, max_lag=max_lag)
    if kpss_toggle:
        kpss_test(shift_one, max_lag=max_lag)
    shift_one['Date'] = shift_one.index
    # ar_model(shift_one, max_lag=max_lag)
    plot_acf_data(shift_one, name=name)


no_diff_series(time_series_hog, max_lag=0, name='no_diff_series_hog')
diff_one_series(time_series_hog, max_lag=0, name='diff_one_series_hog')
# diff_one_series(time_series_soybean, 'diff_one_series_soybean')
