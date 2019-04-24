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
def ar_model(time_series_raw, time_lag=10):
    time_lag = time_lag + 1 #one extra for time lag
    train_length = len(time_series_raw['Value']) - time_lag
    y_hat = []
    mse_arr = []
    for train_index in range(0, train_length):
        train, test = time_series_raw['Value'].iloc[train_index:train_index+time_lag], time_series_raw['Value'].iloc[train_index+time_lag]
        start_date_train = time_series_raw['Date'].iloc[train_index]
        end_date_train = time_series_raw['Date'].iloc[train_index+time_lag]
        model = AR(train, dates=pd.date_range(start=start_date_train, end=end_date_train, freq='M'))
        model_fit = model.fit(maxlag=1)
        print(end_date_train)
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # model fit predict
        predictions = model_fit.predict(start=end_date_train, end=end_date_train, dynamic=False)
        # predictions.index = pd.DatetimeIndex(data=time_series_raw['Date'].iloc[train_index+time_lag])
        y_hat.append(predictions)
        # mse = mean_squared_error(predictions, test)
        # mse_arr.append(mse)
        # print('MSE: %s' % mse)

    mean_value = time_series_raw['Value'].mean
    print(mean_value)
    y_hat_normalized = y_hat
    plt.plot(y_hat_normalized)

    # plt.plot(time_series_raw['Value'])
    plt.show()


# Dickey fuller test to test if time series is stationary
# https://machinelearningmastery.com/time-series-data-stationary-python/
# https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
# Null hypothesis is that this series is not stationary
# p <= 0.05 indicates that this series is stationary
def dickey(time_series_raw):
    result = adfuller(time_series_raw['Value'])
    dickey_output = pd.Series(result[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    print('Results of Dickey Test')
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print (dickey_output)


# Null hypothesis is that the time series is stationary
# p <= 0.5 indicates that the series is not stationary
def kpss_test(time_series_raw):
    print ('Results of KPSS Test:')
    kpsstest = kpss(time_series_raw['Value'])
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
def no_diff_series(time_series_raw, name=None):
    dickey(time_series_raw)
    kpss_test(time_series_raw)
    ar_model(time_series_raw)
    plot_acf_data(time_series_raw, name=name)


def diff_one_series(time_series_raw, name=None):
    shift_one = diff_series(time_series_raw)
    dickey(shift_one)
    kpss_test(shift_one)
    shift_one['Date'] = shift_one.index
    ar_model(shift_one)
    #plot_acf_data(shift_one, name=name)


diff_one_series(time_series_hog, 'diff_one_series_hog')
# diff_one_series(time_series_soybean, 'diff_one_series_soybean')

