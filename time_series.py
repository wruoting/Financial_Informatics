import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
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


def plot_pacf_data(time_series_raw, nlags=50, name=None):
    plt.bar(np.arange(0, nlags+1), pacf(time_series_raw['Value'], nlags=nlags), width=0.5)
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.savefig(name)
    plt.close()


def plot_seasonal_decompose(time_series_raw):
    time_series_raw = time_series_raw.drop(columns='Date')
    decomposition = sm.tsa.seasonal_decompose(time_series_raw, model='additive')
    decomposition.plot()
    plt.show()


# https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
def ar_model(time_series_raw, time_lag=10, max_lag=1, y_label='', name=None, title="AR Model"):
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
    plt.plot(time_series_raw.index, time_series_raw['Value'], label='Real Values')
    plt.plot(y_hat.index, y_hat['Value'], label='Predicted Values')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.savefig(name)
    plt.close()


def arma_model(time_series_raw, coefficients=None, time_lag=10, max_lag=1, y_label=''):
    train_length = len(time_series_raw['Value']) - time_lag
    y_hat = pd.DataFrame([], columns=['Value'])
    for train_index in range(0, train_length):
        train, test = time_series_raw['Value'].iloc[train_index:train_index+time_lag], time_series_raw['Value'].iloc[train_index+time_lag]
        start_date_train = time_series_raw['Date'].iloc[train_index]
        end_date_train = time_series_raw['Date'].iloc[train_index+time_lag-1]
        predict_test = time_series_raw['Date'].iloc[train_index+time_lag]
        model = ARMA(train, [coefficients[0], coefficients[1]], dates=pd.date_range(start=start_date_train, end=end_date_train, freq='M'))
        model_fit = model.fit(maxlag=max_lag, disp=-1)
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
    plt.plot(y_hat.index, y_hat['Value'], label='Predicted Values')
    plt.plot(time_series_raw.index, time_series_raw['Value'], label='Real Values')
    plt.legend(loc='upper left')
    plt.title("ARMA Model")
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.show()


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
        values = values.shift(1) - values
        values = values.dropna()
    values = values.to_frame()
    values.iloc[1].name = 'Value'
    return values


# plot_seasonal_decompose(time_series_hog)
def no_diff_series(time_series_raw, dickey_toggle=False, kpss_toggle=False, name=None, max_lag=None):
    if dickey_toggle:
        dickey(time_series_raw, max_lag=max_lag)
    if kpss_toggle:
        kpss_test(time_series_raw, max_lag=max_lag)
    if name:
        plot_acf_data(time_series_raw, name=name)
    return time_series_raw


def diff_one_series(time_series_raw, dickey_toggle=False, kpss_toggle=False, name=None, max_lag=None):
    shift_one = diff_series(time_series_raw)
    if dickey_toggle:
        dickey(shift_one, max_lag=max_lag)
    if kpss_toggle:
        kpss_test(shift_one, max_lag=max_lag)
    shift_one['Date'] = shift_one.index
    if name:
        plot_acf_data(shift_one, name=name)
    return shift_one


time_series_hog = open_data(path="{}".format('./ODA-PPORK_USD_LEAN_HOG_1980_2017.csv'))
time_series_soybean = open_data(path="{}".format('./ODA-PSOYB_USD_SOYBEAN_PRICE_1980_2017.csv'))

time_series_hog = parse_into_dataframe(time_series_hog)
time_series_soybean = parse_into_dataframe(time_series_soybean)

## No diff series and shift one series for AR models as well as potential dickey and kpss modeling
no_diff_series_hog = no_diff_series(time_series_hog, max_lag=0, name='no_diff_series_hog')
shift_one_series_hog = diff_one_series(time_series_hog, max_lag=0, name='diff_one_series_hog')
# ar_model(no_diff_series, max_lag=1,  y_label='No Diff Values')
# ar_model(shift_one_series, max_lag=1, y_label='Diff 1 Values')
# plot_acf_data(no_diff_series, name='ACF_diff_0')
# plot_pacf_data(shift_one_series, name='PACF_diff_1')
# plot_pacf_data(no_diff_series, name='PACF_diff_0')

## Code used to generate diff two series
# shift_two_series = diff_series(time_series_hog, diff=2)
# plot_acf_data(shift_two_series, name='ACF_diff_2')

# Can we improve on this series with an ARIMA?
# ar_model(no_diff_series_hog, max_lag=0, y_label='Diff 0 Values', name='Diff_0_Overlay', title='AR(0) Model')
# ar_model(no_diff_series_hog, max_lag=1, y_label='Diff 1 Values', name='Diff_1_Overlay', title='AR(1) Model')
# arma_model(no_diff_series, coefficients=[1, 1], max_lag=1, y_label='Diff 1 Values')
# arma_model(shift_one_series, coefficients=[0, 1], max_lag=0, y_label='Diff 1 Values')


# Max lag 0 is the best for no ar model (looks like an AR(1))
# We can see that arma 0, 1 for a shift doesn't have as good a result
# We could smooth the data with mean and try to get a better MSE with that?

## Soy Futures

no_diff_series_soy = no_diff_series(time_series_soybean, dickey_toggle=True, kpss_toggle=True, max_lag=0, name='no_diff_series_soy')
shift_one_series_soy = diff_one_series(time_series_soybean, dickey_toggle=True, kpss_toggle=True, max_lag=0, name='diff_one_series_soy')

#
# ADF (Augmented Dickey Fuller)
# https://freakonometrics.hypotheses.org/12729
# https://www.investopedia.com/articles/trading/07/stationary.asp
# https://www.xycoon.com/ma1_process.htm
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://freakonometrics.hypotheses.org/12729
# https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
# Plotting rolling mean and std dev could help
# https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/