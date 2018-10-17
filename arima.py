from matplotlib import pyplot
import pandas as pd
import numpy
import statsmodels.api as sm
from pylab import rcParams
from pyramid.arima import auto_arima
from sklearn.metrics import r2_score


def read_data(path):
    data = pd.read_csv(path)
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index)
    ts = data['item_cnt_day'].resample('M').sum()

    return ts


def predict(ts):

    #decomposition = sm.tsa.seasonal_decompose(ts, freq=12, model='multiplicative')  # freq=1
    #fig = decomposition.plot()

    model = auto_arima(ts, start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                m=12,
                                start_P=0,
                                seasonal=True,
                                d=1, D=1,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    #print(stepwise_model.aic())

    train = ts.loc['2013-01-01':'2014-10-31']
    test = ts.loc['2014-11-01':'2015-04-30']
    nt = len(test)

    model.fit(train)

    future_forecast = model.predict(n_periods=nt)

    arima_forecast = future_forecast
    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    test = pd.DataFrame(test.values, index=test.index, columns=["Real sales"])
    tot = pd.DataFrame(ts.values, index=ts.index, columns=["Real sales"])
    pd.concat([test, future_forecast], axis=1).plot()
    pd.concat([tot, future_forecast], axis=1).plot()

    real = test.values
    arima_r2 = r2_score(real, arima_forecast)

    print("Pyramid ARIMA R2: ", arima_r2)
