import os
import arima
import arima_sparkts
import flint_regression
from matplotlib import pyplot


if __name__ == "__main__":


    data_path = "data/sales_data.csv"

    df = arima_sparkts.read_data(data_path)
    arima_sparkts.arima_ts(df)

    dfl = flint_regression.read_data(data_path)
    flint_regression.linear_regression(dfl)

    ts = arima.read_data(data_path)
    arima.predict(ts)

    #pyplot.show()
