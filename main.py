import os
import arima
import arima_sparkts
import flint_regression
from pyspark import SparkContext
from matplotlib import pyplot


if __name__ == "__main__":

    os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"

    os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars lib/flint-assembly-0.6.0-SNAPSHOT.jar,' \
                                        'lib/sparkts-0.4.0-SNAPSHOT-jar-with-dependencies.jar' \
                                        ' --master local[*] pyspark-shell'

    data_path = "data/sales_data.csv"

    print("\nFitting spark-ts ARIMA.....")
    df = arima_sparkts.read_data(data_path)
    r2_ts = arima_sparkts.arima_ts(df)
    print("\nspark-ts ARIMA R squared: ", r2_ts)

    print("\n\nFitting flint linear regression.....", end=' ')
    dfl = flint_regression.read_data(data_path)
    r2_flint = flint_regression.linear_regression(dfl)
    print("done")
    print("\nLinear regression R squared: ", r2_flint)

    print("\n\nFitting pyramid ARIMA.....")
    ts = arima.read_data(data_path)
    r2_py = arima.predict(ts)
    print("\nPyramid ARIMA R squared:", r2_py)

    #pyplot.show()
