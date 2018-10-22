import numpy
import pandas as pd
from pyspark import SparkContext, SQLContext
from pyspark.mllib.linalg import Vectors
from sparkts.models.ARIMA import autofit, fit_model
from pyspark.mllib.common import _java2py
from sklearn.metrics import r2_score


def read_data(path):

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    sql_context = SQLContext.getOrCreate(sc=sc)

    created_at = pd.read_csv(path)
    created_at = created_at.set_index('date')
    created_at.index = pd.to_datetime(created_at.index)
    ts = created_at['item_cnt_day'].resample('M').sum()
    ts_df = pd.Series.to_frame(ts)
    ts_df.reset_index(level=0, inplace=True)
    ts_df = ts_df.rename(columns={"date": "date", "item_cnt_day": "sales"})

    df = sql_context.createDataFrame(ts_df)

    return df


def arima_ts(df):

    sc = SparkContext.getOrCreate()

    train = df.filter(df['date'].between('2013-01-01', '2014-11-01'))
    test = df.filter(df['date'].between('2014-11-01', '2015-05-01'))

    tr = numpy.array(train.select("sales").collect()).flatten()
    te = numpy.array(test.select("sales").collect()).flatten()
    nte = len(te)

    #model = autofit(Vectors.dense(tr), sc=sc)
    model = fit_model(p=0, d=1, q=0, ts=Vectors.dense(tr), sc=sc)
    prev = model.forecast(Vectors.dense(tr), nte)

    x = _java2py(sc, prev)[len(tr):]

    #print("ARIMA spark-ts R2: ", r2_score(te, x))

    test = test.toPandas()
    test = test.set_index('date')

    df = df.toPandas()
    df = df.set_index('date')

    x = pd.DataFrame(x, index=test.index, columns=['prediction'])

    pd.concat([test, x], axis=1).plot()
    pd.concat([df, x], axis=1).plot()

    return r2_score(te, x)