import ts.flint
from pyspark import SparkContext, SQLContext
from pyspark.sql.functions import col
from pyspark.sql import types


def date_parser(fmt):
    @ts.flint.udf(types.LongType())
    def parse(x):
        dt = types.datetime.datetime.strptime(str(x), fmt)
        return int(dt.strftime("%s%f")) * 1000
    return parse


def read_data(path):
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext.getOrCreate(sc)
    fc = ts.flint.FlintContext(sqlContext)

    data = (sqlContext.read.csv(path, header=True, inferSchema=True)
          .withColumn('time', date_parser('%Y-%m-%d %H:%M:%S')(col('date'))))
    df = fc.read.dataframe(data, is_sorted='False')

    return df


def linear_regression(df):
    model = df.summarize(ts.flint.summarizers.linear_regression('item_cnt_day', []))
    print("Linear regression R2: ", model.select('rSquared').collect())