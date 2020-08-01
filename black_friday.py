from pyspark import SparkContext, SQLContext
sc = SparkContext()

sqlContext = SQLContext(sc)

train = sqlContext.read.load(source="com.databricks.spark.csv", path = 'PATH/train.csv', header = True,inferSchema = True)
test = sqlContext.read.load(source="com.databricks.spark.csv", path = 'PATH/test-comb.csv', header = True,inferSchema = True)


