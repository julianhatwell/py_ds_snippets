from pyspark import SparkContext
sc = SparkContext()

data = range(1,1000)
rdd = sc.parallelize(data)

rdd.collect()
rdd.take(2) # It will print first 2 elements of rdd

data = ['Hello' , 'I' , 'AM', 'Ankit ', 'Gupta']
Rdd = sc.parallelize(data)
Rdd1 = Rdd.map(lambda x: (x,1))
Rdd1.collect()
