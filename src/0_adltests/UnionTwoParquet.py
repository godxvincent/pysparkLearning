
from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, LongType

conf = SparkConf().setAppName('readParquet').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Trying add an schema on a parquet file does not work
parquetUnion = spark.read.parquet("tmp/output/testWriteDataParquet")
parquetUnion.printSchema()
parquetUnion.show()

parquetUnion2 = spark.read.parquet("tmp/output/testWriteDataParquet2")
parquetUnion2.printSchema()
parquetUnion2.show()
# parquetUnion2.

parquetSalida = parquetUnion.union(parquetUnion2)

parquetSalida.write.mode("append").parquet("tmp/output/testWriteDataParquetUnion")

