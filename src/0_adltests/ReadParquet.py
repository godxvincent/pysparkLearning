
from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, LongType

conf = SparkConf().setAppName('readParquet').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Trying add an schema on a parquet file does not work
csvSchema = StructType().add("Nombre", StringType(), True).add("Edad",IntegerType(),True).add("Telefono",LongType(),True).add("anio_nacimiento",IntegerType(),True)
parquetUnion = spark.read.schema(csvSchema).parquet("tmp/output/testWriteDataParquet")
parquetUnion.printSchema()
parquetUnion.show()


