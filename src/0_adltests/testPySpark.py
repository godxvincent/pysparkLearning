
from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, LongType

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

csvSchema = StructType().add("Nombre", StringType(), True).add("Edad",IntegerType(),True).add("Telefono",LongType(),True).add("anio_nacimiento",IntegerType(),True)
csvTest = spark.read.options(delimiter=";", header = True).schema(csvSchema).csv("testReadData.csv")
csvTest.printSchema()
csvTest.show()

csvTest2 = spark.read.options(delimiter=";", header = True).csv("testReadData2.csv")
csvTest2.printSchema()
csvTest2.show()

unionDf = csvTest.union(csvTest2)
unionDf.printSchema()
unionDf.show()

unionDf.write.options(header='True', delimiter=',').mode("append").csv("tmp/output/testWriteData")

unionDf.write.mode("append").parquet("tmp/output/testWriteDataParquet")

csvTest3 = spark.read.options(delimiter=";", header = True).schema(csvSchema).csv("testReadData2.csv")
csvTest3.printSchema()
csvTest3.show()

csvTest3.write.mode("append").parquet("tmp/output/testWriteDataParquet2")
