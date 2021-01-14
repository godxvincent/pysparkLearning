from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

spark = SparkSession.builder.appName('Basics').getOrCreate()

df = spark.read.json('people.json')
df.show()
df.printSchema()

print('These are the columns {0}'.format(df.columns))

# Muestra estadisticas básicas sobre el dataframe las columnas numericas.
df.describe().show()

# Primer parametro de structField es el nombre de la columna, el tipo y si el campo va a tener nulos o no
dataSchema = [StructField('age', IntegerType(), True),
              StructField('name', StringType(), True)]

finalStruct = StructType(fields=dataSchema)
df = spark.read.json('people.json', schema=finalStruct)
df.printSchema()