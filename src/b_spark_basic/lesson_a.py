from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType
from pyspark.sql.functions import countDistinct, abs,  stddev, format_number


def readingDataFrames():


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

    # Esto retorna un objeto de tipo columna
    print(df['age'])
    # Aqui se aprecia mejor lo anterior.
    print(type(df['age']))

    # Para seleccionar una columna
    df.select('age').show()

    # Para recuperar las dos primeras filas de un dataframe (Arreglo de filas)
    print(df.head(2))

    # Primera fila del dataframe
    print(df.head(2)[0])

    # Typo de dato row
    print(type(df.head(2)[0]))

    # Seleccionar varias columnas arreglo con el nombre de las columnas
    df.select(['age','name']).show()

    # adicionar una nueva columna ==> Esto no sobre escribe el objeto df.
    df.withColumn('double_age', df['age']*2).show()

    # Renombrar una columba ==> Esto no sobre escribe el objeto df.
    df.withColumnRenamed('age', 'new_age_test').show()

    # se puede crear una vista temporal (Tabla) de un dataframe para trabajar SQL sobre el dataframe
    df.createOrReplaceTempView('peopleTable')

    result = spark.sql('SELECT * FROM peopleTable')
    result2 = spark.sql('SELECT * FROM peopleTable WHERE age = 30')
    print('Resultado 1')
    result.show()
    print('Resultado 2')
    result2.show()


def basicOperations(spark):
    df = spark.read.csv('appl_stock.csv', sep=',', header=True, inferSchema=True)
    # spark.sparkContext.setLogLevel("OFF")
    df.printSchema()
    df.show()

    # Forma similar a como se haría en SQL
    df.filter('Close < 500').select(['Open', 'Close']).show()

    # Forma similar a como se haría en SQL
    df.filter(df['Close'] < 500).select(['Open', 'Close']).show()

    # multiples filtros, cada condicion entre parentesis el and es & el or es | y la negación ~
    df.filter((df['Close'] > 200) & (df['Open'] > 200)).show()

    # Filtrar un dato en particular, para trabajar con el dato y no sacarlo por consola es con collect
    result = df.filter(df['Close'] == 197.75).collect()

    # En este caso entrega un arreglo de filas, que para el caso será una.
    row = result[0]

    # Estp devuelve un dictionario
    print(row.asDict())

    # Estp devuelve un dictionario
    print(row.asDict()['Volume'])

    df.groupBy('Date').mean().show()

def groupByAndAggregateOperations(spark):
    df = spark.read.csv('sales_info.csv', header=True, inferSchema=True, sep=',')
    df.show()
    print(df.groupBy('Company'))
    # Si no se le especifica el nombre de la columna entrega el agregado de todas las columnas
    df.groupBy('Company').mean().show()

    # si solo se quiere agrupar pero sin especificar columnas
    df.agg({'Sales' :'max'}).show()

    df.select(countDistinct('Company')).show()
    df.select(stddev('Sales').alias('std')).select(format_number('std',2).alias('std final')).show()

    # Ordenando dataframes
    df.orderBy('Sales').show()

    df.orderBy(['Company', 'Person'], ascending=[0, 1]).show()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('Basics').getOrCreate()
    # readingDataFrames(spark)
    # basicOperations(spark)
    groupByAndAggregateOperations(spark)


