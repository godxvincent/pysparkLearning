from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, StructType, DoubleType
from pyspark.sql.functions import (
    corr,
    min,
    max,
    countDistinct,
    abs,
    stddev,
    format_number,
    mean,
    dayofmonth,
    hour,
    dayofyear,
    month,
    year,
    weekofyear,
    format_number,
    date_format,
    Column)

def test(*args):
    suma = 0
    for arg in args:
        suma = suma + arg
    print(suma)


def filter(**kwargs):
    query = "SELECT * FROM clientes"
    i = 0
    for key, value in kwargs.items():
        if i == 0:
            query += " WHERE "
        else:
            query += " AND "
        query += "{}='{}'".format(key, value)
        i += 1
    query += ";"
    return query


def reading_dataframes(path, spark):
    df = spark.read.json(path + 'people.json')
    df.show()
    df.printSchema()

    print('These are the columns {0}'.format(df.columns))

    # Muestra estadisticas básicas sobre el dataframe las columnas numericas.
    df.describe().show()

    # Primer parametro de structField es el nombre de la columna, el tipo y si
    # el campo va a tener nulos o no
    data_schema = [StructField('age', IntegerType(), True),
                   StructField('name', StringType(), True)]

    final_struct = StructType(fields=data_schema)
    df = spark.read.json(path + 'people.json', schema=final_struct)
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
    df.select(['age', 'name']).show()

    # adicionar una nueva columna ==> Esto no sobre escribe el objeto df.
    df.withColumn('double_age', df['age'] * 2).show()

    # Renombrar una columba ==> Esto no sobre escribe el objeto df.
    df.withColumnRenamed('age', 'new_age_test').show()

    # se puede crear una vista temporal (Tabla) de un dataframe para trabajar
    # SQL sobre el dataframe
    df.createOrReplaceTempView('peopleTable')

    result = spark.sql('SELECT * FROM peopleTable')
    result2 = spark.sql('SELECT * FROM peopleTable WHERE age = 30')
    print('Resultado 1')
    result.show()
    print('Resultado 2')
    result2.show()


def basic_operations(path, spark):
    df = spark.read.csv(
        path + 'appl_stock.csv',
        sep=',',
        header=True,
        inferSchema=True)
    # spark.sparkContext.setLogLevel("OFF")
    df.printSchema()
    df.show()

    # Forma similar a como se haría en SQL
    df.filter('Close < 500').select(['Open', 'Close']).show()

    # Forma similar a como se haría en SQL
    df.filter(df['Close'] < 500).select(['Open', 'Close']).show()

    # multiples filtros, cada condicion entre parentesis el and es & el or es
    # | y la negación ~
    df.filter((df['Close'] > 200) & (df['Open'] > 200)).show()

    # Filtrar un dato en particular, para trabajar con el dato y no sacarlo
    # por consola es con collect
    result = df.filter(df['Close'] == 197.75).collect()

    # En este caso entrega un arreglo de filas, que para el caso será una.
    row = result[0]

    # Estp devuelve un dictionario
    print(row.asDict())

    # Estp devuelve un dictionario
    print(row.asDict()['Volume'])

    df.groupBy('Date').mean().show()


def groupby_and_aggregate_operations(path, spark):
    df = spark.read.csv(
        path + 'sales_info.csv',
        header=True,
        inferSchema=True,
        sep=',')
    df.show()
    print(df.groupBy('Company'))
    # Si no se le especifica el nombre de la columna entrega el agregado de
    # todas las columnas
    df.groupBy('Company').mean().show()

    # si solo se quiere agrupar pero sin especificar columnas
    df.agg({'Sales': 'max'}).show()

    df.select(countDistinct('Company')).show()
    df.select(
        stddev('Sales').alias('std')).select(
        format_number(
            'std',
            2).alias('std final')).show()

    # Ordenando dataframes
    df.orderBy('Sales').show()

    df.orderBy(df['Sales'].desc())

    df.orderBy(['Company', 'Person'], ascending=[0, 1]).show()


def working_with_missing_data(path, spark):
    df = spark.read.csv(
        path +
        'ContainsNull.csv',
        header=True,
        inferSchema=True,
        sep=',')
    # Para eliminar filas con nulos uno puede utilizar el atributo na y la
    # funcion drop.
    df.na.drop().show()

    # Para especificar cuantas columnas deben ser nulas para quitar los nulos,
    # se define un umbral (threshold)
    df.na.drop(thresh=2).drop().show()

    # Otro parametro dice como debe hacerse el drop, por default el valor esta en any, lo que significa que cualquier
    # campo que tenga null debe ser descartado, pero puede decirse también el
    # valor de all
    df.na.drop(how='any').show()
    df.na.drop(how='all').show()

    # Se puede especificar un subconjunto de columnas para que sean evaluadas.
    df.na.drop(subset=['Sales']).show()

    # Ahora para completar los datos es con el metodo fill y es capaz de determinar a que columnas aplicarles el valor
    # de acuerdo al tipo de dato que tenga la columna.
    df.printSchema()

    df.na.fill('FILL VALUE').show()
    df.na.fill(0).show()
    df.na.fill('0', subset=['Name'])

    # Retorna un arreglo de rows.
    mean_value = df.select(mean('Sales')).collect()
    # Re uso la variable primer indice corresponde a la primera posición de la
    # lista y el segundo al numero de row.
    mean_value = mean_value[0][0]
    df.na.fill(mean_value, subset=['Sales']).show()

    print('End of method')


def working_with_dates_and_timestamps(path, spark):
    df = spark.read.csv(
        path + 'appl_stock.csv',
        sep=',',
        header=True,
        inferSchema=True)
    df.select(dayofmonth(df['Date']), df['Date']).show()
    df.select(hour(df['Date']), df['Date']).show()
    df.select(month(df['Date']), df['Date']).show()
    df.select(year(df['Date']), df['Date']).show()
    result = df.groupBy(year(df['Date']).alias(
        'anio_agg')).agg({'Close': 'mean'})
    result.select(
        result['anio_agg'].alias('año'),
        format_number(result['avg(Close)'], 3).
        alias('Promedio Cierre')).orderBy('año').show()


def basic_practice_exercise(path, spark):
    # Use the walmart_stock.csv file to Answer and complete the  tasks below!
    # Start a simple Spark Session

    # Load the Walmart Stock CSV File, have Spark infer the data types.
    df = spark.read.csv(
        path + 'walmart_stock.csv',
        sep=',',
        header=True,
        inferSchema=True)

    # What are the column names?
    print(df.columns)

    # What does the Schema look like?
    df.printSchema()

    # Print out the first 5 columns.
    print(df.head(5))

    # Use describe() to learn about the DataFrame.
    df.describe().show()

    # Bonus Question!
    # There are too many decimal places for mean and stddev in the describe() dataframe. Format the numbers to just
    # show up to two decimal places. Pay careful attention to the datatypes that .describe() returns, we didn't cover
    # how to do this exact formatting, but we covered something very similar. [Check this link for a
    # hint] (http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.cast)
    # If you get stuck on this, don 't worry, just view the solutions.
    new_df = df.describe()
    new_df.printSchema()
    new_df.select(
        new_df['Date'], format_number(
            new_df['Open'].cast(
                DoubleType()), 2).alias('Open'), format_number(
            new_df['High'].cast(
                DoubleType()), 2).alias('High'), format_number(
            new_df['Close'].cast(
                DoubleType()), 2).alias('Close'), format_number(
            new_df['Volume'].cast(
                IntegerType()), 2).alias('Volume'), format_number(
            new_df['Adj Close'].cast(
                DoubleType()), 2).alias('Adj Close')).show()

    # Create a new dataframe with a column called HV Ratio that is the ratio
    # of the High Price versus volume of stock traded for a day.
    df2 = df.withColumn('HV Ratio', df['High'] / df['Volume'])
    df2.show()

    # What day had the Peak High in Price?
    max_price = df2.agg({'High': 'max'}).collect()[0][0]
    df2.filter((df2['High'] == max_price)).select(df2['Date']).show()

    # What is the mean of the Close column?
    df2.agg({'Close': 'mean'}).show()
    df2.groupBy().mean('Close').show()

    # What is the max and min of the Volume column?
    print([df2['Volume'], df2['Close']])
    # print(isinstance(df2['Volume'], Column))
    df2.agg(min(df2['Volume']), max(df2['Volume'])).show()

    #### How many days was the Close lower than 60 dollars?
    df2.filter((df2.Close < 60)).groupBy().count().show()

    #### What percentage of the time was the High greater than 80 dollars ?
    #### In other words, (Number of Days High>80)/(Total Days in the dataset)
    # print(df2.filter(df2['High'] > 80).count())
    print("Percentage", (df2.filter(df2['High'] > 80).count() / df2.count()) * 100 )

    #### What is the Pearson correlation between High and Volume?
    #### [Hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions.corr)
    df2.select(corr(df2['High'], df2['Volume'])).show()


    #### What is the max High per year?
    df2.groupBy(year(df2['Date']).alias('Year')).max('High').orderBy('Year').show()

    #### What is the average Close for each Calendar Month?
    #### In other words, across all the years, what is the average Close price for Jan,Feb, Mar, etc...
    #### Your result will have a value for each of these months.
    df2.groupBy(month(df2['Date']).alias('Months')).mean('Close').orderBy('Months').show()





if __name__ == "__main__":
    resources_folder = '../../resources/b_sparks_basic/'
    spark = SparkSession.builder.appName('Basics').getOrCreate()
    # reading_dataframes(resources_folder, spark)
    # basic_operations(resources_folder, spark)
    # groupby_and_aggregate_operations(resources_folder, spark)
    # working_with_missing_data(resources_folder, spark)
    # working_with_dates_and_timestamps(resources_folder, spark)
    basic_practice_exercise(resources_folder, spark)
