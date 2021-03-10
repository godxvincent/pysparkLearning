from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer


def training(spark, resources_folder):
    trainingData = spark.read.format('libsvm').load(
    resources_folder + 'sample_linear_regression_data.txt')
    # el parametro featureCol es para indicarle a la función de regresión lineal que campo contiene los atributos o caracteristicas
    # que describen a la variable o label.
    # Se debe hacer una instalación de numpy en local, en caso de tener un cluster se debe instalar esa dependencia en cada worker.
    # pip3 install numpy
    linealRegresion = LinearRegression(
    featuresCol='features',
    labelCol='label',
     predictionCol='prediction')
    linealRegresionModel = linealRegresion.fit(trainingData)
    print("Coeficientes del modelo ")
    print(linealRegresionModel.coefficients)
    # El valor que toma Y cuando X vale cero.
    print("Intercepto del modelo ")
    print(linealRegresionModel.intercept)

    training_summary = linealRegresionModel.summary
    # El R2 indica en que porcentaje los valores de X pueden estar incidiendo
    # en el resultado de Y
    print("r2:")
    print(training_summary.r2)
    print("Root Mean Squared Error:")
    print(training_summary.rootMeanSquaredError)
    # trainingData.show()


def separando_datos(spark, resources_folder):
    # Se debe crear siempre dos dataframe
    all_data = spark.read.format('libsvm').load(
    resources_folder + 'sample_linear_regression_data.txt')
    training_data, test_data = all_data.randomSplit([0.7, 0.3])
    training_data.describe().show()
    test_data.describe().show()
    linealRegresion = LinearRegression(
    featuresCol='features',
    labelCol='label',
     predictionCol='prediction')
    correct_model = linealRegresion.fit(training_data)
    test_results = correct_model.evaluate(test_data)
    print("root mean squared error ")
    print(test_results.rootMeanSquaredError)

    unlabeled_data = test_data.select('features')
    unlabeled_data.show()
    predictions = correct_model.transform(unlabeled_data)
    predictions.show()


def realistic_example(spark, resources_folder):
    realistic_data = spark.read.csv(
    resources_folder +
    'Ecommerce_Customers.csv',
    inferSchema=True,
     header=True)
    realistic_data.printSchema()
    print(realistic_data.columns)
    # Aqui se convierte un archivo de datos al formato esperado por las funciones de regresión lineal.
    # Se entrega un listado de las columbnas que serviran de inputs
    assembler = VectorAssembler(
    inputCols=[
        'Avg Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership'],
         outputCol='features')
    output = assembler.transform(realistic_data)
    output.printSchema()
    final_data = output.select('features', 'Yearly Amount Spent')
    training_data, test_data = final_data.randomSplit([0.7, 0.3])
    training_data.describe().show()
    test_data.describe().show()

    lr = LinearRegression(
    labelCol='Yearly Amount Spent',
     featuresCol='features')
    lr_model = lr.fit(training_data)
    test_result = lr_model.evaluate(test_data)
    # residuals son la diferencia entre los valores reales y los valores
    # producidos.
    test_result.residuals.show()
    print("root mean squared error ")
    print(test_result.rootMeanSquaredError)
    print("r2:")
    print(test_result.r2)

    #
    unlabeled_data = test_data.select('features')

    unlabeled_data.show()
    predictions = lr_model.transform(unlabeled_data)
    predictions.show()


def consulting_project(spark, resources_folder):
    # StringIndexer
    realistic_data = spark.read.csv(
    resources_folder +
    'cruise_ship_info.csv',
    inferSchema=True,
     header=True)
    realistic_data.printSchema()
    realistic_data.show()
    string_indexer = StringIndexer(
    inputCol='Cruise_line',
     outputCol='Cruise_line_Index')
    output = string_indexer.fit(realistic_data)

    output.setHandleInvalid("error")
    df_string_indexed = output.transform(realistic_data)
    df_string_indexed.show()
    

    
    df_string_indexed.printSchema()
    # Se formatea los datos de entrada para generar un input para la función
    # de regresión líneal.
    assembler = VectorAssembler(
    inputCols=[
        'Age',
        'Tonnage',
        'passengers',
        'length',
        'cabins',
        'passenger_density',
        'Cruise_line_Index'],
         outputCol='features')
    output = assembler.transform(df_string_indexed)
    output.printSchema()
    # se crean los datos de entrenamiento y de prueba
    final_data = output.select('features', 'crew')
    training_data, test_data = final_data.randomSplit([0.7, 0.3])
    training_data.describe().show()
    test_data.describe().show()

    lr = LinearRegression(labelCol='crew', featuresCol='features', solver="normal")
    lr_model = lr.fit(training_data)
    print("Esto es el tipo del objeto lr_model"+str(lr_model))
    print(lr_model.coefficients)
    test_result = lr_model.evaluate(test_data)

    # print("coefficientStandardErrors")
    # print(test_result.coefficientStandardErrors)
    print("degreesOfFreedom")
    print(test_result.degreesOfFreedom)
    print("devianceResiduals")
    print(test_result.devianceResiduals)
    print("explainedVariance")
    print(test_result.explainedVariance)
    # print("featuresCol")
    # print(test_result.featuresCol)
    print("meanAbsoluteError")
    print(test_result.meanAbsoluteError)
    print("meanSquaredError")
    print(test_result.meanSquaredError)
    # print("tValues")
    # print(test_result.tValues)
    # print("pValues")
    # print(test_result.pValues)





    
    # residuals son la diferencia entre los valores reales y los valores
    test_result.residuals.show()
    print("root mean squared error ")
    print(test_result.rootMeanSquaredError)
    print("r2:")
    print(test_result.r2)



if __name__ == "__main__":
    resources_folder='../../resources/c_lineal_regression/'
    spark=SparkSession.builder.appName('LinealRegresion').getOrCreate()
    # training(spark, resources_folder)
    # separando_datos(spark, resources_folder)
    # realistic_example(spark, resources_folder)
    consulting_project(spark, resources_folder)
