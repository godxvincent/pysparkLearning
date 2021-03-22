from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline


def training(spark, resources_folder):
    df = spark.read.format('libsvm').load(
        resources_folder + 'sample_libsvm_data.txt')
    df.show()
    # Esto es una instancia de un modelo logistico
    my_log_reg_model = LogisticRegression()
    # Se entrena el modelo
    fitted_log_reg = my_log_reg_model.fit(df)
    log_summary = fitted_log_reg.summary
    print('clase del objeto log_summary')
    print(log_summary)
    print('clase log_summary.predictions ')
    # Esto log_summary.predictions es un dataframe
    print(log_summary.predictions)
    log_summary.predictions.printSchema()
    # Estos son los campos del dataframe
    # root
    # | -- label: double(nullable=true)           variable a predecir o correct label
    # | -- features: vector(nullable=true)        variables input del modelo
    # | -- rawPrediction: vector(nullable=true)   predicción en crudo
    # | -- probability: vector(nullable=true)     probabilidad para ese valor predicho
    # | -- prediction: double(nullable=false)     predicción
    log_summary.predictions.show()

    # Validación de que son los evaluators
    training_df, test_df = df.randomSplit([0.7, 0.3])
    logistic_regression_instance = LogisticRegression()
    fitted_trained_log_reg = logistic_regression_instance.fit(training_df)
    predictions_and_labels = fitted_trained_log_reg.evaluate(test_df)
    # muestra las predicciones para los datos de prueba
    predictions_and_labels.predictions.show()

    my_evaluator = BinaryClassificationEvaluator()
    my_final_roc = my_evaluator.evaluate(predictions_and_labels.predictions)

    print(my_final_roc)


def titanic_example(spark, resources_folder):
    df = spark.read.csv(
        resources_folder +
        'titanic.csv',
        inferSchema=True,
        header=True)
    df.printSchema()
    # SibSp Siblins or spouses
    # Parch Parent Childrend
    # fare lo que se pago por los tickets
    # Cabina
    # Embarked
    df.show()
    my_data = df.select(['Survived', 'Pclass', 'Sex', 'Age',
                         'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])
    # Esto elimina los registros que tuvieran algun null en sus columnas
    my_data_fixed = my_data.na.drop()

    # Creo todos los pasos que le voy a aplicar a mi dataframe
    gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndexed')
    # A B C Suponga que estos son valores de la columna Sex
    # 0 1 2 Con String Indexer quedan así
    # Se aplica One Hot Encoder
    # Se crea un arreglo de tantas posiciones como valores distintos hayan
    # y se rellena con 1 o 0 dependiendo del indice.
    # Suponga A y dado los indices anteriores se crea un arreglo así => [1 , 0
    # , 0]

    gender_encoder = OneHotEncoder(inputCol='SexIndexed', outputCol='SexVec')

    embark_indexer = StringIndexer(
        inputCol='Embarked',
        outputCol='EmbarkedIndexed')
    embark_encoder = OneHotEncoder(
        inputCol='EmbarkedIndexed',
        outputCol='EmbarkVec')

    assembler = VectorAssembler(
        inputCols=[
            'Pclass',
            'SexVec',
            'EmbarkVec',
            'Age',
            'SibSp',
            'Parch',
            'Fare'],
        outputCol='features')

    log_reg_titanic = LogisticRegression(
        featuresCol='features', labelCol='Survived')
    pipeline = Pipeline(
        stages=[
            gender_indexer,
            gender_encoder,
            embark_indexer,
            embark_encoder,
            assembler,
            log_reg_titanic])
    train_data, test_data = my_data_fixed.randomSplit([0.7, 0.3])
    fitted_model = pipeline.fit(train_data)
    results = fitted_model.transform(test_data)
    results.printSchema()
    results.show()
    #
    my_eval_object = BinaryClassificationEvaluator(
        rawPredictionCol='prediction', labelCol='Survived')

    results.select(['Survived', 'prediction']).show()
    # Area Under the Curve
    # https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it
    AUC = my_eval_object.evaluate(results)
    print(AUC)
    # 0.6629711751662971


def one_hot_indexer_example():
    # https://spark.apache.org/docs/latest/mllib-data-types.html#local-vector
    # https://stackoverflow.com/questions/42295001/how-to-interpret-results-of-spark-onehotencoder#42297197
    df = spark.createDataFrame([
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (3.0, 1.0),
        (2.0, 0.0)
    ], ["categoryIndex1", "categoryIndex2"])

    encoder = OneHotEncoder(
        inputCols=[
            "categoryIndex1",
            "categoryIndex2"],
        outputCols=[
            "categoryVec1",
            "categoryVec2"],
        dropLast=False)
    model = encoder.fit(df)
    encoded = model.transform(df)
    encoded.show()


def consulting_project_by_me(spark, resources_folder):
    df = spark.read.csv(
        resources_folder +
        'customer_churn.csv',
        inferSchema=True,
        header=True)
    df.printSchema()
    df.show()
    # 1. Seleccionar los datos del modelo
    my_data = df.select(['Age', 'Total_Purchase', 'Years', 'Num_Sites',
                         'Location', 'Company', 'Churn'])
    df.groupBy('Company').count().show()
    df.groupBy('Location').count().show()
    # ¡Creo que la fecha Onboard_date podría servir!
    # 2. Elimino los datos que tengan algun dato en nulo.
    my_data_fixed = my_data.na.drop()

    # Creo todos los pasos que le voy a aplicar a mi dataframe
    # 3. Creo una codificacion para las distintas compañias
    assembler = VectorAssembler(
        inputCols=[
            'Age', 'Total_Purchase', 'Years', 'Num_Sites'],
        outputCol='features')

    log_reg_customer_churn = LogisticRegression(
        featuresCol='features', labelCol='Churn')
    pipeline = Pipeline(
        stages=[assembler, log_reg_customer_churn])
    train_data, test_data = my_data_fixed.randomSplit([0.7, 0.3])

    fitted_model = pipeline.fit(train_data)
    results = fitted_model.transform(test_data)
    results.printSchema()
    results.show()
    # #
    my_eval_object = BinaryClassificationEvaluator(
        rawPredictionCol='prediction', labelCol='Churn')

    results.select(['Churn', 'prediction']).show()
    # Area Under the Curve
    # https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it
    AUC = my_eval_object.evaluate(results)
    print(AUC)


def consulting_project_by_tutor(spark, resources_folder):
    df = spark.read.csv(
        resources_folder +
        'customer_churn.csv',
        inferSchema=True,
        header=True)
    df.printSchema()
    df.show()
    # Recordar que VectorAssembler es una clase para adecuar los datos para la
    # función de regresión logistica.
    assembler = VectorAssembler(
        inputCols=[
            'Age', 'Total_Purchase', 'Years', 'Num_Sites'],
        outputCol='features')
    # Se ejecuta el metodo transform para que se aplique el VectorAssembler
    # sobre los datos
    my_data_transformed = assembler.transform(df)
    my_data_transformed.show()
    # Se separa los datos entre entranamiento y test.
    training_data, test_data = my_data_transformed.randomSplit([0.7, 0.3])
    training_data.describe().show()
    test_data.describe().show()
    # Esto crea una instancia de un objeto que analiza los datos aplicado un
    # modelo lógistico
    log_reg_customer_churn = LogisticRegression(
        featuresCol='features', labelCol='Churn')
    print(log_reg_customer_churn)

    # Entrego los datos para que genere el objeto sea entrenado.
    fitted_model_customer_churn = log_reg_customer_churn.fit(training_data)
    # Objecto de clase LogisticRegressionModel
    print(fitted_model_customer_churn)
    print("A continuación imprimo los coeficientes {a}".format(
        a=fitted_model_customer_churn.coefficients))
    print("A continuación imprimo una matriz {a} \n".format(
        a=fitted_model_customer_churn.coefficientMatrix))
    print("A continuación imprimo el resumen del modelo accuracy {a} \n".format(
        a=fitted_model_customer_churn.summary.accuracy))
    print("A continuación imprimo el resumen del modelo AUROC {a} \n".format(
        a=fitted_model_customer_churn.summary.areaUnderROC))
    # Muestra el recall y la precisón por cada dato sobre los datos entrenados
    # Summary es un atributo de tipo BinaryLogisticRegressionTrainingSummary
    fitted_model_customer_churn.summary.pr.show()
    # Muestra las prediciones sobre cada dato
    # Predictions y Labels.
    fitted_model_customer_churn.summary.predictions.show()
    fitted_model_customer_churn.summary.predictions.describe().show()
    # Muestra los valores de TPR (True Positive Rate) y FPR (False Positive
    # Rate)
    fitted_model_customer_churn.summary.roc.show()

    # Evaluo el objeto contra los datos de prueba
    results_agains_test_data = fitted_model_customer_churn.evaluate(test_data)
    # Objeto de clase BinaryLogisticRegressionSummary
    print(results_agains_test_data)
    print("*********************+ transform data - begin *********************+")
    results_agains_test_data_transform = fitted_model_customer_churn.transform(
        test_data)
    print(results_agains_test_data_transform)
    results_agains_test_data_transform.show()
    print("*********************+ transform data - end *********************+")
    # Este objeto ya solo contiene los resultados sobre los datos de test.
    results_agains_test_data.predictions.describe().show()
    results_agains_test_data.predictions.show()
    results_agains_test_data.predictions.describe().show()
    # Area bajo la curva devuelto por la función!
    print(results_agains_test_data.areaUnderROC)
    print(
        "Este es el valor del area bajo la curva del objeto de test ==> {a}".format(
            a=results_agains_test_data.areaUnderROC))

    # Para medio entender entre el rawPrediction y prediction
    # https://stackoverflow.com/questions/37903288/what-do-columns-rawprediction-and-probability-of-dataframe-mean-in-spark-mll#37909854

    # Se evalua el resultado del modelo sobre los datos de test.
    print("*********************** Se hace la evaluación usando la clase BinaryClassificationEvaluator ***********************")
    my_eval_object = BinaryClassificationEvaluator(
        rawPredictionCol='prediction', labelCol='Churn')
    AUC = my_eval_object.evaluate(results_agains_test_data.predictions)
    print(
        "Este es el valor del area bajo la curva de usar la clase ** BinaryClassificationEvaluator ** ==> {a} ".format(a=AUC))
    my_eval_object2 = BinaryClassificationEvaluator(
        rawPredictionCol='rawPrediction', labelCol='Churn')
    AUC = my_eval_object2.evaluate(results_agains_test_data.predictions)
    print(
        "Este es el valor del area bajo la curva de usar la clase ** BinaryClassificationEvaluator with rawPrediction ** ==> {a} ".format(a=AUC))

    df = spark.read.csv(
        resources_folder +
        'new_customers.csv',
        inferSchema=True,
        header=True)
    my_new_data_transformed = assembler.transform(df)
    my_new_data_transformed.show()

    # results_agains_test_data = fitted_model_customer_churn.evaluate(my_new_data_transformed)
    results_agains_test_data = fitted_model_customer_churn.transform(my_new_data_transformed)
    results_agains_test_data.show()




if __name__ == "__main__":
    resources_folder = '../../resources/d_logistic_regression/'
    spark = SparkSession.builder.appName('LogisticRegresion').getOrCreate()
    # training(spark, resources_folder)
    # titanic_example(spark, resources_folder)
    # one_hot_indexer_example()
    # consulting_project_by_me(spark, resources_folder)
    consulting_project_by_tutor(spark, resources_folder)
