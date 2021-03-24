from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline


def basic_example(spark, resources_folder):
    data = spark.read.format('libsvm').load(
        resources_folder + 'sample_libsvm_data.txt')
    data.printSchema()
    data.show()
    train_data, test_data = data.randomSplit([0.6, 0.4])
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    gbtc = GBTClassifier()

    dtc_model = dtc.fit(train_data)
    rfc_model = rfc.fit(train_data)
    gbtc_model = gbtc.fit(train_data)

    dtc_predictions = dtc_model.transform(test_data)
    rfc_predictions = rfc_model.transform(test_data)
    gbtc_predictions = gbtc_model.transform(test_data)

    dtc_predictions.show()
    rfc_predictions.show()
    # GBT No tiene rawPrediction Column, si esta haciendo un predictor de clasificacion binaria o multiclasificacion
    # puede que pida el rawPrediction como un input
    gbtc_predictions.show()

    acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')
    print("DTC Accuracy")
    print(acc_eval.evaluate(dtc_predictions))
    print("RFC Accuracy")
    print(acc_eval.evaluate(rfc_predictions))
    print("GBTC Accuracy")
    print(acc_eval.evaluate(gbtc_predictions))

    print(rfc_model.featureImportances)


def universities_example(spark, resources_folder):
    data = spark.read.csv(
        resources_folder + 'College.csv', header=True, inferSchema=True)
    data.printSchema()
    data.show()

    assembler = VectorAssembler(
        inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate'
                                                                                                       'Room_Board',
                   'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni',
                   'Expend', 'Grad_Rate'],
        outputCol='features')
    data_assembled = assembler.transform(data)
    private_state_indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')
    data_transformed = private_state_indexer.fit(data_assembled).transform(data_assembled)

    train_data, test_data = data_transformed.select(['features', 'PrivateIndex']).randomSplit([0.6, 0.4])

    dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
    rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol='features')
    gbtc = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')

    dtc_college_model = dtc.fit(train_data)
    rfc_college_model = rfc.fit(train_data)
    gbtc_college_model = gbtc.fit(train_data)

    dtc_predictions = dtc_college_model.transform(test_data)
    rfc_predictions = rfc_college_model.transform(test_data)
    gbtc_predictions = gbtc_college_model.transform(test_data)

    my_binary_evaluator = BinaryClassificationEvaluator(labelCol='PrivateIndex')
    print("DTC Evaluator")
    print(my_binary_evaluator.evaluate(dtc_predictions))
    print("RFC Evaluator")
    print(my_binary_evaluator.evaluate(rfc_predictions))
    print("DTC Evaluator")
    my_binary_evaluator = BinaryClassificationEvaluator(labelCol='PrivateIndex', rawPredictionCol='prediction')
    print(my_binary_evaluator.evaluate(gbtc_predictions))

    # No se puede hacer una evaluación del accuracy con un BinaryClassificationEvaluator para eso toca usar un
    # MulticlassClassificationEvaluator
    acc_eval = MulticlassClassificationEvaluator(labelCol='PrivateIndex', metricName='accuracy')
    rfc_accuracy = acc_eval.evaluate(rfc_predictions)
    print(rfc_accuracy)


def consulting_project(spark, resources_folder):
    data = spark.read.csv(
        resources_folder + 'dog_food.csv', header=True, inferSchema=True)
    data.printSchema()
    data.show()
    data.describe().show()
    # data.filter((data['Spoiled']==0)).show()
    assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'], outputCol='features')
    data_prepared = assembler.transform(data)
    rfc = RandomForestClassifier(labelCol='Spoiled', featuresCol='features')
    rfc_model = rfc.fit(data_prepared)
    print(rfc_model)
    rfc_model_pred = rfc_model.transform(data_prepared)

    print("Predicciones del modelo")
    print(rfc_model_pred)
    rfc_model_pred.show()

    print("Evaluación del modelo")
    my_binary_evaluator = BinaryClassificationEvaluator(labelCol='Spoiled')
    print(my_binary_evaluator.evaluate(rfc_model_pred))

    print("featureImportances")
    print(rfc_model.featureImportances)
    print(type(rfc_model.featureImportances))
 

if __name__ == "__main__":
    resources_folder = '../../resources/e_tree_methods/'
    spark = SparkSession.builder.appName('myTree').getOrCreate()
    # basic_example(spark, resources_folder)
    # universities_example(spark, resources_folder)
    consulting_project(spark, resources_folder)
