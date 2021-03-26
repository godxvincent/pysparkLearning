from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler


def basic_example(spark, resources_folder):
    data = spark.read.format('libsvm').load(
        resources_folder + 'sample_kmeans_data.txt')
    data.printSchema()
    data.show()
    final_data = data.select(data['features'])
    kmeans = KMeans().setK(2).setSeed(1)
    model = kmeans.fit(final_data)
    print(type(model))

    # Withinn Sum Square Error
    # ClusteringEvaluator
    # computeCost is deprecated and now we have the values on summary
    wssse = model.summary
    print(type(wssse))
    wssse.predictions.show()
    print("Training Costs!!!!!")
    print(wssse.trainingCost) # esto era en remplazo de model.computeCost(final_data)
    print(model.clusterCenters())

    data = spark.read.format('libsvm').load(
        resources_folder + 'sample_kmeans_data.txt')
    data.printSchema()
    data.show()


def practice_exercise(spark, resources_folder):
    data = spark.read.csv(resources_folder + 'seeds_dataset.csv', header=True, inferSchema=True)
    data.printSchema()
    data.show()
    # assembler = VectorAssembler(inputCols=['area','perimeter', 'compactness','length_of_kernel','width_of_kernel','asymmetry_coefficient','length_of_groove'], outputCol='feature')
    assembler = VectorAssembler(inputCols=data.columns, outputCol='features')
    final_data = assembler.transform(data)

    scaler = StandardScaler(inputCol='features', outputCol='scalefeatures')
    scaler_model = scaler.fit(final_data)
    scaled_final_data = scaler_model.transform(final_data) # Estandariza los datos aparentemente usando distancia de desviación standard.

    kmeans = KMeans(featuresCol='scalefeatures', k=3)
    model = kmeans.fit(scaled_final_data)
    wssse = model.summary.trainingCost
    print(wssse)
    print(model.clusterCenters())
    model.summary.predictions.show()

def hacker_test(spark, resources_folder):
    data = spark.read.csv(resources_folder + 'hack_data.csv', header=True, inferSchema=True)
    data.printSchema()
    data.show()
    print(data.columns)
    assembler = VectorAssembler(inputCols=['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 'Pages_Corrupted', 'WPM_Typing_Speed'], outputCol='features')
    data_assembled = assembler.transform(data)
    data_assembled.show()

    scaler = StandardScaler(inputCol='features', outputCol='scaledfeatures')
    scaler_model = scaler.fit(data_assembled)
    data_assembled_scaled = scaler_model.transform(data_assembled)
    data_assembled_scaled.show()

    data_assembled = data_assembled_scaled.select('scaledfeatures').withColumn('features', data_assembled_scaled['scaledfeatures'])
    data_assembled.show()

    print("************************************* con tres cluster *************************************")
    kmeans3 = KMeans(featuresCol='features', k=3, seed=10)
    model3 = kmeans3.fit(data_assembled)
    wssse3 = model3.summary.trainingCost
    print(wssse3)
    print(model3.clusterCenters())
    model3.summary.predictions.show()

    predictions3 = model3.summary.predictions
    predictions3.groupBy('prediction').count().show()
    # predictions3.agg({'prediction': 'count'}).show()


    print("************************************* con dos cluster *************************************")
    kmeans2 = KMeans(featuresCol='features', k=2, seed=10)
    model2 = kmeans2.fit(data_assembled)
    wssse2 = model2.summary.trainingCost
    print(wssse2)
    print(model2.clusterCenters())
    model2.summary.predictions.show()

    predictions2 = model2.summary.predictions
    predictions2.groupBy('prediction').count().show()
    # model = kmeans.fit(final_data)

if __name__ == "__main__":
    resources_folder = '../../resources/f_clustering/'
    spark = SparkSession.builder.appName('clustering').getOrCreate()
    # basic_example(spark, resources_folder)
    # practice_exercise(spark, resources_folder)
    hacker_test(spark, resources_folder)

