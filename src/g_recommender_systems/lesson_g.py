from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def basic_example(spark, resources_folder):
    data = spark.read.csv(
        resources_folder + 'movielens_ratings.csv', header=True, inferSchema=True)
    data.printSchema()
    data.show()

    train_data, test_data = data.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')
    model = als.fit(train_data)
    predictions = model.transform(test_data)
    predictions.show()

    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rsme = evaluator.evaluate(predictions)
    print(rsme)

    test_data.show()
    # Ahora como actua en un solo user
    single_user = test_data.filter(test_data['userId']==13).select(['movieId','userId'])
    single_user.show()

    recommendations = model.transform(single_user)
    recommendations.orderBy('prediction', ascending=False).show()




if __name__ == "__main__":
    resources_folder = '../../resources/g_recommender_systems/'
    spark = SparkSession.builder.appName('recommender_systems').getOrCreate()
    basic_example(spark, resources_folder)

