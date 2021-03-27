from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF, CountVectorizer, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, udf, length
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



def functions_for_deal_with_texts(spark, resources_folder):
    send_df = spark.createDataFrame([
        (0, 'Hi I heard about Spark'),
        (1, 'I wish java could use case classes'),
        (2, 'Logistic,regression,models,are,neat'),
    ], ['id', 'sentence'])

    tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
    regularTokenizer = RegexTokenizer(
        inputCol='sentence',
        outputCol='words',
        pattern='\\W')
    count_token = udf(lambda words: len(words), IntegerType())
    tokenize = tokenizer.transform(send_df)
    tokenize.show()
    tokenize.withColumn('tokens', count_token(col('words'))).show()

    rg_tokenize = regularTokenizer.transform(send_df)
    rg_tokenize.show()
    rg_tokenize.withColumn('tokens', count_token(col('words'))).show()

    # remover palabras comunes
    sentenceData = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])

    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
    remover.transform(sentenceData).show(truncate=False)

    wordDataFrame = spark.createDataFrame([
        (0, ["Hi", "I", "heard", "about", "Spark"]),
        (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
        (2, ["Logistic", "regression", "models", "are", "neat"])
    ], ["id", "words"])

    ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

    ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.select("ngrams").show(truncate=False)


def functions_for_deal_with_texts_2(spark, resources_folder):
    send_df = spark.createDataFrame([
        (0, 'Hi I heard about Spark'),
        (1, 'I wish java could use case classes'),
        (2, 'Logistic regression models are neat'),
    ], ['label', 'sentence'])

    tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
    words_data = tokenizer.transform(send_df)
    words_data.show(truncate=False)
    hashing_tf = HashingTF(inputCol='words', outputCol='rawFeatures')
    featurized_data = hashing_tf.transform(words_data)
    print("show featurized_data")
    featurized_data.show(truncate=False)
    # Inverse Document Frecuency
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idf_model = idf.fit(featurized_data)
    print(idf_model)
    rescaled_data = idf_model.transform(featurized_data)
    rescaled_data.select('label', 'features').show(truncate=False)


def functions_for_deal_with_texts_3(spark, resources_folder):
    df = spark.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" "))
    ], ["id", "words"])
    df.show()
    cv = CountVectorizer(
        inputCol='words',
        outputCol='features',
        vocabSize=3,
        minDF=2.0)
    model = cv.fit(df)
    result = model.transform(df)
    result.show(truncate=False)

def filter_detections(spark, resources_folder):
    messages = spark.read.csv(resources_folder+'SMSSpamCollection', inferSchema=True, sep='\t')
    messages.printSchema()
    messages.show()
    messages = messages.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1','text')
    messages.show()

    messages = messages.withColumn('length', length(messages['text']))
    messages.show()
    messages.groupBy('class').mean().show()
    tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
    stop_remover = StopWordsRemover(inputCol='token_text', outputCol='stop_tokens')
    count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='count_vec')
    # idf = inverse document frecuency
    # td = term frequency
    idf = IDF(inputCol='count_vec', outputCol='tf_idf')
    ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')

    assembler = VectorAssembler( inputCols=['tf_idf', 'length' ], outputCol='features')
    nb = NaiveBayes()

    data_pre_pipeline = Pipeline(stages=[
        ham_spam_to_numeric, tokenizer, stop_remover, count_vec, idf, assembler
    ])

    clean_data = data_pre_pipeline.fit(messages).transform(messages)
    clean_data.show()
    clean_data = clean_data.select('label', 'features')
    training_messages, test_messages = clean_data.randomSplit([0.7, 0.3])
    spam_detector = nb.fit(training_messages)
    test_results = spam_detector.transform(test_messages)
    test_results.show()


    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate( test_results )
    print("ACC of NB Model")
    print(acc)


if __name__ == "__main__":
    resources_folder = '../../resources/h_nlp/'
    spark = SparkSession.builder.appName('nlp').getOrCreate()
    # functions_for_deal_with_texts(spark, resources_folder)
    # functions_for_deal_with_texts_2(spark, resources_folder)
    # functions_for_deal_with_texts_3(spark, resources_folder)
    filter_detections(spark, resources_folder)
