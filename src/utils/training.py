

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator




def train(df):

    df = df.withColumnRenamed("fraud_reported","label")
    features_cols = df.columns
    features_cols.remove("label")
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    df = assembler.transform(df)

    train, test = df.randomSplit([0.8, 0.2], seed = 2018)
    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))
    
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    accuracy =  evaluator.evaluate(rfModel.transform(test))
    print("Accuracy = %s" % (accuracy))
