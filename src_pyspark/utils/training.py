import pandas as pd
import numpy as np
import pickle
import os

from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def prepare_df(df):
    """Prepare the df by creating feature and label columns used for training the data

    Args:
        df: spark training dataframe

    Returns:
        Processed spark df
    """

    df = df.withColumnRenamed("fraud_reported","label")
    features_cols = df.columns
    features_cols.remove("label")
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    df = assembler.transform(df)

    return df
    

def train_model_with_hyper_params_tuning(train, best_params_path):
    """Train the model and tune hyperparameters

    Args:
        train: training data
        best_params_path: path to save the best hyper parameters

    Returns:
        trained model
    """

    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

    pipeline = Pipeline(stages=[rf])
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20, 30])\
                                .addGrid(rf.impurity , ['gini', 'entropy'])\
                                .addGrid(rf.maxDepth, [3, 5, 7, 10]).build()

    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=5) 

    cvModel = crossval.fit(train)

    params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]
    best_params = params[np.argmax(cvModel.avgMetrics)]

    best_file = open(os.getcwd() + best_params_path + '/best_hyper_parameters.pkl', "wb")
    pickle.dump(best_params, best_file)
    best_file.close()

    return cvModel

def train_model(best_params_path, train):
    """Train model with existing hyper parameter values

    Args:
        best_params_path: Best hyperparamter file path
        train: training data

    Returns:
        trained model
    """

    best_hyper_params = pickle.load( open(os.getcwd() + best_params_path + '/best_hyper_parameters.pkl', "rb" ) )
    
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=best_hyper_params['numTrees'],
                                        impurity= best_hyper_params['impurity'], maxDepth=best_hyper_params['maxDepth'])
                                        
    rfModel = rf.fit(train)

    return rfModel


def train(spark, df, args):
    """Perform data processing, call the training methods and save the trained model

    Args:
        spark (_type_): _description_
        df (_type_): _description_
        args (_type_): _description_
    """

    df = prepare_df(df)

    train, test = df.randomSplit([0.8, 0.2], seed = 2018)
    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))

    if args.tune_hyper_params:
        model = train_model_with_hyper_params_tuning(train, args.best_hyper_params_filepath)
    else:
        model = train_model(args.best_hyper_params_filepath, train)

    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc =  evaluator.evaluate(predictions)
    print("AUC = %s" % (auc))

    model.write().overwrite().save(os.getcwd() + args.model_path + '/rf')

