import os
import pathlib
import argparse
import json
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

from utils.utils import data_preprocessing
import requests

from flask import Flask, render_template, request

app = Flask(__name__)

spark = SparkSession.builder.appName("Zeiss_predict").getOrCreate()
args = None

@app.route('/')
def index():
    return render_template('/index.html')
 

@app.route('/predict-fraud', methods = ['GET'])
def prediction_product(data=None):

    args = parse_args()

    data_json = request.args.get('data')
    
    a_json = json.loads(data_json)
    df_pandas = pd.DataFrame.from_dict(a_json)
    df_spark = spark.createDataFrame(df_pandas)

    df_processed = data_preprocessing(df_spark, args)
    
    features_cols = df.columns
    assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
    df = assembler.transform(df)

    rfModel = RandomForestClassificationModel.load(os.getcwd() + args.model_path + '/rf')
    predictionsDF = rfModel.transform(df)

    predictionsDF_pandas = predictionsDF.select("prediction").toPandas().toPandas()
    predictions_json = predictionsDF_pandas.to_json()

    return predictions_json


def parse_args():

    parser = argparse.ArgumentParser('Flask Service')
    parser.add_argument("--model-path", type=str, default="../model/rf")
    parser.add_argument("--loglevel", type=str, default="INFO")
    parser.add_argument("--columns_to_drop", type=list, 
                        default=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                        'incident_date','auto_make','auto_model','insured_occupation','age', 'total_claim_amount'])
    parser.add_argument("--columns_to_encode", type=list, 
                        default=['policy_csl', 'insured_sex', 'insured_education_level','insured_hobbies', 'insured_relationship',
                        'incident_type', 'incident_severity','authorities_contacted', 'incident_state', 'incident_city','collision_type'])
    parser.add_argument("--preprocess_hobbies", type=bool, default=True)
    parser.add_argument("--store_schema", type=bool, default=False)
    parser.add_argument("--schema_path", type=str, default="/src/data/feature_store/")

    parser.add_argument("--best_hyper_params_filepath", type=str, default='/src/data/best_hyper_params')
    parser.add_argument("--model_path", type=str, default='/src/model')
    
    parser.add_argument("--host", type=str,
                        default='127.0.0.1')
    
    parser.add_argument("--port", type=int,
                        default=8080)
    
    return parser.parse_args()


def start_server(host, port):
    app.run(host=host, port=port, debug=True)

    
def main():
    args = parse_args()
    start_server(args.host, args.port)


if __name__ == "__main__":
    main()
















# import json
# import time
# import schedule
# import http.client
# import os

# from pyspark.sql import SparkSession



# def get_prediction_result():
    
#     c = http.client.HTTPConnection('localhost', 9000)
#     spark = SparkSession.builder.appName("Zeiss_prediction").getOrCreate()

#     df = spark.read.csv(os.getcwd() + '/src/data/insurance_claims.csv', header=True, sep=',', inferSchema=True)
#     df_pandas = df.toPandas()
#     df_json = df_pandas.to_json()
#     c.request('POST', '/get_prediction_results', json.dumps(df_json))
#     result = c.getresponse().read()
#     prediction_result = json.loads(result)
#     prediction_result_df = spark.read.json(sc.parallelize([prediction_result]))
#     print(prediction_result_df.show(5))

#     return prediction_result_df
    

# if __name__ == '__main__':
#     schedule.every(0).minutes.do(get_prediction_result)
#     while True:
#         schedule.run_pending()
#         time.sleep(1)