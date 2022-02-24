
import pathlib
import argparse
# import pickle

from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd

# from logger import LOG
# from features.build_features import encode_test_input_output
# from models.model import load_model

app = Flask(__name__)

# use gloabl values to avoid reloading the models from disk for every requrest

le = None
td_idf_vectorizer_main_text = None
td_idf_vectorizer_add_text = None
features = None
classifier = None
args = None


@app.route('/')
def index():
    # return render_template('/index.html')
    return "OK"


@app.route('/predict-product')
def prediction_product(main_text=None, add_text=None):

    global le, td_idf_vectorizer_main_text, td_idf_vectorizer_add_text, classifier
    main_text = request.args.get('main_text', default='', type=str)
    add_text = request.args.get('add_text', default='', type=str)
    df = pd.DataFrame(columns=['main_text', 'add_text'])
    df = df.append(
        {'main_text': main_text, 'add_text': add_text}, ignore_index=True)
    try:
        X_test, _, _ = encode_test_input_output(args, df)
        
    except Exception as e:
       LOG.exception('Error in vectorizing')
    
    y_pred = classifier.predict(X_test)

    return {
        'product-group': le.inverse_transform(y_pred)[0]
    }

def parse_args():
    """ Parse default  and optional arguments

    Returns:
        argparse: [description]
    """
    parser = argparse.ArgumentParser('Flask Service')
    parser.add_argument(
        "--model-path", type=str, default="../models/inference/ml-model/productgroup_DTC_main_text,add_text.pkl", help="name of model in {RFC, GBC, LogR, DTC, SVC}"
    )

    parser.add_argument("--features", type=str, default="main_text,add_text", help="main_text, add_text")
    
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default="../../data/testset_C.csv",
        help="path to csv",
    )

    parser.add_argument("--loglevel", type=str, default="INFO")
    parser.add_argument("--label-encoding-path", type=str,
                        default='../models/inference/labels')
    parser.add_argument("--word-embeddings-path", type=str,
                        default='../models/inference/embeddings')
    
    parser.add_argument("--host", type=str,
                        default='127.0.0.1')
    
    parser.add_argument("--port", type=int,
                        default=8080)
    

    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path("../../models/inference/ml-model"),
        help="dir to save models",
    )
    return parser.parse_args()

def start_server(host, port):
    app.run(host=host, port=port, debug=True)

    
def main():
    
    
    # use gloabl values to avoid reloading the models from disk for every requrest
    # global le, td_idf_vectorizer_main_text, td_idf_vectorizer_add_text, features, classifier,args

    args = parse_args()
    
    # classifier = load_model(args.model_path)
    # le = pickle.load(open("{}/label_encoding.pickle".format(args.label_encoding_path), "rb"))
    # features =  [str(item) for item in args.features.split(',')]
       
    
    # try:
    #     td_idf_vectorizer_main_text = pickle.load(open("{}/{}_vectorizer.pickle".format(args.word_embeddings_path, 'main_text'), "rb"))
    # except:
    #     LOG.exception('Unable to vectorize the main text')

    # try:
    #     td_idf_vectorizer_add_text = pickle.load(open("{}/{}_vectorizer.pickle".format(args.word_embeddings_path, 'add_text'), "rb"))
    # except:
    #     LOG.exception('Unable to vectorize the Additional text')

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