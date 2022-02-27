from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

import os
import pathlib
import argparse
import json
import pandas as pd
import numpy as np

from utils import utils, training, data_validation
import requests

app = Flask(__name__)


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/predictfraud', methods = ['GET'])
def predictfraud(data=None):
       
   
    args = parse_args()

    data_json = request.args.get('data')
    a_json = json.loads(data_json)

    df = pd.DataFrame.from_dict(a_json)
    df = utils.data_preprocessing(df, args)
    df = utils.encode_data(df, args.columns_to_encode)

    #  check original schema

    schema = training.load_pickle(args.schema_path)
    df = data_validation.configure_schema(schema, df)

    model = training.load_pickle(args.model_path)
    df = training.standardize_data(df)
    model.predict(df)
    results = model.predict(df)

    return {
       'results': json. dumps(results.tolist())
    }



def parse_args():
    
   parser = argparse.ArgumentParser('Flask Service')
   parser.add_argument("--columns_to_drop", type=list, 
                     default=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                     'incident_date','auto_make','auto_model','insured_occupation','age', 'total_claim_amount'])
   parser.add_argument("--columns_to_encode", type=list, 
                     default=['policy_csl', 'insured_sex', 'insured_education_level','insured_hobbies', 'insured_relationship',
                     'incident_type', 'incident_severity','authorities_contacted', 'incident_state', 'incident_city','collision_type'])
   
   parser.add_argument("--preprocess_hobbies", type=bool, default=True)
   parser.add_argument("--store_schema", type=bool, default=False)
   
   parser.add_argument("--schema_path", type=str, default="/src/data_repo/feature_store/schema.pkl")
   parser.add_argument("--best_hyper_params_filepath", type=str, default='/src/data_repo/best_hyper_params/best_hyper_parameters.pkl')
   parser.add_argument("--model_path", type=str, default='/src/model/model.pkl')
   
   return parser.parse_args()

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)