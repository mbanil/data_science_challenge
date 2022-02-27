from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

from flask import json
from werkzeug.exceptions import HTTPException

import sys
import os
import pathlib
import argparse
import json
import pandas as pd
import numpy as np
from types import SimpleNamespace


from utils import utils, training, data_validation
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

app = Flask(__name__)




@app.route('/')
def index():
   return render_template('index.html')


@app.route('/predictfraud', methods = ['GET'])
def predictfraud(data=None):
   
   args = {
         'columns_to_drop': ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location', 'incident_date', 'auto_make', 'auto_model', 'insured_occupation', 'age', 'total_claim_amount'], 
         'columns_to_encode': ['policy_csl', 'insured_sex', 'insured_education_level', 'insured_hobbies', 'insured_relationship', 'incident_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city', 'collision_type'], 
         'preprocess_hobbies': True, 
         'store_schema': False, 
         'schema_path': '/data_repo/feature_store/schema.pkl', 
         'best_hyper_params_filepath': '/data_repo/best_hyper_params/best_hyper_parameters.pkl', 
         'model_path': '/model/model.pkl'
      }

   args = SimpleNamespace(**args)
      
   data_json = request.args.get('data')
   a_json = json.loads(data_json)

   df = pd.DataFrame.from_dict(a_json)
   df = utils.data_preprocessing(df, args)

   print(df.columns)
   df = utils.encode_data(df, args.columns_to_encode)

   schema = training.load_pickle(args.schema_path)
   
   # df = data_validation.configure_schema(schema, df)

   cols_original = list(schema["columns"])
   cols_new = list(df.columns)

   for col in cols_new:
      if col.find("\\") != -1:
         df.rename(columns = {col:col.replace("\\", "")}, inplace = True)

   cols_new = list(df.columns)
   for col in cols_original:
      if not col in cols_new:
         df.insert(2, col, np.full(df.shape[0], 0))

   df = df.reindex(columns=cols_original)

   #  check original schema

   print(df.columns)

   model = training.load_pickle(args.model_path)
   df = training.standardize_data(df)
   model.predict(df)
   results = model.predict(df)

   return {
      'results': json. dumps(results.tolist())
   }




if __name__ == '__main__':
   app.run()