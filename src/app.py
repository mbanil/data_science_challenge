from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')


@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


def parse_args():
    
   parser = argparse.ArgumentParser('Flask Service')
   parser.add_argument("--loglevel", type=str, default="INFO")
   parser.add_argument("--columns_to_drop", type=list, 
                     default=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                     'incident_date','auto_make','auto_model','insured_occupation','age', 'total_claim_amount'])
   parser.add_argument("--columns_to_encode", type=list, 
                     default=['policy_csl', 'insured_sex', 'insured_education_level','insured_hobbies', 'insured_relationship',
                     'incident_type', 'incident_severity','authorities_contacted', 'incident_state', 'incident_city','collision_type'])
   parser.add_argument("--preprocess_hobbies", type=bool, default=True)
   parser.add_argument("--store_schema", type=bool, default=False)
   parser.add_argument("--schema_path", type=str, default="/src/data_repo/feature_store/schema.pkl")

   parser.add_argument("--best_hyper_params_filepath", type=str, default='/src/data/best_hyper_params')
   parser.add_argument("--model_path", type=str, default='/src/model/model.pkl')
   
   return parser.parse_args()

if __name__ == '__main__':
   app.run()