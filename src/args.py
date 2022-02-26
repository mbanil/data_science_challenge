import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Zeiss Task')


    parser.add_argument("--app_name", type=str, default="Zeiss_classification_task")

    parser.add_argument("--data_filename", type=str, default="./src/data_repo/data/training_data.csv",
                                    help="data file in csv format")
    parser.add_argument("--schema_path", type=str, default="/src/data_repo/feature_store/schema.pkl")
    parser.add_argument("--best_hyper_params_filepath", type=str, default='/src/data_repo/best_hyper_params/best_hyper_parameters.pkl')
    parser.add_argument("--model_path", type=str, default='/src/model/model.pkl')

    parser.add_argument("--columns_to_drop", type=list, 
                        default=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                        'incident_date','auto_make','auto_model','insured_occupation','age', 'total_claim_amount'])
    parser.add_argument("--columns_to_encode", type=list, 
                        default=['policy_csl', 'insured_sex', 'insured_education_level','insured_hobbies', 'insured_relationship',
                        'incident_type', 'incident_severity','authorities_contacted', 'incident_state', 'incident_city','collision_type'])

    parser.add_argument("--store_schema", type=bool, default=True)
    parser.add_argument("--preprocess_hobbies", type=bool, default=True)
    parser.add_argument("--tune_hyper_params", type=bool, default=False)
    
    
    parser.add_argument("--grid_params", type=dict, 
                        default={
                                    'criterion' : ['gini', 'entropy'],
                                    'max_depth' : [3, 5, 7, 10],
                                    'min_samples_split' : range(2, 10, 1),
                                    'min_samples_leaf' : range(2, 10, 1)
                                })
                                
    return parser.parse_args()


