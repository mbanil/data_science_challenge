import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Zeiss Task')


    parser.add_argument("--app_name", type=str, default="Zeiss_classification_task")
    parser.add_argument("--data_filename", type=str, default="./data/insurance_claims.csv",
                                    help="data file in csv format")
    parser.add_argument("--columns_to_drop", type=list, 
                        default=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                        'incident_date','auto_make','auto_model','insured_occupation','age', 'total_claim_amount'])
    parser.add_argument("--schema_filename", type=str, default="/src/data/feature_store/schema.pkl")
    parser.add_argument("--columns_to_encode", type=list, 
                        default=['policy_csl', 'insured_sex', 'insured_education_level','insured_hobbies', 'insured_relationship',
                        'incident_type', 'incident_severity','authorities_contacted', 'incident_state', 'incident_city','collision_type'])
    parser.add_argument("--preprocess_hobbies", type=bool, default=True)
    parser.add_argument("--is_data_validation_needed", type=bool, default=True)
    parser.add_argument("--hyper_params", type=dict, 
                        default={'criterion': 'gini', 
                                    'max_depth': 5, 
                                    'min_samples_leaf': 2, 
                                    'min_samples_split': 2
                                    })
    parser.add_argument("--target", type=str, default='fraud_reported')

    return parser.parse_args()























# import argparse

# def get_args():
#     """
    
#     Returns
#     ---------------
#     parsed arguments
#     """
#     parser = argparse.ArgumentParser('Adidas Task')
#     parser.add_argument("--app_name", type=str, default="Adidas_prediction_task")
#     parser.add_argument("--mode", type=str, default="development",
#                                     help="define staging mode - development, production")
#     parser.add_argument("--training_data_filename", type=str, default="Data_repository/Data/train.csv",
#                                     help="training data file in csv format")
#     parser.add_argument("--testing_data_filename", type=str, default="Data_repository/Data/test.csv",
#                                     help="testing data file in csv format")
#     parser.add_argument("--feature_store_directory", type=str, default="Data_repository/feature_store")
#     parser.add_argument("--schema_filename", type=str, default="Data_repository/feature_store/schema.pkl")
#     parser.add_argument("--model_filename", type=str, default="Models/model.pkl")
#     parser.add_argument("--old_model_filename", type=str, default="Models/model_old.pkl")

#     parser.add_argument("--preprocessed_data_filename", type=str, default="Data_repository/Data/pre_processed_data.csv")
#     parser.add_argument("--error_logfile", type=str, default="Logs/errorlogs.txt")
#     parser.add_argument("--logfile", type=str, default="Logs/logs.txt")

#     parser.add_argument("-cat_columns", type=list, 
#                                     default=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
#     parser.add_argument("--is_data_validation_needed", type=bool, default=True)
#     parser.add_argument("--hyper_params", type=dict, 
#                         default={'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25],
#                                           'max_depth': [3, 4, 5, 6],
#                                           'gamma': [0, 1, 2, 5, 10, 15],
#                                           'min_child_weight': [1, 3, 5, 7]
#                                           })
#     parser.add_argument("--target", type=str, default='sales')
#     parser.add_argument("--grouping_param", type=str, default='gamma')
#     parser.add_argument("--data_file_name", type=str, default="Data_repository/Data/train.csv",
#                                     help="training data file in csv format")
#     parser.add_argument("--sell_price_threshold", type=int, default=20)
#     parser.add_argument("--sales_threshold", type=int, default=100)   
#     return parser.parse_args('')