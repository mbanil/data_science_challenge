import os

from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder


def data_extraction(spark, file_name, args):

    # try:
    if os.path.isfile(file_name):
        df = spark.read.csv(file_name, header=True, sep=',', inferSchema=True)
        return df
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    
    # except Exception as e:
    #     # writeLogFile(args.error_logfile, datetime.datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: "+  str(e))                    
    

def data_preprocessing(df, cols_todrop, is_preprocess_hobbies, cols_toencode):

    df = drop_unnecessary_column(df, cols_todrop)

    if is_preprocess_hobbies:
        df = preprocess_hobbies(df)
    
    df = remove_outliers(df)

    df = replace_na2mode(df, "collision_type")
    
    columns_with_na = ['property_damage', 'police_report_available']

    for col in columns_with_na:
        df = replace_by_condition(df, col, '?', 'NO', df[col])
        df = replace_by_condition(df, col, 'YES', 1, 0)

    df = replace_by_condition(df, 'fraud_reported', 'Y', 1 , 0)

    for col in cols_toencode:
        df = encode_data(df, col)
    
    return df


def remove_outliers(df):

    return df.where("umbrella_limit>=0")
    

def drop_unnecessary_column(df, cols_todrop):

    return df.drop(*cols_todrop)


def preprocess_hobbies(df):

    return df.withColumn('insured_hobbies', when(df['insured_hobbies'] =='chess',df['insured_hobbies'])\
        .when(df['insured_hobbies'] =='cross-fit',df['insured_hobbies'])\
        .otherwise('Others'))

def replace_na2mode(df, column):

    mode =  df.groupby(column).count().orderBy("count", ascending=False).first()[0]
    return replace_by_condition(df, column, '?', mode, df['collision_type'])


def replace_by_condition(df, column, condition, val_pos, val_neg):

    return df.withColumn(column, when(df[column] == condition,val_pos).otherwise(val_neg))


def encode_data(df,col):
    indexer = StringIndexer(inputCol=col, outputCol=col+'Index')

    df = indexer.fit(df).transform(df)

    onehotencoder = OneHotEncoder(inputCol=col+'Index', outputCol=col+'Vector')

    df = onehotencoder.fit(df).transform(df)

    cols_drop = (col,col+'Index')
    df = df.drop(*cols_drop)

    return df
      

# import os
# import errno
# import pickle
# import datetime
# import pandas as pd
# import sys
# sys.path.append('../')
# from Logs.logger import writeLogFile


# def store_schema(df, schema_file):
#     """    
#     To store data schema if not already exists
    
#     Paramters
#     --------------------------
#     df : pandas dataframe 
#     schema_file : path to schema file
    
#     Returns
#     ---------------
#     None    
#     """   
#     if not os.path.isfile(schema_file):
#         schema = {'columns': set(df.columns)}
#         schema_file = open(schema_file, "wb")
#         pickle.dump(schema, schema_file)
        
    
# def encode_cat_column(df, column, args):
#     """    
#     Encodes categorical series data and saves the info file in pkl format
    
#     Paramters
#     --------------------------
#     df : pandas dataframe 
#     column : categorical column name
#     args : parsed argument object
    
    
#     Returns
#     ---------------
#     cat_code : dictionary of categorical keys and encoded value
#     """   
    
#     unique_elements = list(df[column].unique())
#     value = 0
#     cat_code = dict()
#     for i in unique_elements:
#         cat_code[i] = value 
#         value += 1
        
#     cat_code_file = open(args.feature_store_directory + "/" + column + "_cat_code.pkl", "wb")
#     pickle.dump(cat_code, cat_code_file)
#     return cat_code


# def update_skewed_categorical_data(df, column, cat_code, cat_file):
#     """    
#     Updates encoded values for categorical data if skewed
        
#     Paramters
#     --------------------------
#     df : pandas dataframe 
#     column : skewed feature column
#     cat_code : dictionary of encoded values for categorical data
#     cat_file : categorical file name
    
#     Returns
#     ---------------
#     cat_code : updated cat codes
#     """  
#     unique_elements = list(df[column].unique())
#     skewed_values = set(unique_elements) - set(list(cat_code.keys()))
#     last_element_value = cat_code[list(cat_code.keys())[-1]] # lastly encoded label value
#     for element in skewed_values:
#         last_element_value += 1
#         cat_code[element] = last_element_value
   
#     # Saving updated cat_code data  
#     cat_code_file = open(cat_file, "wb")
#     pickle.dump(cat_code, cat_code_file)
#     print(cat_code)
#     return cat_code
    

# def get_categorical_data(df, args, skewed=False, skewed_columns=None):
#     """    
#     Encodes categorical columns with numeric value
        
#     Paramters
#     --------------------------
#     df : pandas dataframe 
#     args : parsed argument object
    
    
#     Returns
#     ---------------
#     df : dataframe of encoded categorical values
#     """       
    
#     cat_columns = args.cat_columns
#     for column in cat_columns:
#         cat_file = args.feature_store_directory + "/" + column + "_cat_code.pkl"

#         if os.path.isfile(cat_file):
#             cat_code = pickle.load(open(cat_file, "rb"))
#             if skewed and (skewed_columns is not None and len(skewed_columns) > 0) and (column in skewed_columns):
#                 cat_code = update_skewed_categorical_data(df, column, cat_code, cat_file)
#                 print('categorical data '+ column +' updated !!!!')
                
#         else:
#             cat_code = encode_cat_column(df, column, args)
#         df[column] = df[column].map(cat_code)
#     return df



# def data_preprocessing(df):
#     """    
#     Data preprocessing - remove or replace improper data
    
#     Paramters
#     --------------------------
#     df : pandas datafrmae 
    
#     Returns
#     ---------------
#     df : processed data in pandas dataframe format

#     """      
#     df = df.drop(['id'], axis=1)
#     df = df.dropna()
#     return df


# def feature_extraction(df, args, skewed=False, skewed_columns=None):
#     """    
#     Extracts features for the given data
    
#     Paramters
#     --------------------------
#     df : pandas datafrmae 
#     args : parsed argument object

    
#     Returns
#     ---------------
#     df : extracted features in dataframe format

#     """  
    
#     df = get_categorical_data(df, args, skewed, skewed_columns)
#     df.to_csv(args.preprocessed_data_filename)
#     return df    
    