



import os
import errno
import pickle
import datetime
import pandas as pd
import sys
sys.path.append('../')
from Logs.logger import writeLogFile


def store_schema(df, schema_file):
    """    
    To store data schema if not already exists
    
    Paramters
    --------------------------
    df : pandas dataframe 
    schema_file : path to schema file
    
    Returns
    ---------------
    None    
    """   
    if not os.path.isfile(schema_file):
        schema = {'columns': set(df.columns)}
        schema_file = open(schema_file, "wb")
        pickle.dump(schema, schema_file)
        
    
def encode_cat_column(df, column, args):
    """    
    Encodes categorical series data and saves the info file in pkl format
    
    Paramters
    --------------------------
    df : pandas dataframe 
    column : categorical column name
    args : parsed argument object
    
    
    Returns
    ---------------
    cat_code : dictionary of categorical keys and encoded value
    """   
    
    unique_elements = list(df[column].unique())
    value = 0
    cat_code = dict()
    for i in unique_elements:
        cat_code[i] = value 
        value += 1
        
    cat_code_file = open(args.feature_store_directory + "/" + column + "_cat_code.pkl", "wb")
    pickle.dump(cat_code, cat_code_file)
    return cat_code


def update_skewed_categorical_data(df, column, cat_code, cat_file):
    """    
    Updates encoded values for categorical data if skewed
        
    Paramters
    --------------------------
    df : pandas dataframe 
    column : skewed feature column
    cat_code : dictionary of encoded values for categorical data
    cat_file : categorical file name
    
    Returns
    ---------------
    cat_code : updated cat codes
    """  
    unique_elements = list(df[column].unique())
    skewed_values = set(unique_elements) - set(list(cat_code.keys()))
    last_element_value = cat_code[list(cat_code.keys())[-1]] # lastly encoded label value
    for element in skewed_values:
        last_element_value += 1
        cat_code[element] = last_element_value
   
    # Saving updated cat_code data  
    cat_code_file = open(cat_file, "wb")
    pickle.dump(cat_code, cat_code_file)
    print(cat_code)
    return cat_code
    

def get_categorical_data(df, args, skewed=False, skewed_columns=None):
    """    
    Encodes categorical columns with numeric value
        
    Paramters
    --------------------------
    df : pandas dataframe 
    args : parsed argument object
    
    
    Returns
    ---------------
    df : dataframe of encoded categorical values
    """       
    
    cat_columns = args.cat_columns
    for column in cat_columns:
        cat_file = args.feature_store_directory + "/" + column + "_cat_code.pkl"

        if os.path.isfile(cat_file):
            cat_code = pickle.load(open(cat_file, "rb"))
            if skewed and (skewed_columns is not None and len(skewed_columns) > 0) and (column in skewed_columns):
                cat_code = update_skewed_categorical_data(df, column, cat_code, cat_file)
                print('categorical data '+ column +' updated !!!!')
                
        else:
            cat_code = encode_cat_column(df, column, args)
        df[column] = df[column].map(cat_code)
    return df


def data_extraction(file_name, args):
    """    
    Extracts data from given file name
    
    Paramters
    --------------------------
    file_name : path for stored data
    
    Returns
    ---------------
    df : extracted data in pandas dataframe format
    
    Raises
    -----------------
    raise exception if file not found
    """   
    try:
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)
            return df
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    
    except Exception as e:
        writeLogFile(args.error_logfile, datetime.datetime.now().strftime('%Y-%m-%d : %H:%M:%S'), "| Log from file: "+os.path.basename(__file__) +"    "+ "Message: "+  str(e))                    
    

def data_preprocessing(df):
    """    
    Data preprocessing - remove or replace improper data
    
    Paramters
    --------------------------
    df : pandas datafrmae 
    
    Returns
    ---------------
    df : processed data in pandas dataframe format

    """      
    df = df.drop(['id'], axis=1)
    df = df.dropna()
    return df


def feature_extraction(df, args, skewed=False, skewed_columns=None):
    """    
    Extracts features for the given data
    
    Paramters
    --------------------------
    df : pandas datafrmae 
    args : parsed argument object

    
    Returns
    ---------------
    df : extracted features in dataframe format

    """  
    
    df = get_categorical_data(df, args, skewed, skewed_columns)
    df.to_csv(args.preprocessed_data_filename)
    return df    
    