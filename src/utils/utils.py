import os
import pickle

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
        schema_file = open(os.getcwd() + schema_file, "wb")
        pickle.dump(schema, schema_file)
