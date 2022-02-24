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
    

def data_preprocessing(df, args):

    df = drop_unnecessary_column(df, args.columns_to_drop)

    if args.preprocess_hobbies:
        df = preprocess_hobbies(df)
    
    df = remove_outliers(df)

    df = replace_na2mode(df, "collision_type")
    
    columns_with_na = ['property_damage', 'police_report_available']

    for col in columns_with_na:
        df = replace_by_condition(df, col, '?', 'NO', df[col])
        df = replace_by_condition(df, col, 'YES', 1, 0)

    df = replace_by_condition(df, 'fraud_reported', 'Y', 1 , 0)

    for col in args.columns_to_encode:
        df = encode_data(df, col, args.store_schema, args.schema_path)
    
    return df


def remove_outliers(df):
    """Removes outliers in the dataframe (in cloumn 'umbrella_limit')

    Args:
        df : spark dataset

    Returns:
        spark dataframe after removing outliers
    """

    return df.where("umbrella_limit>=0")
    

def drop_unnecessary_column(df, cols_todrop):
    """Drop columns from spark df

    Args:
        df : spark dataset
        cols_todrop (_type_): columns to be dropped

    Returns:
        spark dataframe after dropping columns 
    """

    return df.drop(*cols_todrop)


def preprocess_hobbies(df):
    """ In thr column 'insured_hobbied' all values are converted to 'others' except 'chess' and 'cross-fit'

    Args:
        df : spark dataframe

    Returns:
        dataframe after the 'insured_hibbies' cloumn is cleaned
    """

    return df.withColumn('insured_hobbies', when(df['insured_hobbies'] =='chess',df['insured_hobbies'])\
        .when(df['insured_hobbies'] =='cross-fit',df['insured_hobbies'])\
        .otherwise('Others'))


def replace_na2mode(df, column):
    """Replaced the '?' with mode in the column selected

    Args:
        df : spark dataframe
        column : column in which the '?' values are to be replace

    Returns:
        processed dataframe
    """

    mode =  df.groupby(column).count().orderBy("count", ascending=False).first()[0]
    return replace_by_condition(df, column, '?', mode, df['collision_type'])


def replace_by_condition(df, column, condition, val_pos, val_neg):
    """Replace '?' values in a column based on some condition

    Args:
        df : spark dataframe
        column : column in which dataframe has to be replaced
        condition : condition for replacing the value
        val_pos : value to be replaced with if the condition is satisfied
        val_neg : value to be replaced with if the condition is not satisfied

    Returns:
        processed spark dataframe
    """

    return df.withColumn(column, when(df[column] == condition,val_pos).otherwise(val_neg))


def encode_data(df,col,storeSchema, schema_path):
    """One Hot encoding of categorical columns

    Args:
        df : spark dataframe
        col : categorical columns

    Returns:
        spark dataframe with one-hot encoding
    """

    if(storeSchema):
        indexer = StringIndexer(inputCol=col, outputCol=col+'Index')
        indexer.write().overwrite().save(os.getcwd() + schema_path + '/schemaData/' + col)
    else:
        indexer = StringIndexer.load(os.getcwd() + schema_path + '/schemaData/' + col)

    df = indexer.fit(df).transform(df)

    onehotencoder = OneHotEncoder(inputCol=col+'Index', outputCol=col+'Vector')

    df = onehotencoder.transform(df)

    cols_drop = (col,col+'Index')
    df = df.drop(*cols_drop)

    return df


def store_schema(df, schema_file):
    """Store the column names as a schema file

    Args:
        df : spark dataframe
        schema_file : file path
    """

    if not os.path.isfile(schema_file):
        schema = {'columns': set(df.columns)}
        schema_file = open(os.getcwd() + schema_file, "wb")
        pickle.dump(schema, schema_file)
