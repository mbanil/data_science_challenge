import os
import pickle
import pandas as pd
import numpy as np

# from logger import LOG

def data_extraction(file_path):
    """Method to load data from csv

    Args:
        file_name : path to file

    Raises:
        FileNotFoundError: Exception if the file is not found

    Returns:
        _type_: pandas dataframe
    """

    try:
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            return df

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    except Exception as e:
        print("Exception occured: " + e)
        # LOG.exception("Exception occured: " + e)
            

def data_preprocessing(df, args):
    """Method to preprocess data before training

    Args:
        df : Input dataframe
        args : arguments

    Returns:
        Preprocessed dataframe
    """

    df = drop_unnecessary_column(df, args.columns_to_drop)

    if args.preprocess_hobbies:
        df = preprocess_hobbies(df)
    
    df = remove_outliers(df)

    df = df.replace('?',np.NaN)
    
    df = replace_na2mode(df, "collision_type")
    
    columns_with_na = ['property_damage', 'police_report_available']

    for col in columns_with_na:
        df = remove_nan2no(df,col)

    df = replace2bin(df)

    return df


def replace2bin(df):
    """Replaces columns with "YES" or "NO" to 1 and 0 respectively

    Args:
        df : Input Dataframe

    Returns:
        Processed dataframe
    """

    df['property_damage'].replace(to_replace='YES', value=1, inplace=True)
    df['property_damage'].replace(to_replace='NO', value=0, inplace=True)

    df['police_report_available'].replace(to_replace='YES', value=1, inplace=True)
    df['police_report_available'].replace(to_replace='NO', value=0, inplace=True)

    return df


def remove_outliers(df):
    """Removes outliers in the dataframe (in cloumn 'umbrella_limit')

    Args:
        df : pandas dataset

    Returns:
        pandas dataframe after removing outliers
    """

    df.drop(df[df['umbrella_limit'] < 0].index, inplace = True)

    # LOG.info("Removed outliers in column 'umbrella_limit'")
    # LOG.info(df.head())

    return df


def drop_unnecessary_column(df, cols_todrop):
    """Drop columns from df

    Args:
        df : pandas dataset
        cols_todrop : columns to be dropped

    Returns:
        pandas dataframe after dropping columns 
    """

    df.drop(cols_todrop, inplace = True, axis = 1)

    # LOG.info("Dropped unnecessary columns from the dataframe")
    # LOG.info(df.head())


    return df


def preprocess_hobbies(df):
    """ In the column 'insured_hobbied' all values are converted to 'others' except 'chess' and 'cross-fit'

    Args:
        df : pandas dataframe

    Returns:
        dataframe after the 'insured_hibbies' cloumn is cleaned
    """

    df['insured_hobbies']=df['insured_hobbies'].apply(lambda x :'Other' if x!='chess' and x!='cross-fit' else x)

    # LOG.info("Pre-processed hobbies columns")
    # LOG.info(df.head())

    return df


def remove_nan2no(df, column):
    """Replaces values with "NaN" to "NO"

    Args:
        df: Input pandas dataframe
        column: cloumns in which the values have to be changes

    Returns:
        processed df
    """

    df[column].fillna('NO', inplace = True)

    return df


def replace_na2mode(df, column):
    """Replaced the NaN with mode in the column selected

    Args:
        df : pandas dataframe
        column : column in which the '?' values are to be replace

    Returns:
        processed dataframe
    """
    
    if(df.shape[0]==1):
        df[column].fillna("Front Collision", inplace = True)
    else:
        df[column].fillna(df[column].mode()[0], inplace = True)

    return df


def encode_data(df,columns_to_encode):
    """One Hot encoding of categorical columns

    Args:
        df : pandas dataframe
        col : categorical columns

    Returns:
        pandas dataframe with one-hot encoding
    """

    cat_df = pd.get_dummies(df[columns_to_encode])

    df_clean = pd.concat([cat_df, df._get_numeric_data()], axis=1)  # joining numeric columns

    return df_clean


def store_schema(df, schema_file):
    """Store the column names as a schema file

    Args:
        df : pandas dataframe
        schema_file : file path
    """

    if not os.path.isfile(schema_file):
        schema = {'columns': list(df.columns)}
        schema_file = open(os.getcwd() + schema_file, "wb")
        pickle.dump(schema, schema_file)
