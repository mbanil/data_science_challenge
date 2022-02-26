import os
import sys
import pytest
import numpy as np
import pandas as pd
import copy

# sys.path.append('../../')
sys.path.insert(0,os.getcwd()+'/src')

from utils import utils


@pytest.fixture(scope='module')
def data():
    df = pd.read_csv('./src/testcases/data_testcases/data_testcases.csv')
    return df


########################### Unit test block ##################################################

def test_remove_outliers(data):
    df = copy.deepcopy(data)
    df = utils.remove_outliers(df)
    print(df['umbrella_limit'].head())
    assert df.shape[0]== 3

def test_drop_columns(data):
    data = utils.drop_unnecessary_column(data, ['age'])
    assert ('age' in data.columns) == False

def test_replace2bin(data):

    df = data[data['policy_number'] == 687698]
    data = utils.replace2bin(df)

    assert ((0 in data["property_damage"].unique()) or (1 in data["property_damage"].unique())) and ((0 in data["police_report_available"].unique()) or (1 in data["police_report_available"].unique())) 

def test_preprocess_hobbies(data):

    df = utils.preprocess_hobbies(data)
    assert ('chess' in df["insured_hobbies"].unique()) or ('cross-fit' in df["insured_hobbies"].unique()) or ('Other' in df["insured_hobbies"].unique())


def test_remove_nan2no(data):
    df = data[data['policy_number'] == 227811]
    data = utils.remove_nan2no(df, 'property_damage')

    assert "NO" in data["property_damage"].unique()

def test_replace_na2mode(data):

    data = utils.replace_na2mode(data, "collision_type")

    assert "Front Collision" in data["collision_type"].unique()

# def test_data_preprocessing(data):

#     data = test_data_preprocessing(df)

###################################################################################################


########################### Integration test block ##################################################

# def test_get_categorical_data(preprocessed_data, data):
#     actual = data_preprocessing(data)
#     actual = get_categorical_data(actual, args)
#     assert_frame_equal(preprocessed_data, actual)


# def test_feature_extraction(preprocessed_data, data):
#     actual = data_preprocessing(data)
#     actual = feature_extraction(actual, args)
#     assert_frame_equal(preprocessed_data, actual)
    
###################################################################################################