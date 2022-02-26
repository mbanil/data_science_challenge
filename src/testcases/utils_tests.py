import os
import sys
import pytest
import pandas as pd
import copy

# sys.path.append('../../')
sys.path.insert(0,os.getcwd()+'/src')
print(sys.path)

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
    assert df.shape[0]== 2

def test_drop_columns(data):
    data = utils.drop_unnecessary_column(data, ['age'])
    assert ('age' in data.columns) == False

def test_replace2bin(data):

    df = data[data['policy_number'] == 687698]
    data = utils.replace2bin(df)
    print(data.head())

    assert ((0 in data["property_damage"].unique()) or (1 in data["property_damage"].unique())) and ((0 in data["police_report_available"].unique()) or (1 in data["police_report_available"].unique())) 

def test_preprocess_hobbies(data):

    df = utils.preprocess_hobbies(data)
    assert ('chess' in df["insured_hobbies"].unique()) or ('cross-fit' in df["insured_hobbies"].unique()) or ('Other' in df["insured_hobbies"].unique())


# def 
###################################################################################################

