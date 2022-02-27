import os
import sys
import pytest

from pyspark.sql import SparkSession

sys.path.insert(0,os.getcwd()+'/src_pyspark')
print(sys.path)

from utils import utils


spark = SparkSession.builder.appName('test_framework').getOrCreate()


@pytest.fixture(scope='module')
def data():
    df = spark.read.csv('./src_pyspark/testcases/data_testcases/data_testcases.csv', header=True, sep=',', inferSchema=True)    
    return df


########################### Unit test block ##################################################

def test_remove_outliers(data):
    df = utils.remove_outliers(data)
    print(df.show())
    assert df.agg({'umbrella_limit': 'min'}).first()[0] == 0

def test_drop_columns(data):
    data = utils.drop_unnecessary_column(data, ['age'])
    assert ('age' in data.columns) == False

###################################################################################################
