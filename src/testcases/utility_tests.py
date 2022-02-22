import os
import sys
import pytest

from pyspark.sql import SparkSession

sys.path.append('../')
# sys.path.insert(0,os.getcwd()+'\\src')
from utils import utils


spark = SparkSession.builder.appName('test_framework').getOrCreate()


@pytest.fixture(scope='module')
def data():
    df = spark.read.csv('./src/testcases/data_testcases/data_testcases.csv', header=True, sep=',', inferSchema=True)    
    return df


########################### Unit test block ##################################################

def test_remove_outliers(data):
    df = utils.remove_outliers(data)
    print(df.show())
    assert df.agg({'umbrella_limit': 'min'}).first()[0] == 0

def test_drop_columns(data):
    data = utils.drop_unnecessary_column(data, ['age'])
    assert ('age' in data.columns) == False

# def test_pandas_to_spark(data):
#     df = pandas_to_spark(spark, data)
#     assert isinstance(df, pd.DataFrame) == False

###################################################################################################





# import sys
# import copy
# import pytest
# import pandas as pd
# from test_args import get_args
# from pyspark.sql import SparkSession
# from sklearn.model_selection import train_test_split

# sys.path.append('../')
# from Utilities.training_utility_spark import spark_udf_decorator, pandas_to_spark

# args = get_args()
# spark = SparkSession.builder.appName('test_framework').getOrCreate()

# @pytest.fixture(scope='module')
# def data():
#     df =  pd.DataFrame([{
#                                     'learning_rate': 0.1,
#                                     'max_depth': 10,
#                                     'gamma': 0,
#                                     'min_child_weight': 1
#                                     }])
#     return df

# @pytest.fixture(scope='module')
# def udf():
#     data_processed = pd.read_csv(args.preprocessed_data_filename_tuning)
#     data_processed = copy.deepcopy(data_processed)
#     X = data_processed.drop(['sales'], axis=1)
#     y = data_processed['sales']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
#     udf_ref = spark_udf_decorator(copy.deepcopy(X_train.to_json()), copy.deepcopy(X_test.to_json()),
#             copy.deepcopy(y_train) , copy.deepcopy(y_test))
#     return udf_ref




# ########################### Integration test block ##################################################
# def test_tune_hyper_param_type(data, udf):
#     df = pandas_to_spark(spark, data)
#     tuning_result = df.groupby('gamma').apply(udf).toPandas()
#     assert isinstance(tuning_result, pd.DataFrame) == True
    
    
# def test_tune_hyper_param_result_schema(data, udf):
#     df = pandas_to_spark(spark, data)
#     tuning_result = df.groupby('gamma').apply(udf).toPandas()
#     expected = set(list(data.columns) + ['mae'])
#     actual = set(tuning_result.columns)
#     assert expected == actual
# ###################################################################################################


















# import os
# import pytest
# import pandas as pd
# from pandas.testing import assert_frame_equal
# from test_args import get_args
# import sys
# import copy
# sys.path.append('../')

# from Utilities.utility import data_extraction, data_preprocessing, feature_extraction, get_categorical_data, store_schema, encode_cat_column

# args = get_args()

# @pytest.fixture(scope='module')
# def data():
#     df =  pd.DataFrame([{'id': 'HOBBIES_2_032_TX_3',
#                                        'item_id': 'HOBBIES_2_032',
#                                        'dept_id': 'HOBBIES_2',
#                                        'cat_id': 'HOBBIES',
#                                        'store_id': 'TX_3',
#                                        'state_id': 'TX',
#                                        'sell_price': 0.98,
#                                        'sales': 1200}])
#     return df


# @pytest.fixture(scope='module')
# def preprocessed_data():
#     df =  pd.DataFrame([{'item_id': 0,
#                                        'dept_id': 0,
#                                        'cat_id': 0,
#                                        'store_id': 0,
#                                        'state_id': 0,
#                                        'sell_price': 0.98,
#                                        'sales': 1200}])
#     return df


# ########################### Unit test block ##################################################
# def test_store_schema(data):
#     store_schema(data, args.schema_filename)
#     assert os.path.isfile(args.schema_filename)  == True


# def test_encode_cat_column(data):
#     expected_cat_code = {'HOBBIES_2_032': 0}
#     actual_cat_code = encode_cat_column(data, 'item_id', args)
#     assert expected_cat_code == actual_cat_code
    
    
# def test_data_extraction(data):
#     actual = data_extraction(args.test_training_data_filename, args)
#     assert_frame_equal(data, actual)
    
    
# def test_data_preprocessing(data):
#     expected = copy.deepcopy(data)
#     expected = expected.drop(['id'], axis=1)
#     expected = expected.dropna()
#     actual = data_preprocessing(data)
#     assert_frame_equal(expected, actual)

# ###################################################################################################


# ########################### Integration test block ##################################################

# def test_get_categorical_data(preprocessed_data, data):
#     actual = data_preprocessing(data)
#     actual = get_categorical_data(actual, args)
#     assert_frame_equal(preprocessed_data, actual)


# def test_feature_extraction(preprocessed_data, data):
#     actual = data_preprocessing(data)
#     actual = feature_extraction(actual, args)
#     assert_frame_equal(preprocessed_data, actual)
    
# ###################################################################################################