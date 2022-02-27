import pandas as pd
import numpy as np
import pickle
import os

from logger import LOG

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV

def standardize_data(df):
    """Standardization of the input data

    Args:
        df : pandas dataframe

    Returns:
        Standardized pandas dataframe
    """
    LOG.info("Standardize Data")

    sc = StandardScaler()
    df_std = sc.fit_transform(df)

    return df_std


def train_randomforest(X_train, y_train, params=None):
    """Train a randm forest classifier

    Args:
        X_train : Input training data
        y_train : Output training data
        params (optional): Hyper-parameters for the model. Defaults to None.

    Returns:
        Trained random-forest model
    """

    LOG.info('"Train random forest with hyper-params: {}'.format(str(params)))

    if params:
        model_rfc = BalancedRandomForestClassifier(criterion=params["criterion"], max_depth=params["max_depth"], 
                                min_samples_leaf=params["min_samples_split"], min_samples_split= params["min_samples_leaf"])
    else:
        model_rfc = BalancedRandomForestClassifier()
            
    LOG.info('Fit random forest to training data')
    model_rfc.fit(X_train, y_train)

    return model_rfc


def tune_hyper_params(model, grid_params, X_train, y_train):
    """Tune hyper parameters using GridSearch

    Args:
        model : Classifier model    
        grid_params: Range of hyper parameters for tuning
        X_train: Input training data
        y_train: Output training data

    Returns:
        Grid search output
    """
    LOG.info('"Tune random forest parameters in the range: {}'.format(str(grid_params)))

    grid_search = GridSearchCV(model, grid_params, n_jobs = -1,  cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)

    return grid_search


def save_pickle(data, path):
    """Save data as a pickle file

    Args:
        data : Data to stored
        path : Path of the pickle file
    """

    LOG.info('Save data as a pickle file')

    file = open(os.getcwd() + path, "wb")
    pickle.dump(data, file)
    file.close()


def load_pickle(path):
    """Load pickle file

    Args:
        path : Path for the pickle file

    Returns:
        Loaded pickle file
    """
    LOG.info('Load data as a pickle file')

    data = pickle.load( open(os.getcwd() + path, "rb" ) )

    return data
    

def train(df, args):
    """Train the data and saved the trained model

    Args:
        df: pandas dataframe
        args : Input arguments
    """
    LOG.info("Begin training data")

    df['fraud_reported'].replace(to_replace='Y', value=1, inplace=True)
    df['fraud_reported'].replace(to_replace='N',  value=0, inplace=True)

    LOG.info("Create input and output for training")

    X = df[df.columns.drop('fraud_reported')]
    Y = df['fraud_reported']

    LOG.info("Split the data into training and test set")

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, Y, test_size = 0.2, random_state=42)

    X_train = standardize_data(X_train_df)
    X_test = standardize_data(X_test_df)
    y_train = np.array(y_train_df)
    y_test = np.array(y_test_df)

    LOG.info('"Training Dataset Count: {}'.format(str(X_train.shape)))
    LOG.info('"Test Dataset Count: {}'.format(str(X_test.shape)))    

    if(args.tune_hyper_params):
        model_rfc = train_randomforest(X_train, y_train)

        grid_search_result = tune_hyper_params(model_rfc, args.grid_params, X_train, y_train)

        save_pickle(grid_search_result.best_params_, args.best_hyper_params_filepath)

        model = grid_search_result.best_estimator_
        
    else:
        hyper_params = load_pickle(args.best_hyper_params_filepath)
        model = train_randomforest(X_train, y_train, hyper_params)
 
    LOG.info('Predict test data')
    y_pred = model.predict(X_test)

    LOG.info('Training Accuracy: {}'.format(model.score(X_train, y_train)))
    LOG.info('"Testing Accuarcy: {}'.format(model.score(X_test, y_test)))  

    LOG.info(confusion_matrix(y_test, y_pred))
    LOG.info(classification_report(y_test, y_pred))

    LOG.info("End training data")

    save_pickle(model, args.model_path)
