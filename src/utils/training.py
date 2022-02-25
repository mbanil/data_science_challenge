import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV

def standardize_data(df):

    sc = StandardScaler()
    df_std = sc.fit_transform(df)

    return df_std

def train_randomforest(X_train, y_train, params=None):

    if params:
        model_rfc = BalancedRandomForestClassifier(criterion=params["criterion"], max_depth=params["max_depth"], 
                                min_samples_leaf=params["min_samples_split"], min_samples_split= params["min_samples_leaf"])
    else:
        model_rfc = BalancedRandomForestClassifier()
        
            
    model_rfc.fit(X_train, y_train)

    return model_rfc

def tune_hyper_params(model, grid_params, X_train, y_train):
    
    grid_search = GridSearchCV(model, grid_params, n_jobs = -1,  cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)

    return grid_search

def save_pickle(data, path):

    file = open(os.getcwd() + path, "wb")
    pickle.dump(data, file)
    file.close()

def load_pickle(path):

    data = pickle.load( open(os.getcwd() + path, "rb" ) )

    return data
    

def train(df, args):

    df['fraud_reported'].replace(to_replace='Y', value=1, inplace=True)
    df['fraud_reported'].replace(to_replace='N',  value=0, inplace=True)

    X = df[df.columns.drop('fraud_reported')]
    Y = df['fraud_reported']

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, Y, test_size = 0.2, random_state=42)

    X_train = standardize_data(X_train_df)
    X_test = standardize_data(X_test_df)
    y_train = np.array(y_train_df)
    y_test = np.array(y_test_df)

    print("Training Dataset Count: " + str(X_train.shape))
    print("Test Dataset Count: " + str(X_test.shape))
    

    if(args.tune_hyper_params):
        model_rfc = train_randomforest(X_train, y_train)

        grid_search_result = tune_hyper_params(model_rfc, args.grid_params, X_train, y_train)

        save_pickle(grid_search_result.best_params_, args.best_hyper_params_filepath)

        model = grid_search_result.best_estimator_
        
    else:
        hyper_params = load_pickle(args.best_hyper_params_filepath)
        model = train_randomforest(X_train, y_train, hyper_params)
 
    y_pred = model.predict(X_test)

    print("Training Accuracy: ", model.score(X_train, y_train))
    print('Testing Accuarcy: ', model.score(X_test, y_test))

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    save_pickle(model, args.model_path)

