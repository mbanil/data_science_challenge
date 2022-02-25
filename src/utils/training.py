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

def save_best_params(model, best_params_path):

    best_file = open(os.getcwd() + best_params_path, "wb")
    pickle.dump(model.best_params_, best_file)
    best_file.close()

def load_hyper_params(best_params_path):

    best_hyper_params = pickle.load( open(os.getcwd() + best_params_path, "rb" ) )

    return best_hyper_params
    

def train(df, args):

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

        save_best_params(grid_search_result, args.best_hyper_params_filepath)

        model = grid_search_result.best_estimator_
        
    else:
        hyper_params = load_hyper_params(args.best_hyper_params_filepath)
        model = train_randomforest(X_train, y_train, hyper_params)
 
    y_pred = model.predict(X_test)

    print("Training Accuracy: ", model_rfc.score(X_train, y_train))
    print('Testing Accuarcy: ', model_rfc.score(X_test, y_test))

    print(confusion_matrix(Y_test, y_pred_rf))
    print(classification_report(Y_test, y_pred_rf))

    model.write().overwrite().save(os.getcwd() + args.model_path + '/rf')

