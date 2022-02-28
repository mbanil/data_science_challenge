# About the repository

The repository consists of the following folders:

* Jupyter Notebook: Exploratory Data Analysis is performed in the jupyter notebook.
    * The properties of the dataset is analysed. 
    * Then, the data is visualized using pandas profiling.
        *  Numerical and categorical variables are identified. Statistical analysis is performed on the numerical data. Further correlation between variables is performed to remove similar variables. Outliers are detected from the analysis. 
        * The distribution of the categorical variables between the  positive and negative results are analysed.  
    * This is followed by data pre-processing. Based on the visualization, unnecessary columns are dropped, pecularities are removed, empty values are imputed and finally the categorical values are encoded.
    * Then, the data is split into training and test sets and standardization is done.
    * Different binary classification models (Random forest, XGBoost, KNN, Cat Boost) are tested to find out the best performing model. 
    * The best performing model is selected and feature importance is identified. Least affecting features are removed using a threshold.
    * After the least important features are removed, the model is tuned to find the best hyperparamter using grid search.

* src : Contains the code for training a classification model. This is built using pandas.
    * data_repo folder consists of training data, schema information and best tuned hyper parameters.
    * logger folder contains code for logging and log files generated
    * model folder consists of a trained classification model that is saved
    * templates folder consists of html file used for the app
    * app.py - flask app used to call the model for prediction
    * args.py - parse args for the training script
    * train-classifier-model.py - main script for training the model

* src_pyspark: This folder has the same structure as 'src' folder. The only missing feature here is the restoration of one-hot encoding order while predicting results.

* CI/CD is carried out using Github actions. The python app is deployed to the Azure web app



usage: train-classification-model.py [-h] [--app_name APP_NAME]
                                     [--data_filename DATA_FILENAME]
                                     [--schema_path SCHEMA_PATH]
                                     [--best_hyper_params_filepath BEST_HYPER_PARAMS_FILEPATH]
                                     [--model_path MODEL_PATH]
                                     [--columns_to_drop COLUMNS_TO_DROP]
                                     [--columns_to_encode COLUMNS_TO_ENCODE]
                                     [--store_schema STORE_SCHEMA]
                                     [--preprocess_hobbies PREPROCESS_HOBBIES]
                                     [--tune_hyper_params TUNE_HYPER_PARAMS]
                                     [--grid_params GRID_PARAMS]

Zeiss Task



optional arguments:
  -h, --help            show this help message and exit
  --app_name APP_NAME
  --data_filename DATA_FILENAME
                        data file in csv format
  --schema_path SCHEMA_PATH
  --best_hyper_params_filepath BEST_HYPER_PARAMS_FILEPATH
  --model_path MODEL_PATH
  --columns_to_drop COLUMNS_TO_DROP
  --columns_to_encode COLUMNS_TO_ENCODE
  --store_schema STORE_SCHEMA
  --preprocess_hobbies PREPROCESS_HOBBIES
  --tune_hyper_params TUNE_HYPER_PARAMS
  --grid_params GRID_PARAMS