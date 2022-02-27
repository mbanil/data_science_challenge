# About the repository

The repository consists of the following folders:

* Jupyter Notebook: Here the jupyter notebook is located in which Exploratory Data Analysis is performed. 
    * The structure of the data is analysed. 
    * Then, the data is visulaized using pandas profiling.
        *  Numerical and categorical variables are identified. In the numerical data, statistics are understood. Further correlation between variables is performed to remove similar variables. Outliers too are detected from the analysis. 
        * From the categorical variables the distribution of data among positive and negative results are identified.  
    * This is followed by data pre-processing. Based on the visualization, unnecessary columns are dropped, pecularities are removed, empty values are imputed and finally the categorical values are encoded.
    * Then, the data is split into training and test sets and standardization is done.
    * Different binary classification models (Random forest, XGBoost, KNN, Cat Boost) are tested to find out the best performing model. 
    * The best performing model is selected and feature importance is identified. Least affecting features are removed using a threshold.
    * After the least important features are removed, the model is tuned to find the best hyperparamter using grid search.

* src :