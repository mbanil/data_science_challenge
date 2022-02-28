import numpy as np

from logger import LOG

def configure_schema(schema, df):
    """Adjust the schema of the new data to that used for training  

    Args:
        schema : schema of the training data
        df: dataframe used for prediction

    Returns:
        schema adjusted dataframe
    """
    LOG.info("Configuring one hot encoding for prediction data")

    cols_original = list(schema["columns"])
    cols_new = list(df.columns)

    for col in cols_new:
       if col.find("\\") != -1:
          df.rename(columns = {col:col.replace("\\", "")}, inplace = True)

    cols_new = list(df.columns)
    for col in cols_original:
       if not col in cols_new:
          df.insert(2, col, np.full(df.shape[0], 0), True)

    df = df.reindex(columns=cols_original)

    return df