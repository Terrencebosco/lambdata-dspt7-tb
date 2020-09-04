import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# helper function for dropping columns with nan values. 

def drop_high_nan(df, num_nans):
    '''
    drop columns with preselected number of nans

    df = selected dataframe
    num_nans = the number of nans as a threshold to drop
    '''
    col = df.columns.to_list()
    for col in df:
        if df[col].isnull().sum() > num_nans:
            df = df.drop(col, axis=1)
    return df



def num_nans(df):
    """
    print the number of nans for your dataframe.
    """
    return print(df.isnull().sum())



def train_val_test_spit(x,y,test_size):
    '''
    function will return X_train, X_val, X_test, y_train, y_val, y_test
    inputs from user, x (data without target), y (target vector), and split size.

    invoke: X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                test_size=test_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test



def enlarge(n):
    """
    Param n is a number
    Function will enlarge the number
    """
    return n * 100


def date_splitter(dataframe, date_column_name):
    """
    Takes a passed in dataframe and converts the date feature into a Datetime
    column, then extracts the years, months and days to separate features.
    """
    dataframe[date_column_name] = pd.to_datetime(dataframe[date_column_name],
                                   infer_datetime_format=True)
    dataframe['Year'] = dataframe[date_column_name].dt.year
    dataframe['Month'] = dataframe[date_column_name].dt.month
    dataframe['Day'] = dataframe[date_column_name].dt.day
    dataframe.drop(date_column_name, axis=1, inplace=True)
    return dataframe


def train_validation_test_split(self, df, features, target,
                                train_size=0.7, val_size=0.1,
                                test_size=0.2, random_state=None,
                                shuffle=True):
    '''
    This function is a utility wrapper around the Scikit-Learn train_test_split that splits arrays or 
    matrices into train, validation, and test subsets.

    Args:
        df (Pandas DataFrame): Dataframe with code.
        X (list): A list of features.
        y (str): A string with target column.
        train_size (float or int): Proportion of the dataset to include in the train split (0 to 1).
        val_size (float or int): Proportion of the dataset to include in the validation split (0 to 1).
        test_size (float or int): Proportion of the dataset to include in the test split (0 to 1).
        random_state (int): Controls the shuffling applied to the data before applying the split for reproducibility.
        shuffle (bool): Whether or not to shuffle the data before splitting

    Returns:
        Train, test, and validation dataframes for features (X) and target (y). 
    '''

    X = df[features]
    y = df[target]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
        random_state=random_state, shuffle=shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    
    d = {'col1': [1, 5, 3, 100, 8], 'col2': [3, 4, np.nan, 5, 6]}
    df = pd.DataFrame(data=d)

    x = df['col1']
    y = df['col2']

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)

    print(X_val)

    print(enlarge(7))