# utilities_build.py
import pandas as pd
from sklearn.model_selection import train_test_split


def num_nans(df):
    """
    Produces the number of nans in a DataFrame.

    :param df: an instance of :class: pd.DataFrame
    """
    return df.isnull().sum()


def train_val_test_spit(X, y, test_size):
    '''
    Creates a train-val-test split for the specified DataFrame X and
    target vector y.

    :param X: an instance of :class: pd.DataFrame
    :param y: an instance of :class: pd.Series
    :param test_size: a float, where 0 <= test_size < 1
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                test_size=test_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
