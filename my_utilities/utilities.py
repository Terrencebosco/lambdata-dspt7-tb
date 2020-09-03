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

if __name__ == "__main__":
    
    d = {'col1': [1, 5, 3, 100, 8], 'col2': [3, 4, np.nan, 5, 6]}
    df = pd.DataFrame(data=d)

    x = df['col1']
    y = df['col2']

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)

    print(X_val)

    print(enlarge(7))