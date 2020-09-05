import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split


class MyUtilities(DataFrame):
    """
    Adding additional functionality to pandas DataFrame.
    
    Drop_high_nan: droping columns based on specified number of nan's

    num_nans: print number of nans with a single line.

    data_splitter: split a pre-existing data column into day, month, year columns.

    train_validation_test_split: split data using sklearn library into train, val, 
        test sets.
    """   

    def drop_high_nan(self, num_nans):
        '''
        drop columns with preselected number of nans

        df = selected dataframe
        num_nans = the number of nans as a threshold to drop
        '''
        col = self.columns.to_list()
        for col in self:
            if self[col].isnull().sum() >= num_nans:
                self.drop(col, axis=1, inplace=True)
        return self


    def num_nans(self):
        """
        print the number of nans for your dataframe.
        """
        return print(self.isnull().sum())


    def date_splitter(self, date_column_name):
        """
        Takes a passed in dataframe and converts the date feature into a Datetime
        column, then extracts the years, months and days to separate features.
        """
        self[date_column_name] = pd.to_datetime(
                                self[date_column_name],
                                infer_datetime_format=True
                                )
        self['Year'] = self[date_column_name].dt.year
        self['Month'] = self[date_column_name].dt.month
        self['Day'] = self[date_column_name].dt.day
        self.drop(date_column_name, axis=1, inplace=True)
        return self


    def train_validation_test_split(self, features, target,
                                    train_size=0.7, val_size=0.1,
                                    test_size=0.2, random_state=None,
                                    shuffle=True):
        '''
        This function is a utility wrapper around the Scikit-Learn train_test_split
        that splits arrays or matrices into train, validation, and test subsets.

        Args:
            df (Pandas DataFrame): Dataframe with code.
            X (list): A list of features.

            y (str): A string with target column.

            train_size (float|int): Proportion of data for train split (0 to 1).

            val_size (float|int): Proportion of data for validation split (0 to 1).

            test_size (float or int): Proportion of data for test split (0 to 1).

            random_state (int): Controls the shuffling applied to the data before
                applying the split for reproducibility.

            shuffle (bool): Whether or not to shuffle the data before splitting

        Returns:
            Train, test, and validation dataframes for features (X) and target (y).
        '''

        X = self[features]
        y = self[target]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
            random_state=random_state, shuffle=shuffle)

        return X_train, X_val, X_test, y_train, y_val, y_test
