# making utility functions 

# %%
import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
#%%
d = {'col1': [1, 5, 3, 100, 8], 'col2': [3, 4, np.nan, 5, 6]}
df = pd.DataFrame(data=d)


# %%  num nans helper function
def num_nans(df):
    """
    print the number of nans for your dataframe.
    """
    return print(df.isnull().sum())

num_nans(df)


# %%
x = df['col1']
y = df['col2']

X_train, X_test, y_train, y_test = train_test_split(x, y,
 test_size=.2, random_state=42)



 # %%
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

# %%
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)



# %% 
from utilities import train_val_test_spit

# %% 
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)

# %% 
from utilities import num_nans

#%%
from sklearn.model_selection import train_test_split
from utilities import train_val_test_spit

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)