# making utility functions 

# %%
import pandas as pd
import numpy as np

#%%
d = {'col1': [1, 5, 3, 7, 8], 'col2': [3, 4, np.nan, 5, 6]}
df = pd.DataFrame(data=d)


# %%  num nans helper function
def num_nans(df):
    """
    print the number of nans for your dataframe.
    """
    return print(df.isnull().sum())

num_nans(df)


# %%
