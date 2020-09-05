from my_lambda.my_classes import MyUtilities
import numpy as np
import pandas as pd

if __name__ == "__main__":
    d = {'col1': [1, 5, 3, 100, 8], 'col2': [3, 4, np.nan, 5, 6]}
    df = pd.DataFrame(data=d)

    x = df['col1']
    y = df['col2']

    dt = MyUtilities(df)

    print(dt.num_nans())
        
    X_train, X_val, X_test, y_train, y_val, y_test = dt.train_validation_test_split('col1','col2',random_state=42)

    print(X_val)

    print(dt.shape)

    print(dt.drop_high_nan(1))