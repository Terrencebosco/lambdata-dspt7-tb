from utilities import train_val_test_spit, num_nans
from sklearn.model_selection import train_test_split


d = {'col1': [1, 5, 3, 100, 8], 'col2': [3, 4, np.nan, 5, 6]}
df = pd.DataFrame(data=d)

num_nans(df)

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_spit(x, y, .2)

x = df['col1']
y = df['col2']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
