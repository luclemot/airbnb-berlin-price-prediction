import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from Random_Forest import (
    Random_Forest
)


def get_data_split():
    """returns the correct X_train, X_test, y_train, y_test to compute your models"""
    train = pd.read_csv('./Data/Stratified_train_set.csv')
    test = pd.read_csv('./Data/Stratified_test_set.csv')

    train = train.drop(['Listing_ID', 'Host_ID', 'Unnamed: 0', 'Postal_Code'], axis = 1)
    test = test.drop(['Listing_ID', 'Host_ID', 'Unnamed: 0', 'Postal_Code'], axis = 1)

    y_train = train['Price']
    X_train = train.drop('Price', axis = 1)
    y_test = test['Price']
    X_test = test.drop('Price', axis = 1)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_data_split() 

regressor = Random_Forest(X_train, X_test, y_train, y_test, max_depth = 49 , max_features = 'sqrt', n_estimators = 900)

y_pred = regressor.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test,y_pred)

print('Pour le Random Forest on obtient')
print('RMSE :', RMSE)
print('RMSE :', R2)