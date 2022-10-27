from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

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

def Grid_search(train, param_grid):
    """returns the parameters that give the best result among the ones in your param_grid"""    
    # Create the parameter grid based on the results of random search 
    y = train['Price']
    X = train.drop('Price', axis = 1)
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X, y)
    return grid_search.best_params_

def Random_Forest(X_train, X_test, y_train, y_test, max_depth, max_features, n_estimators):
    """returns RMSE and R2 for random forest"""
    X_train, X_test, y_train, y_test = get_data_split()

    regressor = RandomForestRegressor(bootstrap = True,
                max_depth = max_depth,
                max_features = max_features,
                n_estimators = n_estimators)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return RMSE, R2


param_grid = {
        'max_depth': [44, 45, 46, 47, 48, 49],
        'n_estimators': [850, 875, 900, 925]
    }

X_train, X_test, y_train, y_test = get_data_split() 

#print(Grid_search(train, param_grid))
print(Random_Forest(X_train, X_test, y_train, y_test, max_depth = 49 , max_features = 'sqrt', n_estimators = 900))
