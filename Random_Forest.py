from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from preprocessing_wrapper import (
    load_preprocessed_data
)
from PCA import (airbnb_PCA_n)

def get_data_split(data):
    """returns the correct X_train, X_test, y_train, y_test to compute your models"""
    data = data.drop(['Postal_Code','Listing_ID', 'Host_ID', 'neighbourhood_nan'], axis = 1)
    y = data['Price']
    X = data.drop('Price', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def Grid_search(data, param_grid):
    """returns the parameters that give the best result among the ones in your param_grid"""
    X_train, X_test, y_train, y_test = get_data_split(data)
    # Create the parameter grid based on the results of random search 
    
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def Random_Forest(data, max_depth, max_features, n_estimators):
    """returns RMSE and R2 for random forest"""
    X_train, X_test, y_train, y_test = get_data_split(data)

    regressor = RandomForestRegressor(bootstrap = True,
                max_depth = max_depth,
                max_features = max_features,
                n_estimators = n_estimators)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return RMSE, R2

data = load_preprocessed_data(cleaning = True, missing_value = True, multivariate_imputation = False, cat_encoding = True,
                           scaling = False, OneHotEncoding = True, LabelEncoding = False)
param_grid = {
        'max_depth': [40, 42, 43, 44, 45, 46, 47, 48, 49],
        'n_estimators': [850, 875, 900, 925]
    }
print(Random_Forest(data, max_depth = 49 , max_features = 'sqrt', n_estimators = 875))
