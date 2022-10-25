from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from preprocessing_wrapper import (
    load_preprocessed_data
)
from stats import (print_stats)

def get_data_split(data):
    """returns the correct X_train, X_test, y_train, y_test to compute your models"""
    data = data.drop('Postal_Code', axis = 1)
    y = data['Price']
    X = data.drop('Price', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def decision_tree_grid_search(data, param_grid):
    """returns the parameters that give the best result among the ones in your param_grid"""
    X_train, X_test, y_train, y_test = get_data_split(data)
    regressor = tree.DecisionTreeRegressor()
    grid = GridSearchCV(regressor, param_grid = param_grid, cv = 10,verbose = 1, n_jobs = -1)
    grid.fit(X_train,y_train)
    return grid.best_params_

def Decision_Tree(data, max_depth, min_samples_leaf, min_samples_split):
    """returns RMSE and R2 for decision tree"""
    X_train, X_test, y_train, y_test = get_data_split(data)
    regressor = tree.DecisionTreeRegressor(criterion = 'squared_error',
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    min_samples_split = min_samples_split)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return RMSE, R2

data = load_preprocessed_data(cleaning = True, missing_value = True, cat_encoding = True,
                           scaling = False, OneHotEncoding = False, LabelEncoding = True)
param_grid = {'max_depth':range(1,10),"min_samples_split": range(1,10), 'min_samples_leaf':range(1,5)}
    
print(decision_tree_grid_search(data, param_grid))
print(Decision_Tree(data))