from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from PCA import (airbnb_PCA_n)

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

def decision_tree_grid_search(data, param_grid):
    """returns the parameters that give the best result among the ones in your param_grid"""
    y = data['Price']
    X = data.drop('Price', axis = 1)
    regressor = tree.DecisionTreeRegressor()
    grid = GridSearchCV(regressor, param_grid = param_grid, cv = 10,verbose = 1, n_jobs = -1)
    grid.fit(X,y)
    return grid.best_params_

def Decision_Tree(X_train, X_test, y_train, y_test, max_depth, min_samples_leaf, min_samples_split):
    """returns RMSE and R2 for decision tree"""
    regressor = tree.DecisionTreeRegressor(criterion = 'squared_error',
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    min_samples_split = min_samples_split)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return RMSE, R2

X_train, X_test, y_train, y_test = get_data_split() 

param_grid = {'max_depth':[2,3,4,5,6,8,10,12],"min_samples_split": range(1,10), 'min_samples_leaf':range(1,5)}

#print(decision_tree_grid_search(data, param_grid))  

print(Decision_Tree(X_train, X_test, y_train, y_test, max_depth = 4, min_samples_leaf = 4, min_samples_split = 6))
