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

data = load_preprocessed_data(cleaning = True, missing_value = True, cat_encoding = True,
                           scaling = False, OneHotEncoding = False, LabelEncoding = True)

def get_data_split(data):
    data = data.drop('Postal_Code', axis = 1)
    y = data['Price']
    X = data.drop('Price', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def Grid_search(data):
    X_train, X_test, y_train, y_test = get_data_split(data)
    regressor = tree.DecisionTreeRegressor()
    param_dict = {'max_depth':range(1,10),"min_samples_split": range(1,10), 'min_samples_leaf':range(1,5)}
    grid = GridSearchCV(regressor, param_grid=param_dict, cv = 10,verbose = 1, n_jobs = -1)
    grid.fit(X_train,y_train)
    return grid.best_params_

def Decision_Tree(data):
    X_train, X_test, y_train, y_test = get_data_split(data)

    regressor = tree.DecisionTreeRegressor(criterion = 'squared_error',
    max_depth = 3,
    min_samples_leaf = 1,
    min_samples_split = 2)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    return RMSE

print(Grid_search(data))
print(Decision_Tree(data))