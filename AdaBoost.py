import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
from PCA import airbnb_PCA_n
from preprocessing_wrapper import load_preprocessed_data
from Cross_validation import cross_validation
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

"""
Ce script est composé de deux fonctions. La première, searchAdaBoost_parameters, est faite pour déterminer les paramètres optimaux pour un algorithme AdaBoost.
La première fonction peut mettre 20 minutes à s'éxecuter, c'est pourquoi les paramètres optimaux (résultats) sont annotés au niveau de celle-ci.
A partir des paramètres retournés par la fct searchAdaBoost_parameters, on peut appliquer la fct adaBoost pour entrainer notre modèle.
"""

def searchAdaBoost_parameters(pca:bool=True): #the execution can take 20 minutes // Result : Best_params:{n_estimators:50, learning_rate:0.0005}
    data = load_preprocessed_data()
    data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])
    X = data.drop(columns=['Price'])
    Y = data[['Price']]
    # Create the training and testing set
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.3)
    if pca:
        x_train, x_test = airbnb_PCA_n(x_train, x_test, 80)
    #Create AdaBoost Regressor
    ada=AdaBoostRegressor()
    #Parameters to test for a GridSearch
    search_grid={'n_estimators':[50,100,200,500],'learning_rate':[0.0001,0.0005,.001,0.01,0.1,1],'random_state':[1]}
    
    grid_search=GridSearchCV(estimator=ada,param_grid=search_grid,scoring='neg_mean_squared_error')
    grid_search.fit(x_train,y_train)
    print('Score best estimator : ',grid_search.best_estimator_.score(x_test,y_test))
    print('params : ',grid_search.best_params_)
    pred = grid_search.best_estimator_.predict(x_test)
    rmse = metrics.mean_squared_error(y_test, pred, squared = False)
    print('rmse :',rmse)
    
def adaBoost(nb_estimators,learn_rate, pca:bool=True):
    data = load_preprocessed_data(False,False,False,False,False,False,False)
    data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])
    X = data.drop(columns=['Price'])
    Y = data[['Price']]
    #Create AdaBoost Regressor
    ada=AdaBoostRegressor(n_estimators=nb_estimators, learning_rate=learn_rate)
    #Apply Cross validation
    return cross_validation(ada,5,X,Y,pca)

#adaBoost(50,0.0005,True)