# Import block
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import numpy as np

from PCA import airbnb_PCA_n
from preprocessing_wrapper import load_preprocessed_data
from Cross_validation import cross_validation

"""
Ce script est composé de deux fonctions. La première, knn, est faite pour déterminer le k optimal.
Pour que le code finisse de tourner, il faut fermer le graphe qui s'ouvrira sur votre IDE.
A partir du graphe, choisir la valeur k (où la mse est minimale).
Appliquer la seconde fonction knn_n pour obtenir la régression en k nearest neighbors.
"""

data = load_preprocessed_data()
data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])

# Create X and Y, the target value from data
X = data.drop(columns=['Price'])
Y = data[['Price']]

features = data.columns.drop("Price")
target = 'Price'

def knn():
    # Create the training and testing set
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
    error = []
    for i in range(1, 40):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        mse = metrics.mean_squared_error(y_test, pred_i, squared = False)
        error.append(mse)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', 
            linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('K Value kMSE')
    plt.xlabel('K Value')
    plt.ylabel('Root Mean Squared Error')
    plt.show()

#knn()

def knn_n(n, pca:bool=False):
    data = load_preprocessed_data()
    data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])
    X = data.drop(columns=['Price'])
    Y = data[['Price']]
    # Create the training and testing set
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
    if pca:
        x_train, x_test = airbnb_PCA_n(x_train, x_test, 80)
    # Create regressor
    clf = KNeighborsRegressor(n)
    # Fit on the training set
    clf.fit(x_train, y_train)
    # Create predictions
    pred = clf.predict(x_test)
    # Print the accuracy and root mean square error
    acc = clf.score(x_test,y_test)
    print(acc)
    rmse = metrics.mean_squared_error(y_test, pred, squared = False)
    print(rmse)
    return(rmse)

print(knn_n(8, True))

print(cross_validation(KNeighborsRegressor(8), 5, X, Y, pca = True))