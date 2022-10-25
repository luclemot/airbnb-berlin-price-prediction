import os
import pandas as pd
from sklearn.model_selection import KFold
import statistics as st
from sklearn import metrics
from PCA import airbnb_PCA_n

def cross_validation(clf,K,X,Y,pca:bool=False):
    cv=KFold(K)
    accu=[]
    mse=[]
    for train_index, test_index in cv.split(X):
        x_train, y_train = X.iloc[train_index], Y.iloc[train_index]
        x_test, y_test = X.iloc[test_index], Y.iloc[test_index]
        if pca:
            x_train, x_test = airbnb_PCA_n(x_train, x_test, 80)
        clf.fit(x_train, y_train)
        pred=clf.predict(x_test)
        accu.append(clf.score(x_test,y_test))
        mse.append(metrics.mean_squared_error(y_test, pred, squared = False))
    print("Accuracy for each fold :")
    print(accu)
    print("Mean of accuracy :",st.mean(accu))
    print(" ")
    print("Mse for each fold :")
    print(mse)
    print("Mean of mse :",st.mean(mse))
    return (st.mean(accu),st.mean(mse))