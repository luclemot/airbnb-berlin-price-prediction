# Import block
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

from preprocessing_wrapper import load_preprocessed_data

data = load_preprocessed_data()
data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])
# Create X and Y, the target value from data
X = data.drop(columns=['Price'])
Y = data[['Price']]

def stratify(X, Y, field):
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
    x_train_strat, x_test_strat, y_train_strat, y_test_strat = train_test_split(X, Y, stratify = X[field], test_size = 0.2)
    def accomodation_proportions(data, field):
        return data[field].value_counts() / len(data)
    compare_props = pd.DataFrame({
    "Input_dataset": accomodation_proportions(X, 'Accomodates'),
    "Test_set": accomodation_proportions(x_test, 'Accomodates'),
    "Strat_set": accomodation_proportions(x_test_strat, 'Accomodates')
    }).sort_index()
    compare_props["Test set. %error"] = 100 * compare_props["Test_set"] / compare_props["Input_dataset"] - 100
    compare_props["Strat test set. %error"] = 100 * compare_props["Strat_set"] / compare_props["Input_dataset"] - 100
    print(compare_props)
    return(x_train_strat, x_test_strat, y_train_strat, y_test_strat)


def Reg(stratify:bool=False, field:str=None):
    # Create the training and testing set
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
    if stratify and field!=None:
        x_train, x_test, y_train, y_test = stratify(X,Y,field)
    # Create regressor
    clf = LinearRegression()
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

Reg()