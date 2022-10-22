# Import block
from preprocessing import df
from stats import print_stats, heat_map

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Create X and Y variables
X= df.drop(columns = ["Price","Postal_Code"])
Y= df[["Price"]]

# Create the train and test set, without stratifying.
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
x_train_strat, x_test_strat, y_train_strat, y_test_strat = train_test_split(X, Y, stratify = X['Accomodates'], test_size = 0.2)

# Check for disproportion in field of choice distribution
def accomodation_proportions(data, field):
    return data[field].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Input_dataset": accomodation_proportions(X, 'Accomodates'),
    "Test_set": accomodation_proportions(x_test, 'Accomodates'),
    "Strat_set": accomodation_proportions(x_test_strat, 'Accomodates')
}).sort_index()
compare_props["Test set. %error"] = 100 * compare_props["Test_set"] / compare_props["Input_dataset"] - 100
compare_props["Strat test set. %error"] = 100 * compare_props["Strat_set"] / compare_props["Input_dataset"] - 100

# print(compare_props)

# Rename training and testing set as stratified dfs

# x_train, x_test, y_train, y_test = x_train_strat, x_test_strat, y_train_strat, y_test_strat

# Create regressor

from sklearn.linear_model import LinearRegression
from sklearn import metrics
clf = LinearRegression()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

acc = clf.score(x_test,y_test)
print(acc)
mse = metrics.mean_squared_error(y_test, pred)
print(mse)


"""En entrée, prendre un training set et un testing set.
En sortie, rendre la prédiction, et le mse. + autres métriques?"""
