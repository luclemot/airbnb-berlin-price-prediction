import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from Random_Forest import Random_Forest
from xgBoost import xgboost


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


X_train, X_test, y_train, y_test = get_data_split() 

regressor_rf = Random_Forest(X_train, X_test, y_train, y_test, max_depth = 49 , max_features = 'sqrt', n_estimators = 900)
y_pred_rf = regressor_rf.predict(X_test)
RMSE_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
R2_rf = r2_score(y_test,y_pred_rf)

        
regressor_xgb = xgboost(
    objective="reg:squarederror",
    random_state=42,
    max_depth=5,
    subsample=1,
    colsample_bytree=0.4,
    learning_rate=0.05,
    gamma=0,
    eta=0.1,
    n_estimators=100
)
y_pred_xgb = regressor_xgb.predict(X_test)
RMSE_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
R2_xgb = r2_score(y_test,y_pred_xgb)


print('Pour le Random Forest on obtient')
print('RMSE :', RMSE_rf)
print('RMSE :', R2_rf)

print('Pour le XGBoost on obtient')
print('RMSE :', RMSE_xgb)
print('RMSE :', R2_xgb)