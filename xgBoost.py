import xgboost


def xgboost(X_train, Y_train, **kwargs):
    xgb_model = xgb.XGBRegressor(**kwargs).fit(x_train, y_train)
    return xgb_model


#Set of optimal parameters

#objective="reg:squarederror",
#random_state=42,
#max_depth=5,
#subsample=1,
#colsample_bytree=0.4,
#learning_rate=0.05,
#gamma=0,
#eta=0.1,
#n_estimators=100