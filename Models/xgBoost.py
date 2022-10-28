import xgboost as xgb


def xgboost(
    X_train,
    Y_train, 
    objective,
    random_state,
    max_depth,
    subsample,
    colsample_bytree,
    learning_rate,
    gamma,
    eta,
    n_estimators
):
    xgb_model = xgb.XGBRegressor(
        objective=objective,
        random_state=random_state,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        gamma=gamma,
        eta=eta,
        n_estimators=n_estimators
    ).fit(X_train, Y_train)
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