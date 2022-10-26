# pandas is used to manipulate tabular data.
import pandas as pd

# statsmodels provides classes and functions for the estimation of many different statistical models.
import statsmodels.api as sm
import itertools
# time provides various time-related functions
import time
from preprocessing_wrapper import (
    load_preprocessed_data
)

# Matplotlib is used to plot graphs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Style options for plots.
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# # Saves a figure to a file
# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join("./figs", fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)

def processSubset(feature_set):
    # Fit OLS (Ordinary Least Squares) model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)].astype(float))
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def forward(features):
    # Pull out features we still need to process
    remaining_features = [d for d in X.columns if d not in features]
    tic = time.time()
    results = []
    for d in remaining_features:
        results.append(processSubset(features+[d]))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(features)+1, "features in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

def getBest(k):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model



data = load_preprocessed_data(cleaning = True, missing_value = True, multivariate_imputation = False, cat_encoding = True,
                           scaling = False, OneHotEncoding = True, LabelEncoding = False)

data = data.drop(['Postal_Code','Listing_ID', 'Host_ID'], axis = 1)
y = data['Price']
X = data.drop('Price', axis = 1)
print(list(X.columns))

features = []
models_fwd = pd.DataFrame(columns=["RSS", "model"])

for i in range(1,len(X.columns)+1):   
    print(features)
    models_fwd.loc[i] = forward(features)
    features = models_fwd.loc[i]["model"].model.exog_names
    
models_best = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
for i in range(1,10):
    models_best.loc[i] = getBest(i)

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

print(models_fwd.loc[1, "model"].summary())
print(models_fwd.loc[2, "model"].summary())