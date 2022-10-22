import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from preprocessing_wrapper import load_preprocessed_data
import numpy as np

data = load_preprocessed_data()
features = data.columns.drop('Price')
target = 'Price'

def airbnb_PCA(df, features, target):
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,target].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
    #                           ,columns = ['PC'+str(i) for i in range(1,n+1)]
                            )

    principalDf.index = df.index

    print(principalDf.shape)
    print(df2[target].shape)
    finalDf = pd.concat([principalDf, pd.Series(df2[target])], axis = 1)
    print(finalDf.shape)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    # Plot the cumulative variance to identify optimal n_components
    fig, ax = plt.subplots()
    xi = np.arange(1, 4, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 20, step=50)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

    # Return PCA fields and target variable.
    return(finalDf)

#test = airbnb_PCA(data, features, target)
#print(test)

def airbnb_PCA_n(df, features, target,n):
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,target].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                               ,columns = ['PC'+str(i) for i in range(1,n+1)]
                            )

    principalDf.index = df.index

    print(principalDf.shape)
    print(df2[target].shape)
    finalDf = pd.concat([principalDf, pd.Series(df2[target])], axis = 1)
    print(finalDf.shape)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    # Return PCA fields and target variable.
    return(finalDf)

print(data.loc[5])