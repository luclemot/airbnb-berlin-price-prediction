import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from preprocessing_wrapper import load_preprocessed_data
import numpy as np

data = load_preprocessed_data()
data = data.drop(columns = ["Listing_ID", "Host_ID", "Postal_Code"])
features = data.columns.drop('Price')
target = 'Price'



"""
Ce script est composé de deux fonctions. La première, airbnb_PCA, est faite pour déterminer le nombre optimal
de PCA components. Pour que le code finisse de tourner, il faut fermer le graphe qui s'ouvrira sur votre IDE.
A partir du graphe, choisir la valeur n (dès que la variance cumulative expliquée dépasse la barre des 95%.
Appliquer la seconde fonction aribnb_PCA_n pour obtenir les coordonnées PCA."""


def airbnb_PCA(df, features, target):
    df.dropna(inplace=True)
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,target].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # Create the PCA regressor with a maximum number of components
    pca = PCA()
    PrincipalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = PrincipalComponents)
    # Ensure that the PCA values will merge to the target values
    principalDf.index = df.index
    # Concatenante the PCA components values to the target balues
    finalDf = pd.concat([principalDf, pd.Series(df[target])], axis = 1)
    # Create the explained variance
    pca2 = pca.fit(x)
    explained_variance = pca2.explained_variance_ratio_
    print(explained_variance)
    # Plot the cumulative variance to identify optimal n_components
    plt.plot(np.cumsum(pca2.explained_variance_ratio_), linewidth=2)
    plt.xlabel('Components')
    plt.ylabel('Cumulative sum of explained Variances')
    plt.show()

airbnb_PCA(data, features, target)

def airbnb_PCA_n(df, features, target,n):
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,target].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # Create the PCA regressor with the optimal number of components n
    pca = PCA(n)
    principalComponents = pca.fit_transform(x)
    # Create the new dataframe
    principalDf = pd.DataFrame(data = principalComponents
                               ,columns = ['PC'+str(i) for i in range(1,n+1)]
                            )
    # Ensure that the PCA values will merge to the target values      
    principalDf.index = df.index
    # Concatenante the PCA components values to the target balues
    finalDf = pd.concat([principalDf, pd.Series(df[target])], axis = 1)
    print(finalDf.shape)
    # Create the explained variance
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    # Return PCA fields and target variable.
    return(finalDf)

test = airbnb_PCA_n(data, features, target, 80)
print(test)