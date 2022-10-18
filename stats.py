import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# stats on datas
def print_stats(df): 
    stats_df = pd.DataFrame({
        "min":df.min(numeric_only = True), 
        "max":df.max(numeric_only = True), 
        "mean":df.mean(numeric_only = True),
        "std":df.std(numeric_only = True),
        "median":df.median(numeric_only = True),
        "nunique":df.nunique(), 
        "count_na": df.isna().sum()    
    })
    return stats_df


#HeatMap Correlations


def heat_map(df,figsize=(20,20)):
    corr = df.corr() #Matrice
    plt.figure(figsize=figsize)
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    

#heat_map(df, figsize=(18,18))
