import io
import pandas as pd
import datetime
import numpy as np
import os
from sklearn import preprocessing


# load dataset
data_path = "/Users/saadchtouki/Documents/airbnb-berlin-price-prediction-ml-2223/Data/train_airbnb_berlin.csv" ##MODIFICATION DU PATH
df = pd.read_csv(data_path)

# clean dataset
def clean_df(df):
    to_drop = ['Listing Name', 'Host Name', 'City', 'Country Code', 'Country', 'Square Feet']
    df = df.drop(to_drop, axis = 1)
    df.columns = df.columns.str.replace(' ','_')
    df = df.replace('*', np.nan)
    df = df.replace('%', '', regex = True)

    # set data_type
    data_type = {'Listing_ID' : 'float', 'Host_ID' : 'float', 'Host_Since' : 'datetime64', 'Host_Response_Time' : 'string',
        'Host_Response_Rate' : 'float', 'Is_Superhost' : 'bool', 'neighbourhood' : 'str',
        'Neighborhood_Group' : 'str', 'Postal_Code' : 'float', 'Latitude' : 'float', 'Longitude' : 'float',
        'Is_Exact_Location' : 'bool', 'Property_Type' : 'str', 'Room_Type' : 'str', 'Accomodates' : 'float',
        'Bathrooms' : 'float', 'Bedrooms' : 'float', 'Beds' : 'float', 'Guests_Included' : 'float', 'Min_Nights' : 'float',
        'Reviews' : 'float', 'First_Review' :'datetime64', 'Last_Review' : 'datetime64', 'Overall_Rating' : 'float',
        'Accuracy_Rating' : 'float', 'Cleanliness_Rating' : 'float', 'Checkin_Rating' : 'float',
        'Communication_Rating' : 'float', 'Location_Rating' : 'float', 'Value_Rating' : 'float',
        'Instant_Bookable' : 'bool', 'Business_Travel_Ready' : 'bool', 'Price' : 'float'}

    df = df.astype(data_type)
    return df

df = clean_df(df)

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
import seaborn as sns
import matplotlib.pyplot as plt

def heat_map(df,figsize=(20,20)):
    corr = df.corr() #Matrice
    plt.figure(figsize=figsize)
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    

#heat_map(df, figsize=(18,18))
print(print_stats(df))

def donnees_categorielles():
    return

# # Transform categorical features using OneHotEncoding method
# categorical_features = ["neighbourhood", "Neighborhood_Group", "Property_Type", "Room_Type"]

# def preprocess_using_OneHotEncoding(df, categorical_features = categorical_features):
#     dict_Host_Response_Time = {'within an hour':3, 'within a few hours':2, 'within a day':1, 'a few days or more':0}
#     df = df.replace({"Host_Response_Time": dict_Host_Response_Time})
#     df_categorical_features = df[categorical_features]
#     df_categorical_features = pd.get_dummies(df_categorical_features)
#     df = pd.concat([df, df_categorical_features], axis=1)
#     return df

# df = preprocess_using_OneHotEncoding(df, categorical_features)
# print(df.columns)

# def drop_unnecessary_columns(df):
#     to_drop = ['Property_Type', 'Room_Type', 'Property_Type_nan', 'neighbourhood', 'Neighborhood_Group']
#     df = df.drop(to_drop, axis = 1)
#     return df

# df = drop_unnecessary_columns(df)
# print(len(df.columns))

# Transform categorical features using LabelEncoding method
categorical_features = ["neighbourhood", "Neighborhood_Group", "Property_Type", "Room_Type"]

def preprocess_using_LabelEncoding(df, categorical_features = categorical_features):
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(df[feature])
        feature_new_name = 'Label_Encoder_' + str(feature)
        df[feature_new_name]= le.transform(df[feature])
        df = df.drop(feature, axis = 1)
    return df

df = preprocess_using_LabelEncoding(df, categorical_features)
print(df.columns)
