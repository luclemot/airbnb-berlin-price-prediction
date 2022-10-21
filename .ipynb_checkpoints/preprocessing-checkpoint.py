import io
import pandas as pd
import datetime
import numpy as np
import os
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


# load dataset
input_dir = os.path.join(os.getcwd(), "Data")
data_path = os.path.join(input_dir, 'train_airbnb_berlin_off.csv')
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

#df = clean_df(df)

# Deal with missing values

def handle_missing_values(df):
    df.dropna(subset=['Price'],how='any',inplace=True)
    df.dropna(subset=['Listing_ID'],how='any',inplace=True)
    df['Accomodates'] = df['Accomodates'].fillna(2)
    df['Accuracy_Rating'] = df['Accuracy_Rating'].fillna(df['Accuracy_Rating'].mean())
    df['Bedrooms'] = df['Bedrooms'].fillna(1)
    df['Bathrooms'] = df['Bathrooms'].fillna(1)
    df['Beds'] = df['Beds'].fillna(1)
    df['Checkin_Rating'] = df['Checkin_Rating'].fillna(df['Checkin_Rating'].mean())
    df['Cleanliness_Rating'] = df['Cleanliness_Rating'].fillna(df['Cleanliness_Rating'].mean())
    df['Communication_Rating'] = df['Communication_Rating'].fillna(df['Communication_Rating'].mean())
    df['Location_Rating'] = df['Location_Rating'].fillna(df['Location_Rating'].mean())
    df['Guests_Included'] = df['Guests_Included'].fillna(1)
    df['Min_Nights'] = df['Min_Nights'].fillna(1)
    df['Overall_Rating'] = df['Overall_Rating'].fillna(df['Overall_Rating'].mean())
    df['Value_Rating'] = df['Value_Rating'].fillna(df['Value_Rating'].mean())
    df['Host_Response_Rate'] = df['Host_Response_Rate'].fillna(df['Host_Response_Rate'].mean())
    df['Host_Response_Time'] = df['Host_Response_Time'].fillna('a few days or more')
    df['Host_Since'] = df['Host_Since'].fillna(df['Host_Since'].value_counts().idxmax())
    df['Last_Review'] = df['Last_Review'].fillna(df['Last_Review'].value_counts().idxmax())
    df['First_Review'] = df['First_Review'].fillna(df['First_Review'].value_counts().idxmax())
    return df

# df = handle_missing_values(df)

# Transform categorical features 
 
def preprocessing_categorical_features(df):
    dict_Host_Response_Time = {'within an hour':3, 'within a few hours':2, 'within a day':1, 'a few days or more':0}
    df["Host_Response_Time"] = df["Host_Response_Time"].map(dict_Host_Response_Time)
    # preprocess date
    df['date_ref'] = pd.to_datetime('2020-01-01')
    dates_to_preprocess = ['Last_Review', 'First_Review', 'Host_Since']
    for date_to_preprocess in dates_to_preprocess:
        new_name = 'Relative_' + str(date_to_preprocess)
        df[new_name] = (df['date_ref'] - df[date_to_preprocess])/ np.timedelta64(1, 'M')
        df = df.drop(date_to_preprocess, axis = 1)
    df = df.drop('date_ref', axis = 1)
    return df

# df = preprocessing_categorical_features(df)
# print(df)

# Transform categorical features using OneHotEncoding method
categorical_features = ["neighbourhood", "Neighborhood_Group", "Property_Type", "Room_Type"]

def preprocessing_using_OneHotEncoding(df, categorical_features = categorical_features):
    df_categorical_features = df[categorical_features]
    df_categorical_features = pd.get_dummies(df_categorical_features)
    df = pd.concat([df, df_categorical_features], axis=1)
    return df

# df = preprocessing_using_OneHotEncoding(df, categorical_features)
# print(df.columns)

def drop_unnecessary_columns(df):
    to_drop = ['Property_Type', 'Room_Type', 'Property_Type_nan', 'neighbourhood', 'Neighborhood_Group']
    df = df.drop(to_drop, axis = 1)
    return df

# df = drop_unnecessary_columns(df)
# print(len(df.columns))

# Transform categorical features using LabelEncoding method
# categorical_features = ["neighbourhood", "Neighborhood_Group", "Property_Type", "Room_Type"]

# def preprocessing_using_LabelEncoding(df, categorical_features = categorical_features):
#     for feature in categorical_features:
#         le = preprocessing.LabelEncoder()
#         le.fit(df[feature])
#         feature_new_name = 'Label_Encoder_' + str(feature)
#         df[feature_new_name]= le.transform(df[feature])
#         df = df.drop(feature, axis = 1)
#     return df

# df = preprocessing_using_LabelEncoding(df, categorical_features)
