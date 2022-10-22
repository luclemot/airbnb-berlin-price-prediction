import os
import pandas as pd

from preprocessing import (
    clean_df,
    handle_missing_values,
    preprocessing_categorical_features,
    preprocessing_using_OneHotEncoding,
    preprocessing_using_LabelEncoding,
    multivariate_feature_imputation
)
from scaling import apply_scaling


# df columns to standardize
to_standardize = [
    'Latitude', 'Longitude',
    'Accomodates', 'Guests_Included',
    'Bathrooms', 'Bedrooms', 'Beds',
    'Min_Nights', 
    'Reviews'
]

# df columns to minmax scale
to_minmax = [
    'Overall_Rating',
    'Accuracy_Rating',
    'Cleanliness_Rating',
    'Checkin_Rating',
    'Communication_Rating',
    'Location_Rating',
    'Value_Rating'
]

# Below limits are chosen to set all the non-categorical numerical features close to [0,1]
borne_inf = 0
borne_sup = 1


def load_preprocessed_data(cleaning:bool=True, missing_value:bool=True, cat_encoding:bool=True,
                           scaling:bool=True, pca:bool=True, OneHotEncoding:bool = True, LabelEncoding:bool = False):
    """Global wrapper executing all the preprocessing code to apply to the loaded dataset.
    
    :param cleaning: bool - if True, execute the preprocessing code dealing wtth dataframe cleaning
    :param missing_value: bool - if True, execute the preprocessing code dealing with missing value handling
    :param cat_encoding: bool - if True, execute the preprocessing code dealing with categorical features encoding
    :param scaling: bool - if True, execute the preprocessing code dealing with numerical features scaling
    :param pca: bool - if True, execute the preprocessing code dealing with PCA
    """
    input_dir = os.path.join(os.getcwd(), "Data")
    data_path = os.path.join(input_dir, 'train_airbnb_berlin_off.csv')
    data = pd.read_csv(data_path)
    
    if cleaning:
        data = clean_df(data)
    
    if scaling:
        data = apply_scaling(data, to_standardize, to_minmax, borne_inf, borne_sup)

    if missing_value:
        data = handle_missing_values(data) #Handles all missing values except ratings
        data=multivariate_feature_imputation(data,['Overall_Rating','Accuracy_Rating','Cleanliness_Rating','Checkin_Rating','Communication_Rating','Location_Rating','Value_Rating'])
        #The line above handles ratings
        
    if cat_encoding:
        data = preprocessing_categorical_features(data)
        if OneHotEncoding:
            data = preprocessing_using_OneHotEncoding(data)
        elif LabelEncoding:
            data = preprocessing_using_LabelEncoding(data)
        None
    
    
    if pca:
        # TODO: apply the PCA here if needed
        None
    
    return data
