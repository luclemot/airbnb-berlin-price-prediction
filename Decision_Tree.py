from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from preprocessing_wrapper import (
    load_preprocessed_data
)
from PCA import (airbnb_PCA_n)

def get_data_split(data, PCA:bool=False):
    """returns the correct X_train, X_test, y_train, y_test to compute your models"""
    data = data.drop(['Postal_Code','Listing_ID', 'Host_ID'], axis = 1)
    if PCA:
        features =[ 'Host_Response_Time',
 'Host_Response_Rate',
 'Is_Superhost',
 'Latitude',
 'Longitude',
 'Is_Exact_Location',
 'Accomodates',
 'Bathrooms',
 'Bedrooms',
 'Beds',
 'Guests_Included',
 'Min_Nights',
 'Reviews',
 'Overall_Rating',
 'Accuracy_Rating',
 'Cleanliness_Rating',
 'Checkin_Rating',
 'Communication_Rating',
 'Location_Rating',
 'Value_Rating',
 'Instant_Bookable',
 'Business_Travel_Ready',
 'Relative_Last_Review',
 'Relative_First_Review',
 'Relative_Host_Since',
 'neighbourhood_Adlershof',
 'neighbourhood_Alt-HohenschÃ¶nhausen',
 'neighbourhood_Alt-Treptow',
 'neighbourhood_Altglienicke',
 'neighbourhood_Baumschulenweg',
 'neighbourhood_Biesdorf',
 'neighbourhood_Britz',
 'neighbourhood_Buckow',
 'neighbourhood_Charlottenburg',
 'neighbourhood_Dahlem',
 'neighbourhood_Fennpfuhl',
 'neighbourhood_FranzÃ¶sisch Buchholz',
 'neighbourhood_Friedenau',
 'neighbourhood_Friedrichsfelde',
 'neighbourhood_Friedrichshagen',
 'neighbourhood_Friedrichshain',
 'neighbourhood_Grunewald',
 'neighbourhood_Halensee',
 'neighbourhood_Hansaviertel',
 'neighbourhood_Johannisthal',
 'neighbourhood_Karlshorst',
 'neighbourhood_Karow',
 'neighbourhood_Kreuzberg',
 'neighbourhood_KÃ¶penick',
 'neighbourhood_Lankwitz',
 'neighbourhood_Lichtenberg',
 'neighbourhood_Lichtenrade',
 'neighbourhood_Lichterfelde',
 'neighbourhood_Mahlsdorf',
 'neighbourhood_Mariendorf',
 'neighbourhood_Marzahn',
 'neighbourhood_Mitte',
 'neighbourhood_Moabit',
 'neighbourhood_NeukÃ¶lln',
 'neighbourhood_NiederschÃ¶neweide',
 'neighbourhood_NiederschÃ¶nhausen',
 'neighbourhood_Nikolassee',
 'neighbourhood_OberschÃ¶neweide',
 'neighbourhood_Pankow',
 'neighbourhood_PlÃ¤nterwald',
 'neighbourhood_Potsdamer Platz',
 'neighbourhood_Prenzlauer Berg',
 'neighbourhood_Rahnsdorf',
 'neighbourhood_Reinickendorf',
 'neighbourhood_Rudow',
 'neighbourhood_Rummelsburg',
 'neighbourhood_Schmargendorf',
 'neighbourhood_SchmÃ¶ckwitz',
 'neighbourhood_SchÃ¶neberg',
 'neighbourhood_Spandau',
 'neighbourhood_Steglitz',
 'neighbourhood_Tegel',
 'neighbourhood_Tempelhof',
 'neighbourhood_Tiergarten',
 'neighbourhood_Wannsee',
 'neighbourhood_Wedding',
 'neighbourhood_WeiÃ\x9fensee',
 'neighbourhood_Westend',
 'neighbourhood_Wilhelmstadt',
 'neighbourhood_Wilmersdorf',
 'neighbourhood_Wittenau',
 'neighbourhood_Zehlendorf',
 'Property_Type_Apartment',
 'Property_Type_Bed and breakfast',
 'Property_Type_Boat',
 'Property_Type_Boutique hotel',
 'Property_Type_Bungalow',
 'Property_Type_Condominium',
 'Property_Type_Guest suite',
 'Property_Type_Guesthouse',
 'Property_Type_Hostel',
 'Property_Type_Hotel',
 'Property_Type_House',
 'Property_Type_Loft',
 'Property_Type_Other',
 'Property_Type_Serviced apartment',
 'Property_Type_Townhouse',
 'Room_Type_Entire home/apt',
 'Room_Type_Private room',
 'Room_Type_Shared room']
        target = 'Price'
        data = airbnb_PCA_n(data, features, target,80)
    y = data['Price']
    X = data.drop('Price', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def decision_tree_grid_search(data, param_grid):
    """returns the parameters that give the best result among the ones in your param_grid"""
    y = data['Price']
    X = data.drop('Price', axis = 1)
    regressor = tree.DecisionTreeRegressor()
    grid = GridSearchCV(regressor, param_grid = param_grid, cv = 10,verbose = 1, n_jobs = -1)
    grid.fit(X,y)
    return grid.best_params_

def Decision_Tree(data, PCA, max_depth, min_samples_leaf, min_samples_split):
    """returns RMSE and R2 for decision tree"""
    X_train, X_test, y_train, y_test = get_data_split(data, PCA)
    regressor = tree.DecisionTreeRegressor(criterion = 'squared_error',
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    min_samples_split = min_samples_split)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test,y_pred)
    return RMSE, R2


data = load_preprocessed_data(cleaning = True, missing_value = True, multivariate_imputation = False, cat_encoding = True,
                           scaling = False, OneHotEncoding = True, LabelEncoding = False)
    
param_grid = {'max_depth':[2,3,4,5,6,8,10,12],"min_samples_split": range(1,10), 'min_samples_leaf':range(1,5)}

#print(decision_tree_grid_search(data, param_grid))  

print(Decision_Tree(data, PCA = True, max_depth = 4, min_samples_leaf = 4, min_samples_split = 6))
