import pandas as pd
import numpy as np
import pickle




data=pd.read_excel(r"C:\Users\sachu\Downloads\p.xlsx")




dataset=data.copy()
columns = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeIngredientParts', 
            'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 
            'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'Price']
dataset=dataset[columns]



max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200
max_budget = 50  # Example maximum budget value

max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol,
            max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar,
            max_daily_Protein, max_budget]  # Add max_budget to the list


extracted_data = data[columns]
    # Filter out rows based on the maximum values (excluding 'Location' column)
numerical_columns = extracted_data.columns[6:15]
for column, maximum in zip(numerical_columns, max_list):
    if column != 'Location':
        extracted_data = extracted_data[extracted_data[column] < maximum]


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np

location_encoder = LabelEncoder()
data['Location_encoded'] = location_encoder.fit_transform(data['Location'])
extracted_data['Location'] = data['Location_encoded']

numeric_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 
                        'ProteinContent','Location_encoded', 'Price']

        # Select the columns for preprocessing
preprocessed_data = data[numeric_columns]

        # Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
preprocessed_data_imputed = imputer.fit_transform(preprocessed_data)



        # Add the encoded location to the preprocessed data
        #preprocessed_data_imputed = np.column_stack((preprocessed_data_imputed, data['Location_encoded'])) 

feature_weights = [1, 1, 1, 1, 1, 1, 1, 1, 0.001, 1, 1]  # Add one more weight
feature_weights = np.array(feature_weights).reshape(1, -1) 


        # Standardize numeric columns with weighted scaling for Price
scaler = StandardScaler()
preprocessed_data_imputed = scaler.fit_transform(preprocessed_data_imputed * feature_weights) 


        # Model Modification
        #neigh = NearestNeighbors(metric='cosine', algorithm='brute')
        #neigh = NearestNeighbors(metric='euclidean', algorithm='kd_tree') # Or use 'manhattan'
neigh = NearestNeighbors(metric='manhattan', algorithm='brute')


neigh.fit(preprocessed_data_imputed)

def YourModel(input_data):
    
    # Find nearest neighbors
    recommended_indices = neigh.kneighbors(input_data, n_neighbors=3, return_distance=False)[0]

    # Retrieve recommended items from the dataset
    recommended_items = data.iloc[recommended_indices]
    return recommended_items

with open('your_model1.pkl', 'wb') as file:
            pickle.dump({
                'imputer': imputer,
                'scaler': scaler,
                'feature_weights': feature_weights,
                'location_encoder': location_encoder,
                'model': neigh
            }, file)


