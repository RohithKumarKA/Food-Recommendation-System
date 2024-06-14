import streamlit as st
from ImageFinder.ImageFinder import get_images_links as find_image
import pandas as pd
from streamlit_echarts import st_echarts
import pickle  # Import pickle for loading the model
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import base64


# Load the model from a pickle file
model_path = r"C:\Users\sachu\OneDrive\Desktop\Project\your_model4.pkl"  # Update this path to where your model is stored
loaded_data = pickle.load(open(model_path, 'rb'))
imputer = loaded_data['imputer']
scaler = loaded_data['scaler']
feature_weights = loaded_data['feature_weights']
location_encoder = loaded_data['location_encoder']  # Load location_encoder
model=loaded_data['neigh']

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            position: relative;
        }}
        .form-container {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(255, 255, 255, 0.8); /* Adjust the opacity as needed */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)




def custom_food_recommendations_page(model):
    
    if 'generated' not in st.session_state:
        st.session_state.generated = False
        st.session_state.recommendations = None
        st.session_state.budget = None  # Initialize budget in session state
        st.session_state.location = None  # Initialize location in session state
    
 
    def transform_input_data(input_data, imputer, scaler, feature_weights, location_encoder):
        # Assuming input_data is a 2D array with the nutritional values
        # Include budget and location as part of the input data 
        
        # Retrieve budget and location from session state
        budget = st.session_state.budget  # Assuming budget is stored in session state
        location = st.session_state.location  # Assuming location is stored in session state
        
        # Check if location is not None and is not an empty string
        if location and location.strip()!= '':
            # Convert budget to an array if necessary
            #budget = np.array([budget]).reshape(-1, 1)
            
            # Encode the location data
            encoded_location = location_encoder.transform([location])  # Use transform instead of fit_transform for new data
            #encoded_location = encoded_location.reshape(-1, 1)
            print(encoded_location)
            nutrition_input=[Calories,FatContent,SaturatedFatContent,CholesterolContent,SodiumContent,CarbohydrateContent,FiberContent,SugarContent,ProteinContent,int(encoded_location),int(budget)]
            print(nutrition_input)
            
            # Combine nutritional values, budget, and encoded location
            #combined_data = np.hstack((input_data, budget, encoded_location))
            #print("Combined data shape:", combined_data.shape)  # Debugging statement
        else:
            # Handle the case where location is not available
            print("Warning: Location data is not available.")
            # Optionally, return early or provide a default value for combined_data
            return None
        nutrition_input_array = np.array(nutrition_input)
        print(nutrition_input_array.shape)
        print("Hello")
        
        nutrition_input_array = np.array(nutrition_input).reshape(1, -1)
        # Apply imputation
        input_data_transformed = imputer.transform(nutrition_input_array)
        
        # Apply scaling with feature weights
        input_data_transformed = scaler.transform(input_data_transformed * feature_weights)
        
        return input_data_transformed






    nutrition_values=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']
    if 'generated' not in st.session_state:
        st.session_state.generated = False
        st.session_state.recommendations=None
    
    data=pd.read_excel(r"C:\Users\sachu\Downloads\newest p.xlsx")

    
    class Display:
        def __init__(self):
            self.nutrition_values = nutrition_values
            
        def display_recommendation(self, recommendations):
            st.subheader('Recommended recipes:')
            if recommendations is not None:
                for index in recommendations:
                    recommended_item = data.iloc[index]  # Access the row from the original dataset
                    recipe_name = recommended_item['Name']
                    expander = st.expander(recipe_name)
                    
                    nutritions_df = pd.DataFrame({value: [recommended_item[value]] for value in self.nutrition_values})
                    expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>', unsafe_allow_html=True)
                    expander.dataframe(nutritions_df)
                    
                    expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>', unsafe_allow_html=True)
                    # Preprocess the ingredients string
                    ingredients_string = recommended_item['RecipeIngredientParts']
                    ingredients_list = [ingredient.strip().strip('"') for ingredient in ingredients_string.replace('c(', '').replace(')', '').split(', ') if ingredient.strip()]
                    for i, ingredient in enumerate(ingredients_list, 1):
                        expander.markdown(f"{i}. {ingredient}")  # Display ingredients point-wise
                    
                    expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>', unsafe_allow_html=True)
                    # Preprocess the instructions string
                    instructions_string = recommended_item['RecipeInstructions']
                    # Remove "c()" and double quotes, split into individual instructions
                    instructions_list = [instruction.strip().strip('"') for instruction in instructions_string.replace('c(', '').replace(')', '').split('", "') if instruction.strip()]
                    for i, instruction in enumerate(instructions_list, 1):
                        expander.markdown(f"{i}. {instruction.strip()}")

            else:
                st.info('Couldn\'t find any recipes with the specified ingredients', icon="🙁")
        '''''
        def display_overview(self, recommendations):
            if recommendations is not None:
                recipe_names = []
                selected_recipe = None  # Initialize selected_recipe variable
                for index in recommendations:
                    recommended_item = data.iloc[index]
                    recipe_names.append(recommended_item['Name'])
                st.subheader('Overview:')
                col1, col2, col3 = st.columns(3)
                with col2:
                    selected_recipe_name = st.selectbox('Select a recipe', recipe_names)
                    # Find the selected recipe
                    for index in recommendations:
                        recommended_item = data.iloc[index]
                        if recommended_item['Name'] == selected_recipe_name:
                            selected_recipe = recommended_item
                            break
                    if not selected_recipe.empty:
                        st.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values:</h5>', unsafe_allow_html=True)
                        options = {
                            "title": {"text": "Nutrition values", "subtext": f"{selected_recipe_name}", "left": "center"},
                            "tooltip": {"trigger": "item"},
                            "legend": {"orient": "vertical", "left": "left",},
                            "series": [
                                {
                                    "name": "Nutrition values",
                                    "type": "pie",
                                    "radius": "50%",
                                    "data": [{"value":selected_recipe[value],"name":value} for value in self.nutrition_values],
                                    "emphasis": {
                                        "itemStyle": {
                                            "shadowBlur": 10,
                                            "shadowOffsetX": 0,
                                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                                        }
                                    },
                                }
                            ],
                        }
                        st_echarts(options=options, height="600px",)
                        st.caption('You can select/deselect an item (nutrition value) from the legend.')
'''
        def display_overview(self, recommendations):
                if recommendations is not None:
                    recipe_names = []
                    for index in recommendations:
                        recommended_item = data.iloc[index]  # Assuming data is defined somewhere
                        recipe_names.append(recommended_item['Name'])
                        nutritions_df = pd.DataFrame({value: [recommended_item[value]] for value in self.nutrition_values})
                            
                    st.subheader('Overview:')
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        selected_recipe_name = st.selectbox('Select a recipe', recipe_names)
                            
                    st.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values:</h5>', unsafe_allow_html=True)
                    for recipe in recipe_names:
                        if recipe==selected_recipe_name:
                            selected_recipe=recommended_item
                            
                    options = {
                "title": {"text": "Nutrition values", "subtext": f"{selected_recipe_name}", "left": "center"},
                "tooltip": {"trigger": "item"},
                "legend": {"orient": "vertical", "left": "left",},
                "series": [
                    {
                        "name": "Nutrition values",
                        "type": "pie",
                        "radius": "50%",
                        "data": [{"value":selected_recipe[value],"name":value} for value in self.nutrition_values],
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)",
                            }
                        },
                    }
                ],
            }
                    st_echarts(options=options, height="600px",)
                    st.caption('You can select/deselect an item (nutrition value) from the legend.')

        title="<h1 style='text-align: center;'>Custom Food Recommendation</h1>"
        st.markdown(title, unsafe_allow_html=True)


    display=Display()
    with st.form("recommendation_form"):
         # Add text input fields for budget and location
        st.session_state.budget = st.text_input("Budget", "Enter your budget")
        st.session_state.location = st.text_input("Location", "Enter your location")
        st.header('Nutritional values:')
        Calories = st.slider('Calories', 0, 2000, 500)
        FatContent = st.slider('FatContent', 0, 100, 50)
        SaturatedFatContent = st.slider('SaturatedFatContent', 0, 13, 0)
        CholesterolContent = st.slider('CholesterolContent', 0, 300, 0)
        SodiumContent = st.slider('SodiumContent', 0, 2300, 400)
        CarbohydrateContent = st.slider('CarbohydrateContent', 0, 325, 100)
        FiberContent = st.slider('FiberContent', 0, 50, 10)
        SugarContent = st.slider('SugarContent', 0, 40, 10)
        ProteinContent = st.slider('ProteinContent', 0, 40, 10) 
        nutritions_values_list=[Calories,FatContent,SaturatedFatContent,CholesterolContent,SodiumContent,CarbohydrateContent,FiberContent,SugarContent,ProteinContent]
        st.header('Recommendation options (OPTIONAL):')
        nb_recommendations = st.slider('Number of recommendations', 5, 20,step=5)
        st.caption('Example: Milk;eggs;butter;chicken...')
        
        
        
        generated = st.form_submit_button("Generate")
 
    if generated:
        if st.session_state.budget == "" or st.session_state.budget == "Enter your budget":
            st.error("ENTER A VALID BUDGET")
        elif st.session_state.location == "" or st.session_state.location == "Enter your location":
            st.error("ENTER A VALID LOCATION")
        elif st.session_state.location not in location_encoder.classes_:
            st.error("LOCATION NOT FOUND")
        else:    
            with st.spinner('Generating recommendations...'): 
                # Transform input data
                nutrition_input = nutritions_values_list
                transformed_data = transform_input_data(nutrition_input, imputer, scaler, feature_weights, location_encoder)
                if transformed_data is not None:  # Check if location was found
                    recommendations = model.kneighbors(transformed_data, n_neighbors=nb_recommendations, return_distance=False)[0]
                    st.session_state.recommendations = recommendations.tolist()  # Convert to list for compatibility
                    st.session_state.generated = True
                else:
                    st.error("LOCATION NOT FOUND")

      
    

    
    # Display recommendations if they exist
    if st.session_state.generated and st.session_state.recommendations is not None:
        display.display_recommendation(st.session_state.recommendations)

    # Optionally, display an overview of the recommendations
    if st.button('Show Overview'):
        display.display_overview(st.session_state.recommendations)