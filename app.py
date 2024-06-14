import streamlit as st
from Hello1 import hello_page
from Custom_Food_Recommendation1 import custom_food_recommendations_page
import pickle
import numpy as np
# Define navigation options
pages = {
    "Hello": hello_page,
    "Custom Food Recommendation": custom_food_recommendations_page
}

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))

    # Load the model from a pickle file
   # model_path = r"C:\Users\sachu\OneDrive\Desktop\Project\your_model.pkl"   # Update this path to where your model is stored
   # model = pickle.load(open(model_path, 'rb'))
    with open(r"C:\Users\sachu\OneDrive\Desktop\Project\your_model.pkl", 'rb') as file:
        loaded_data = pickle.load(file)

    imputer = loaded_data['imputer']
    scaler = loaded_data['scaler']
    feature_weights = loaded_data['feature_weights']
    model = loaded_data['model']

    # Call the selected page function, passing the model if necessary
    if selected_page == "Custom Food Recommendation":
        pages[selected_page](model)
    else:
        pages[selected_page]()

if __name__ == "__main__":
    main()
