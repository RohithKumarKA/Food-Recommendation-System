import streamlit as st
import base64

def hello_page():

    set_background(r"C:\Users\sachu\Downloads\eeab44e6-6249-448d-bcd8-97b445a48d9e.jpg")
    st.write("<h1 style='color:white; font-size: 62px; text-shadow: -1px -1px 0 #FFF, 1px -1px 0 #FFF, -1px 1px 0 #FFF, 1px 1px 0 #FFF;'><strong>Welcome to Food Recommendation System!</strong> 👋</h1>", unsafe_allow_html=True)

    st.sidebar.success("Select a recommendation app.")



    st.markdown(
        """
        <p style='color:white; font-size: 35px; text-shadow: -1px -1px 0 #FFF, 1px -1px 0 #FFF, -1px 1px 0 #FFF, 1px 1px 0 #FFF;'><strong>A food recommendation web application based on budget, location, and nutrition.</strong></p>
        """, unsafe_allow_html=True
    )


def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: auto;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


