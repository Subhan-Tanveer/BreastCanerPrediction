import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the trained model
loaded_model = pickle.load(open('D:/AI Classification Projects/Breast Cancer Prediction/trained_tumour.pkl', 'rb'))

# Function for user input with very simple labels
def user_input_features():

    page_bg_img = """

    <style>  
    [data-testid= "stMain"]{
    background-image: url("https://news.harvard.edu/wp-content/uploads/2022/09/20220908_cancercells_2500.jpg");
    background-size: cover;
    }

    [data-testid= "stHeader"]{
    background-color:rgba(0,0,0,0);
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.sidebar.header("Enter Tumour Details")

    # Sidebar inputs for the user with even simpler labels
    size = st.sidebar.slider('Tumour Size (radius)', 10.0, 30.0, 17.99)
    surface = st.sidebar.slider('Tumour Surface (texture)', 10.0, 30.0, 10.38)
    edge = st.sidebar.slider('Tumour Edge (perimeter)', 50.0, 200.0, 122.80)
    area = st.sidebar.slider('Tumour Area (size)', 100.0, 2000.0, 1001.0)
    smoothness = st.sidebar.slider('Tumour Smoothness', 0.0, 1.0, 0.11840)
    tightness = st.sidebar.slider('Tumour Tightness (compactness)', 0.0, 1.0, 0.27760)
    indentation = st.sidebar.slider('Tumour Indentation (concavity)', 0.0, 1.0, 0.3001)
    dip_points = st.sidebar.slider('Tumour Dip Points', 0.0, 1.0, 0.14710)
    balance = st.sidebar.slider('Tumour Balance (symmetry)', 0.0, 1.0, 0.2419)
    roughness = st.sidebar.slider('Tumour Roughness', 0.0, 1.0, 0.07871)

    # Standard Error Inputs (these are additional details about the tumour)
    error_size = st.sidebar.slider('Size Error', 0.0, 5.0, 0.4066)
    error_surface = st.sidebar.slider('Surface Error', 0.0, 5.0, 1.8030)
    error_edge = st.sidebar.slider('Edge Error', 0.0, 5.0, 3.4030)
    error_area = st.sidebar.slider('Area Error', 0.0, 200.0, 53.1300)
    error_smoothness = st.sidebar.slider('Smoothness Error', 0.0, 1.0, 0.00704)
    error_tightness = st.sidebar.slider('Tightness Error', 0.0, 1.0, 0.01498)
    error_indentation = st.sidebar.slider('Indentation Error', 0.0, 1.0, 0.05373)
    error_dip_points = st.sidebar.slider('Dip Points Error', 0.0, 1.0, 0.01587)
    error_balance = st.sidebar.slider('Balance Error', 0.0, 1.0, 0.03003)
    error_roughness = st.sidebar.slider('Roughness Error', 0.0, 1.0, 0.006193)

    # Worst case inputs (the most extreme values for each feature)
    worst_size = st.sidebar.slider('Worst Size', 10.0, 50.0, 25.38)
    worst_surface = st.sidebar.slider('Worst Surface', 10.0, 50.0, 17.33)
    worst_edge = st.sidebar.slider('Worst Edge', 50.0, 200.0, 184.60)
    worst_area = st.sidebar.slider('Worst Area', 100.0, 5000.0, 2019.0)
    worst_smoothness = st.sidebar.slider('Worst Smoothness', 0.0, 1.0, 0.1622)
    worst_tightness = st.sidebar.slider('Worst Tightness', 0.0, 1.0, 0.6656)
    worst_indentation = st.sidebar.slider('Worst Indentation', 0.0, 1.0, 0.7119)
    worst_dip_points = st.sidebar.slider('Worst Dip Points', 0.0, 1.0, 0.2654)
    worst_balance = st.sidebar.slider('Worst Balance', 0.0, 1.0, 0.4601)
    worst_roughness = st.sidebar.slider('Worst Roughness', 0.0, 1.0, 0.11890)

    # Collecting all user inputs in a tuple
    user_input = (
        size, surface, edge, area, smoothness, tightness, indentation, dip_points, balance, roughness,
        error_size, error_surface, error_edge, error_area, error_smoothness, error_tightness, 
        error_indentation, error_dip_points, error_balance, error_roughness,
        worst_size, worst_surface, worst_edge, worst_area, worst_smoothness, worst_tightness, 
        worst_indentation, worst_dip_points, worst_balance, worst_roughness
    )
    
    return user_input

# Title and description
st.title('Breast Cancer Prediction App')
st.markdown("""
This is a machine learning app to predict whether a tumour is benign (non-cancerous) or malignant (cancerous).
Simply enter the tumour details below, and the app will provide a prediction after you click "Predict".
""")

# Get user input
input_data = user_input_features()

# Convert the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prediction button and results
if st.button('Predict'):
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 'M':
        st.subheader("Prediction: Malignant Tumour")
        st.write("The tumour is predicted to be **cancerous**.")
    else:
        st.subheader("Prediction: Benign Tumour")
        st.write("The tumour is predicted to be **non-cancerous**.")