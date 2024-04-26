import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# pip install tensorflow
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('diabetes_prediction_dataset.csv')
df = data.copy()
df.head(3)


import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('Model.h5')

st.sidebar.image('pngwing.com (12).png', width = 300,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Diabetes Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>This is a Diabetes Prediction Model that was built Using Machine Learning to Enhance Early Detection and Improve Patient Outcomes.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('newk.svg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)


    # Background story
    st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: center; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)
    st.markdown("This project is personal to me because my grandpa had diabetes for a long time. I want to create a computer program that can tell if someone might get diabetes in the future. By using information about a person's health, the program will try to help them know if they need to be careful. I'm doing this so that others don't have to go through what my grandpa did. Let's work together to use technology to help people stay healthy and avoid diabetes", unsafe_allow_html = True)

    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Gender</h3>", unsafe_allow_html=True)
    st.markdown("<p>Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>Age is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Hypertension</h3>", unsafe_allow_html=True)
    st.markdown("<p>Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Heart Diseases</h3>", unsafe_allow_html=True)
    st.markdown("<p>Heart disease is another medical condition that is associated with an increased risk of developing diabetes.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Smoking history</h3>", unsafe_allow_html=True)
    st.markdown("<p>Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Body Mass Index</h3>", unsafe_allow_html=True)
    st.markdown("<p>BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Hemoglobin A1c</h3>", unsafe_allow_html=True)
    st.markdown("<p>HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Blood glucose level</h3>", unsafe_allow_html=True)
    st.markdown("<p>Blood glucose level refers to the amount of glucose in the bloodstream at a given time. </p>", unsafe_allow_html=True)


    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by Datapsalm</p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(df.head())
    # st.sidebar.image('pngwing.com (13).png', width = 300,  caption = 'customer and deliver agent info')


if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()



if selected_page == "Modeling":
    st.sidebar.markdown("Add your modeling content here")
    Gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
    Age = st.sidebar.number_input("Age", 0,100)
    Hypertension = st.sidebar.selectbox("Hypertension", df['Hypertension'].unique())
    Heart_disease = st.sidebar.selectbox("Heart_disease", df['Heart_disease'].unique())
    Smoking_history = st.sidebar.selectbox('Smoking_history', df['Smoking_history'].unique())
    Body_Mass_Index = st.sidebar.number_input("Body_Mass_Index",  0.0, 100.0, format="%.2f")      
    Hemoglobin_A1c = st.sidebar.number_input("Hemoglobin_A1c", 0.0, 100.0, format="%.1f")
    Blood_glucose_level = st.sidebar.number_input("Blood_glucose_level",0,1000)
    st.sidebar.markdown('<br>', unsafe_allow_html= True)


    input_variables = pd.DataFrame([{
        'Gender':Gender,
        'Age': Age,
        'Hypertension': Hypertension,
        'Heart_disease': Heart_disease,
        'Smoking_history': Smoking_history, 
        'Body_Mass_Index': Body_Mass_Index, 
        'Hemoglobin_A1c': Hemoglobin_A1c,
        'Blood_glucose_level':Blood_glucose_level 
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)
    

    # Standard Scale the Input Variable.
    import pickle
    filename = 'Finalize_Train.sav'
    with open(filename, 'rb') as file:
        saved_data = pickle.load(file)
    label_encoders = saved_data['label_encoders']
    scaler = saved_data['scaler']

    for col in input_variables.columns:
        if col in label_encoders:
            input_variables[col] = label_encoders[col].transform(input_variables[col])

    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Press To Predict'):
        st.markdown("<h4 style = 'color: #2B2A4C; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)
        predicted = model.predict(input_variables)
        st.toast('Predicted Successfully')
        st.image('check icon.png', width = 100)
        st.success(f'Model Predicted {int(np.round(predicted))}')
        if predicted >= 0.5:
            st.error('High risk of diabetes!')
        else:
            st.success('Low risk of diabetes.')

    st.markdown('<hr>', unsafe_allow_html=True)
    

    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>DIABETES PREDICTION MODEL BUILT BY DATAPSALM</h8>",unsafe_allow_html=True)


    
