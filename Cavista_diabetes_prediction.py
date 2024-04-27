import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# pip install tensorflow
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diabetes_prediction_dataset.csv")
# df = data.copy()
# df.head(3)

import pickle
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create LabelEncoder and StandardScaler instances
lbl_en = defaultdict(LabelEncoder)
scaler = StandardScaler()

columns_to_encode = ['gender', 'smoking_history']
columns_to_scale = ['age','bmi', 'blood_glucose_level']  # Add your numeric columns here

# Apply LabelEncoder to categorical columns
df[columns_to_encode] = df[columns_to_encode].apply(lambda x: lbl_en[x.name].fit_transform(x))

# Apply StandardScaler to numeric columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Save LabelEncoder, StandardScaler, and other necessary information
filename = 'labSca.sav'
data_to_save = {
    'label_encoders': dict(lbl_en),
    'scaler': scaler,
    'columns_to_encode': columns_to_encode,
    'columns_to_scale': columns_to_scale
    # Add any other information you want to save
}

# pickle.dump(data_to_save, open(filename, 'wb'))


# x = ds.drop('diabetes',axis=1)
# y = df.diabetes
# #using XGBOOST to find feature importance
# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x,y)

# # first feature importance scores
# xgb.plot_importance(model)


# # feature selection
# selected_columns = ['age','gender', 'bmi', 'blood_glucose_level', 'HbA1c_level', 'smoking_history', 'heart_disease']
# new_ds = ds[selected_columns]
# new_ds.head()

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# x_train, x_test, y_train, y_test = train_test_split(new_ds, y, test_size = 0.10, random_state = 47, stratify = y)
# print(f'x_train: {x_train.shape}')
# print(f'x_test: {x_test.shape}')
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))

# DEEPE LEARNING MODEL

# model = tf.keras.Sequential([ #........................ Instantiate the model creating class.
#     tf.keras.layers.Dense(units=12, activation='relu'), #... Input layer of 12 features
#     tf.keras.layers.Dense(20, activation='relu'), #.... Add the second 20 layer, and instantiate the activation to be used.
#     tf.keras.layers.Dense(40, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(20, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(40, activation='relu'), #..... Add the third layer.
#     tf.keras.layers.Dense(1, activation='sigmoid') #... Add the last output layer
# ])
# model.compile(optimizer='adam', # ..................... The optimizer that adjusts weight and bias for a given neuron
#               loss = 'binary_crossentropy', #...... Loss calculates the error of the prediction
#               metrics=['accuracy']) #.................. Accuracy calculates the precision of the prediction.

# model.fit(x_train, y_train, epochs=25) #..... Fit the model on the dataset and define the number of epochs







import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('cavistadiabetespred.h5')

st.sidebar.image('pngwing.com (1).png', width = 250,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Diabetes Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>This is a Diabetes Prediction Model that was built by the Orpheus Sniper at the Cavista Hackathon Using Machine Learning to Enhance Early Detection and Improve Patient Outcomes.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('glowing-abstract-design-cancer-cells-generated-by-ai.jpg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Gender</h3>", unsafe_allow_html=True)
    st.markdown("<p>Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>Age is an important factor as diabetes is more commonly diagnosed in older adults.age ranges from 0-80 in our dataset.</p>", unsafe_allow_html=True)
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
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>CREATED BY THE ORPHEUS SNIPERS</p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(df)
    # st.sidebar.image('pngwing.com (13).png', width = 300,  caption = 'customer and deliver agent info')


if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()



if selected_page == "Modeling":
    st.sidebar.markdown("Add your modeling content here")
    gender = st.sidebar.selectbox("gender", df['gender'].unique())
    age = st.sidebar.number_input("age", 0,100)
    heart_disease = st.sidebar.selectbox("heart_disease", df['heart_disease'].unique())
    smoking_history = st.sidebar.selectbox('smoking_history', df['smoking_history'].unique())
    bmi = st.sidebar.number_input("bmi",  0.0, 100.0, format="%.2f")      
    HbA1c_level = st.sidebar.number_input("HbA1c_level", 0.0, 100.0, format="%.1f")
    blood_glucose_level = st.sidebar.number_input("blood_glucose_level",0,1000)
    st.sidebar.markdown('<br>', unsafe_allow_html= True)


    input_variables = pd.DataFrame([{
        'gender':gender,
        'age': age,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history, 
        'bmi': bmi, 
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level':blood_glucose_level 
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)
    

    # Standard Scale the Input Variable.
    import pickle
    filename = 'labSca.sav'
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
    

    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>DIABETES PREDICTION MODEL BUILT BY ORPHEUS SNIPERS </h8>",unsafe_allow_html=True)


    
