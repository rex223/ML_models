# -*- coding: utf-8 -*-
"""streamlit_diabetes_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Q15gf-_ImzYaOAJ6pQgJd11q7aM0BtCm
"""

!pip install streamlit

import streamlit as st
import numpy as np
import pickle

# Load the trained model
model_path = "/content/model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Diabetes Prediction System")
st.write("Enter your details below to check your diabetes risk.")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=25)

# Predict Button
if st.button("Predict Diabetes Risk"):
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(user_input)[0]  # Predict outcome

    # Display prediction
    if prediction == 1:
        st.markdown('<p style="color:red; font-size:20px;">You have a HIGH risk of diabetes.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size:20px;">You have a LOW risk of diabetes.</p>', unsafe_allow_html=True)

