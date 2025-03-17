# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pickle
import os

# ------------------ DEBUG PRINTS ------------------
print("Current working directory:", os.getcwd())
print("Files in current working directory:", os.listdir(os.getcwd()))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Script's BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR))

# ------------------ MODEL LOADING ------------------
# Load model.pkl
model_path = os.path.join(BASE_DIR, "model.pkl")
print("Looking for model at:", model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# ------------------ SCALER LOADING ------------------
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
print("Looking for scaler at:", scaler_path)

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Error: Scaler file not found at {scaler_path}")

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

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

    # Scale the input using the loaded scaler
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)[0]

    # Display prediction
    if prediction == 1:
        st.markdown('<p style="color:red; font-size:20px;">You have a HIGH risk of diabetes.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size:20px;">You have a LOW risk of diabetes.</p>', unsafe_allow_html=True)
