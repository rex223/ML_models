import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ------------------ DEBUG PRINTS ------------------
st.write("Current working directory:", os.getcwd())
st.write("Files in current working directory:", os.listdir(os.getcwd()))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("Script's BASE_DIR:", BASE_DIR)
st.write("Files in BASE_DIR:", os.listdir(BASE_DIR))

# ------------------ MODEL LOADING ------------------
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
means_path = os.path.join(BASE_DIR, "diabetes_means.pkl")

if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"Error: Scaler file not found at {scaler_path}")
    st.stop()

if not os.path.exists(features_path):
    st.error(f"Error: Features file not found at {features_path}")
    st.stop()

if not os.path.exists(means_path):
    st.error(f"Error: Diabetes means file not found at {means_path}")
    st.stop()

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(features_path, 'rb') as file:
    x = pickle.load(file)  # Load feature names

with open(means_path, 'rb') as file:
    diabetes_means = pickle.load(file)  # Load means for comparison

# ------------------ STREAMLIT UI ------------------
st.title("Diabetes Prediction App")
st.write("Enter your details below:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Use friendly labels for hypertension and heart disease
hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease_input = st.selectbox("Heart Disease", ["No", "Yes"])

smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Convert user-friendly inputs to numeric values:
hypertension = 1 if hypertension_input == "Yes" else 0
heart_disease = 1 if heart_disease_input == "Yes" else 0

# Preprocess user input
input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

# Convert gender to binary: Male -> 1, Female -> 0
input_data['gender'] = 1 if input_data['gender'].values[0] == "Male" else 0

# One-hot encode 'smoking_history'
input_data = pd.get_dummies(input_data, columns=['smoking_history'], drop_first=True)

# Ensure correct feature alignment
for col in x:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[x]  # Reorder columns

# Standardize input
input_scaled = scaler.transform(input_data)

# ML Model Prediction
prediction = model.predict(input_scaled)

# Mean Comparison Analysis:
diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[1].values)
non_diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[0].values)
closer_to_diabetes = np.sum(diabetes_mean_diff) < np.sum(non_diabetes_mean_diff)

# Display results
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("⚠️⚠️ The ML Model predicts you may have diabetes. ⚠️⚠️")
    else:
        st.success("✅ The ML Model predicts you do not have diabetes. ✅")

    # Show mean comparison result
    if closer_to_diabetes:
        st.markdown('<p style="color:red; font-weight:bold; font-size:26px;">🔍 Your values are closer to the **diabetic group** based on feature averages.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-weight:bold; font-size:26px;">📊 Looks like you take care of yourself! Your stats belong more to the non-diabetic zone.</p>', unsafe_allow_html=True)
