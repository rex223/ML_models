import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
means_path = os.path.join(BASE_DIR, "diabetes_means.pkl")

# Load model, scaler, and feature names
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(features_path, 'rb') as file:
    x = pickle.load(file)

with open(means_path, 'rb') as file:
    diabetes_means = pickle.load(file)

# ------------------ STREAMLIT UI ------------------
st.title("Diabetes Prediction App")
st.write("Enter your details below:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease_input = st.selectbox("Heart Disease", ["No", "Yes"])
smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Convert categorical inputs
hypertension = 1 if hypertension_input == "Yes" else 0
heart_disease = 1 if heart_disease_input == "Yes" else 0
gender_numeric = 1 if gender == "Male" else 0

# Preprocess input
input_data = pd.DataFrame([[gender_numeric, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

# One-hot encode 'smoking_history'
input_data = pd.get_dummies(input_data, columns=['smoking_history'], drop_first=True)

# Ensure correct feature alignment
for col in x:
    if col not in input_data:
        input_data[col] = 0  # Add missing columns

input_data = input_data[x]  # Reorder to match training features

# Scale input
input_scaled = scaler.transform(input_data)

# ✅ Adjust Decision Threshold
probability = model.predict_proba(input_scaled)[:, 1]
threshold = 0.6  # Adjust threshold (reduce false positives)
prediction = (probability > threshold).astype(int)

# Mean Comparison Analysis Debugging
diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[1].values)
non_diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[0].values)

# Print Debugging Information
st.write("Diabetes Mean Diff:", np.sum(diabetes_mean_diff))
st.write("Non-Diabetes Mean Diff:", np.sum(non_diabetes_mean_diff))

# Fix the Condition (Flip If Necessary)
closer_to_diabetes = np.sum(diabetes_mean_diff) > np.sum(non_diabetes_mean_diff)

# Display Results
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("⚠️⚠️ The ML Model predicts you may have diabetes. ⚠️⚠️")
    else:
        st.success("✅ The ML Model predicts you do not have diabetes. ✅")

    # Fix the Incorrect Else Statement
    if closer_to_diabetes:
        st.markdown('<p style="color:red; font-weight:bold; font-size:26px;">🔍 Your values are closer to the **diabetic group**.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-weight:bold; font-size:26px;">📊 Your values are closer to the **non-diabetic group**.</p>', unsafe_allow_html=True)')
