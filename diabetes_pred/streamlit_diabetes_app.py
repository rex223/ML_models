import streamlit as st
import numpy as np
import pandas as pd
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
# ----------------------------------------------------

#                               STREAMLIT UI
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

# Convert user friendly inputs to numeric values:
hypertension = 1 if hypertension_input == "Yes" else 0
heart_disease = 1 if heart_disease_input == "Yes" else 0

# Preprocess user input: create a DataFrame
input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

# Convert gender to binary: Male -> 1, Female -> 0
input_data['gender'] = 1 if input_data['gender'].values[0] == "Male" else 0

# One-hot encode 'smoking_history'; drop the first category to avoid multicollinearity.
input_data = pd.get_dummies(input_data, columns=['smoking_history'], drop_first=True)

# Ensure that all expected columns from the training data (x) are present
for col in x.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[x.columns]

# Standardize the input using the pre-fitted scaler
input_scaled = scaler.transform(input_data)

# ML Model Prediction
prediction = classifier.predict(input_scaled)

# Mean Comparison Analysis:
# Compare the absolute difference between the input and the group means for diabetic (1) and non-diabetic (0) groups.
diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[1].values)
non_diabetes_mean_diff = np.abs(input_data.values - diabetes_means.loc[0].values)
closer_to_diabetes = np.sum(diabetes_mean_diff) < np.sum(non_diabetes_mean_diff)

# Display results when the "Predict" button is pressed.
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("⚠️⚠️The ML Model predicts you may have diabetes.⚠️⚠️")
    else:
        st.success("✅ The ML Model predicts you do not have diabetes.✅")

    # Show mean comparison result
    if closer_to_diabetes:
        st.markdown('<p style="color:red; font-weight:bold; font-size:26px;">🔍 Your values are closer to the **diabetic group** based on feature averages.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-weight:bold; font-size:26px;">📊 Looks like u take care of urself.Your stats belong more to the non diabetic zone .</p>', unsafe_allow_html=True)
