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

# Load the trained model, scaler, and training feature list.
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

with open(features_path, "rb") as file:
    training_features = pickle.load(file)  # Expected to be a list or Pandas Index

# ------------------ STREAMLIT UI ------------------
st.title("Diabetes Prediction App")
st.write("Enter your details below:")

# --- CATEGORICAL INPUTS ---
# For hypertension and heart disease, display "Yes" or "No".
hypertension_ui = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease_ui = st.selectbox("Heart Disease", ["Yes", "No"])

# Other categorical inputs.
gender = st.selectbox("Select Gender", ["Male", "Female"])
smoking_history_choice = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])

# --- NUMERICAL INPUTS ---
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Map UI values to those used in training.
# (Assuming that training data stored "1" for Yes and "0" for No.)
hypertension_val = "1" if hypertension_ui == "Yes" else "0"
heart_disease_val = "1" if heart_disease_ui == "Yes" else "0"

# ------------------ DATA PREPROCESSING ------------------
# Create a DataFrame from the user inputs.
input_dict = {
    "gender": [gender],
    "hypertension": [hypertension_val],
    "heart_disease": [heart_disease_val],
    "smoking_history": [smoking_history_choice],
    "age": [age],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
}
input_data = pd.DataFrame(input_dict)

# Set fixed categories for consistent one-hot encoding.
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])
input_data["gender"] = pd.Categorical(input_data["gender"], categories=["Female", "Male"])
input_data["smoking_history"] = pd.Categorical(
    input_data["smoking_history"], 
    categories=["No Info", "current", "former", "never"]
)

categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]

# One-hot encode categorical features (using drop_first=True as in training).
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure alignment with the training features by adding any missing columns with zeros.
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to exactly match the training feature order.
input_data_encoded = input_data_encoded[training_features]

# NOTE: No additional normalization of HbA1c is applied, leaving the raw value.
# Scale the input data using the saved scaler.
input_scaled = scaler.transform(input_data_encoded)

# ------------------ DEBUG OPTIONS ------------------
if st.checkbox("Show Debug Info"):
    st.write("Encoded Features:")
    st.write(input_data_encoded)
    st.write("Scaled Features:")
    st.write(input_scaled)

# ------------------ MODEL PREDICTION ------------------
if st.button("Predict"):
    # Use only model.predict since predict_proba isn't enabled.
    prediction = model.predict(input_scaled)
    
    # Display a single result.
    if prediction[0] == 1:
        st.error("The ML Model predicts: DIABETIC")
    else:
        st.success("The ML Model predicts: NON-DIABETIC")
