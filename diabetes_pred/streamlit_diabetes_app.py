import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# --------------------- CONSTANTS ---------------------
# These constants should match the min and max used for HbA1c_level during training.
HBA1c_MIN = 3.0
HBA1c_MAX = 15.0

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
means_path = os.path.join(BASE_DIR, "diabetes_means.pkl")

# Load saved artifacts.
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(features_path, 'rb') as file:
    training_features = pickle.load(file)  # This should be a list or Index of feature names

with open(means_path, 'rb') as file:
    diabetes_means = pickle.load(file)  # DataFrame with index 0 and 1

# ------------------ STREAMLIT UI ------------------
st.title("Diabetes Prediction App")
st.write("Enter your details below:")

# Dropdowns for categorical options
gender = st.selectbox("Select Gender", ["Male", "Female"])
hypertension_choice = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease_choice = st.selectbox("Heart Disease", ["Yes", "No"])
smoking_history_choice = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])

# Other inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# ------------------ DATA PREPROCESSING ------------------
# Create a DataFrame from user input
input_dict = {
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension_choice],
    "heart_disease": [heart_disease_choice],
    "smoking_history": [smoking_history_choice],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
}

input_data = pd.DataFrame(input_dict)

# Convert all categorical columns to string
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
input_data[categorical_columns] = input_data[categorical_columns].astype(str)

# One-hot encode categorical features as in training (drop_first=True)
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure feature alignment with training by adding missing columns with zeros.
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to exactly match the saved training feature order:
input_data_encoded = input_data_encoded[training_features]

# Apply min–max normalization for HbA1c_level as in training.
input_data_encoded['HbA1c_level'] = (
    input_data_encoded['HbA1c_level'] - HBA1c_MIN
) / (HBA1c_MAX - HBA1c_MIN)

# Scale the processed input using the saved scaler.
input_scaled = scaler.transform(input_data_encoded)

# ------------------ MODEL PREDICTION ------------------
# If the model supports probability estimates, use predict_proba with a custom threshold.
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(input_scaled)[:, 1]
    threshold = 0.6  # Adjust threshold as needed
    prediction = (probability > threshold).astype(int)
else:
    prediction = model.predict(input_scaled)
    probability = None

# Mean Comparison Analysis for Debugging
diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[1].values)
non_diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[0].values)
# Lower total difference indicates closeness to that group's average
closer_to_diabetes = np.sum(diabetes_mean_diff) < np.sum(non_diabetes_mean_diff)

# ------------------ DISPLAY RESULTS ------------------
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("⚠️⚠️ The ML Model predicts you may have diabetes. ⚠️⚠️")
    else:
        st.success("✅ The ML Model predicts you do not have diabetes. ✅")
    
    if closer_to_diabetes:
        st.markdown(
            '<p style="color:red; font-weight:bold; font-size:26px;">'
            'Your values are closer to the <strong>diabetic group</strong>.</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:green; font-weight:bold; font-size:26px;">'
            'Your values are closer to the <strong>non-diabetic group</strong>.</p>',
            unsafe_allow_html=True
        )
    
    if probability is not None:
        st.write("Probability of Diabetes:", probability[0])
