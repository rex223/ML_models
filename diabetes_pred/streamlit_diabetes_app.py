import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# --------------------- CONSTANTS ---------------------
# These constants should match the min & max used for HbA1c_level during training.
HBA1c_MIN = 3.0
HBA1c_MAX = 15.0

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
means_path = os.path.join(BASE_DIR, "diabetes_means.pkl")

# Load saved artifacts
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(features_path, 'rb') as file:
    training_features = pickle.load(file)  # Expected to be a list or Pandas Index

with open(means_path, 'rb') as file:
    diabetes_means = pickle.load(file)  # DataFrame indexed by 0 and 1

# ------------------ STREAMLIT UI ------------------
st.title("Diabetes Prediction App")
st.write("Enter your details below:")

# Dropdown inputs for categorical options.
gender = st.selectbox("Select Gender", ["Male", "Female"])
hypertension_choice = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease_choice = st.selectbox("Heart Disease", ["Yes", "No"])
smoking_history_choice = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])

# Other inputs.
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# ------------------ DATA PREPROCESSING ------------------
# Create DataFrame from user inputs.
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

# Convert categorical columns to string.
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
input_data[categorical_columns] = input_data[categorical_columns].astype(str)

# One-hot encode categorical features exactly as during training (drop_first=True).
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure alignment with training features by adding any missing columns.
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the training feature order.
input_data_encoded = input_data_encoded[training_features]

# Apply min–max normalization for HbA1c_level.
input_data_encoded['HbA1c_level'] = (
    input_data_encoded['HbA1c_level'] - HBA1c_MIN
) / (HBA1c_MAX - HBA1c_MIN)

# Scale the input data.
input_scaled = scaler.transform(input_data_encoded)

# ------------------ DEBUG OPTIONS ------------------
# Show debug information if needed.
if st.checkbox("Show Debug Info"):
    st.write("Encoded features:")
    st.write(input_data_encoded)
    st.write("Scaled features:")
    st.write(input_scaled)

# ------------------ MODEL PREDICTION ------------------
# Check if the model supports probability estimates.
use_probability = hasattr(model, "predict_proba")
if use_probability:
    threshold = st.slider("Set Threshold for Diabetes Prediction", 0.0, 1.0, 0.5)
else:
    threshold = 0.5  # Default threshold if probabilities are not used

if st.button("Predict"):
    if use_probability:
        prob = model.predict_proba(input_scaled)[:, 1]
        # Adjust prediction using the chosen threshold.
        prediction = (prob > threshold).astype(int)
    else:
        prediction = model.predict(input_scaled)
        prob = None

    # Compare input features to group means for a debug hint.
    diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[1].values)
    non_diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[0].values)
    # Lower total difference indicates closer to that group.
    closer_to_diabetes = np.sum(diabetes_mean_diff) < np.sum(non_diabetes_mean_diff)

    # Display the prediction results.
    if prediction[0] == 1:
        st.error("⚠️ The ML Model predicts you may have diabetes.")
    else:
        st.success("✅ The ML Model predicts you do not have diabetes.")

    if closer_to_diabetes:
        st.markdown(
            '<p style="color:red; font-weight:bold; font-size:26px;">'
            'Your values are closer to the diabetic group.</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:green; font-weight:bold; font-size:26px;">'
            'Your values are closer to the non-diabetic group.</p>',
            unsafe_allow_html=True
        )

    if prob is not None:
        st.write("Probability of Diabetes:", prob[0])
