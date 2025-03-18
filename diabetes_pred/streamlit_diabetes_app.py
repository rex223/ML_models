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

# For hypertension and heart disease, we map UI-friendly options ("Yes"/"No")
# to the values used during training ("1"/"0").
hypertension_ui = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease_ui = st.selectbox("Heart Disease", ["Yes", "No"])

# For the other categorical fields, we continue using their display values.
gender = st.selectbox("Select Gender", ["Male", "Female"])
smoking_history_choice = st.selectbox("Smoking History", ["never", "current", "former", "No Info"])

# Other inputs.
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Map the UI values for hypertension and heart disease to match training data.
# (Assuming your training data used "1" for Yes and "0" for No.)
hypertension_val = "1" if hypertension_ui == "Yes" else "0"
heart_disease_val = "1" if heart_disease_ui == "Yes" else "0"

# ------------------ DATA PREPROCESSING ------------------
# Create DataFrame from user inputs.
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

# IMPORTANT: Convert categorical columns to fixed categories.
# For hypertension and heart_disease, we now use categories "0" and "1" to match training.
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])

# For other categorical fields, set fixed categories if appropriate.
input_data["gender"] = pd.Categorical(input_data["gender"], categories=["Female", "Male"])
input_data["smoking_history"] = pd.Categorical(
    input_data["smoking_history"], categories=["No Info", "current", "former", "never"]
)

categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]

# One-hot encode categorical features exactly as during training (drop_first=True).
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure alignment with training features by adding any missing columns.
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the training feature order.
input_data_encoded = input_data_encoded[training_features]

# Apply min–max normalization for HbA1c_level.
input_data_encoded["HbA1c_level"] = (
    input_data_encoded["HbA1c_level"] - HBA1c_MIN
) / (HBA1c_MAX - HBA1c_MIN)

# Scale the input data.
input_scaled = scaler.transform(input_data_encoded)

# ------------------ DEBUG OPTIONS ------------------
if st.checkbox("Show Debug Info"):
    st.write("Encoded Features:")
    st.write(input_data_encoded)
    st.write("Scaled Features:")
    st.write(input_scaled)

# ------------------ MODEL PREDICTION ------------------
if hasattr(model, "predict_proba"):
    threshold = st.slider("Set Threshold for Diabetes Prediction", 0.0, 1.0, 0.5)
else:
    threshold = 0.5

if st.button("Predict"):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[:, 1]
        prediction = (prob > threshold).astype(int)
    else:
        prediction = model.predict(input_scaled)
        prob = None

    # Debug: Compare input with group means.
    diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[1].values)
    non_diabetes_mean_diff = np.abs(input_data_encoded.values - diabetes_means.loc[0].values)
    closer_to_diabetes = np.sum(diabetes_mean_diff) < np.sum(non_diabetes_mean_diff)

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
