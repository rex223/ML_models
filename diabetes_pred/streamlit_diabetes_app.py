import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved models and scalers
model_path = os.path.join(BASE_DIR, "model.pkl")
xgb_path = os.path.join(BASE_DIR, "xgb_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
fuzzy_path = os.path.join(BASE_DIR, "fuzzy_model.pkl")

with open(model_path, "rb") as file:
    svm_model = pickle.load(file)

with open(xgb_path, "rb") as file:
    xgb_model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

with open(features_path, "rb") as file:
    training_features = pickle.load(file)

with open(fuzzy_path, "rb") as file:
    fuzzy_model = pickle.load(file)

# ------------------ STREAMLIT UI ------------------
st.title("🩸 Diabetes Prediction & Risk Assessment App")
st.write("Enter your details below to check your diabetes risk.")

# --- CATEGORICAL INPUTS ---
hypertension_ui = st.selectbox("⚕️ Hypertension", ["Yes", "No"])
heart_disease_ui = st.selectbox("❤️ Heart Disease", ["Yes", "No"])
gender = st.selectbox("🚻 Select Gender", ["Male", "Female"])
smoking_history_choice = st.selectbox("🚬 Smoking History", ["never", "current", "former", "No Info"])

# --- NUMERICAL INPUTS ---
age = st.number_input("📅 Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("🩸 HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("🧪 Blood Glucose Level", min_value=50, max_value=300, value=100)

# Map UI values to numerical values
hypertension_val = "1" if hypertension_ui == "Yes" else "0"
heart_disease_val = "1" if heart_disease_ui == "Yes" else "0"

# ------------------ DATA PREPROCESSING ------------------
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

# One-hot encode categorical features.
categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure alignment with the training features.
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the training feature order.
input_data_encoded = input_data_encoded[training_features]

# Scale the input data using the saved scaler.
input_scaled = scaler.transform(input_data_encoded)

# ------------------ PREDICTIONS ------------------
if st.button("🔍 Predict"):

    # Get predictions from ML models
    svm_pred = svm_model.predict(input_scaled)
    svm_prob = svm_pred.astype(float)  # Convert SVM prediction to float (0.0 or 1.0)
    
    xgb_prob = xgb_model.predict_proba(input_scaled)[:, 1]  # Get probability from XGBoost
    ensemble_prob = (svm_prob + xgb_prob) / 2  # Combine SVM and XGBoost probabilities

    # Get fuzzy logic risk score
    fuzzy_model.input['HbA1c_level'] = input_data['HbA1c_level'].values[0]
    fuzzy_model.input['bmi'] = input_data['bmi'].values[0]
    fuzzy_model.input['blood_glucose_level'] = input_data['blood_glucose_level'].values[0]
    fuzzy_model.compute()  # Compute fuzzy logic output
    fuzzy_risk_score = fuzzy_model.output['diabetes_risk']

    # Combine ML Predictions with Fuzzy Logic (weighted combination)
    final_score = (0.7 * ensemble_prob) + (0.3 * fuzzy_risk_score)
    
    # ------------------ DISPLAY RESULTS ------------------
    if final_score >= 0.6:
        st.error("⚠️ **High Diabetes Risk!** Please consult a doctor. 🏥")
        st.markdown(f"**Final Score:** {final_score:.2f}")
    
    elif 0.4 <= final_score < 0.6:
        st.warning("⚠️ **Pre-Diabetic Risk!** Maintain a healthy lifestyle. 🍎🏋️‍♂️")
        st.markdown(f"**Final Score:** {final_score:.2f}")
    
    else:
        st.success("✅ **Low Diabetes Risk!** Keep up the healthy habits! 🌱")
        st.markdown(f"**Final Score:** {final_score:.2f}")

# ------------------ DEBUG OPTIONS ------------------
if st.checkbox("🔎 Show Debug Info"):
    st.write("Encoded Features:")
    st.write(input_data_encoded)
    st.write("Scaled Features:")
    st.write(input_scaled)
