import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# For fuzzy logic:
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved XGBoost model, scaler, and training features
with open(os.path.join(BASE_DIR, "xgb_model.pkl"), "rb") as file:
    xgb_model = pickle.load(file)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as file:
    training_features = pickle.load(file)  # e.g., a list or Pandas Index of feature names

# ------------------ FUZZY LOGIC SYSTEM DEFINITION ------------------
# Define fuzzy variables for clinical parameters
hba1c = ctrl.Antecedent(np.arange(3, 15, 0.1), 'HbA1c_level')
bmi = ctrl.Antecedent(np.arange(10, 50, 0.1), 'bmi')
glucose = ctrl.Antecedent(np.arange(50, 300, 1), 'blood_glucose_level')
risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'diabetes_risk')

# Automatically create 3 membership functions for continuous variables
hba1c.automf(3)   # Defaults: poor, average, good
bmi.automf(3)
glucose.automf(3)
risk.automf(3)

# Optionally, rename them to 'low', 'medium', 'high' for interpretability
hba1c.terms['low'] = hba1c.terms.pop('poor')
hba1c.terms['medium'] = hba1c.terms.pop('average')
hba1c.terms['high'] = hba1c.terms.pop('good')

bmi.terms['low'] = bmi.terms.pop('poor')
bmi.terms['medium'] = bmi.terms.pop('average')
bmi.terms['high'] = bmi.terms.pop('good')

glucose.terms['low'] = glucose.terms.pop('poor')
glucose.terms['medium'] = glucose.terms.pop('average')
glucose.terms['high'] = glucose.terms.pop('good')

risk.terms['low'] = risk.terms.pop('poor')
risk.terms['medium'] = risk.terms.pop('average')
risk.terms['high'] = risk.terms.pop('good')

# Define fuzzy rules.
# These are examples; adjust them based on domain knowledge.
rule1 = ctrl.Rule(hba1c['high'] | glucose['high'], risk['high'])
rule2 = ctrl.Rule(hba1c['medium'] & glucose['medium'] & bmi['medium'], risk['medium'])
rule3 = ctrl.Rule(hba1c['low'] & glucose['low'] & bmi['low'], risk['low'])

# Create and initialize fuzzy control system
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# ------------------ STREAMLIT UI ------------------
st.title("🩸 Diabetes Prediction & Pre-Diabetes Risk Assessment")
st.write("Enter your details below:")

# --- CATEGORICAL INPUTS ---
hypertension_ui = st.selectbox("⚕️ Hypertension", ["Yes", "No"])
heart_disease_ui = st.selectbox("❤️ Heart Disease", ["Yes", "No"])
gender_ui = st.selectbox("🚻 Gender", ["Male", "Female"])
smoking_history_ui = st.selectbox("🚬 Smoking History", ["never", "current", "former", "No Info"])

# --- NUMERICAL INPUTS ---
age = st.number_input("📅 Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("🩸 HbA1c Level", min_value=3.0, max_value=15.0, value=8.0)
blood_glucose_level = st.number_input("🧪 Blood Glucose Level", min_value=50, max_value=300, value=140)

# Map categorical inputs to the format used during training.
hypertension_val = "1" if hypertension_ui == "Yes" else "0"
heart_disease_val = "1" if heart_disease_ui == "Yes" else "0"

# ------------------ DATA PREPROCESSING ------------------
input_dict = {
    "gender": [gender_ui],
    "hypertension": [hypertension_val],
    "heart_disease": [heart_disease_val],
    "smoking_history": [smoking_history_ui],
    "age": [age],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
}
input_data = pd.DataFrame(input_dict)

# Set fixed categories for consistent one-hot encoding
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])
input_data["gender"] = pd.Categorical(input_data["gender"], categories=["Female", "Male"])
input_data["smoking_history"] = pd.Categorical(input_data["smoking_history"], categories=["No Info", "current", "former", "never"])

# One-hot encode categorical features
categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Ensure alignment with training features
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[training_features]

# Scale the input data using the saved scaler
input_scaled = scaler.transform(input_data_encoded)

# ------------------ MODEL PREDICTION ------------------
if st.button("🔍 Predict"):
    # Get XGBoost prediction probabilities for class 1
    xgb_prob = xgb_model.predict_proba(input_scaled)[:, 1]
    
    # Use XGBoost's probability for final decision (primary classifier)
    # For simplicity, we'll define:
    #  - if probability >= 0.6, classify as DIABETIC
    #  - if probability <= 0.4, classify as NON-DIABETIC
    #  - if between 0.4 and 0.6, use fuzzy logic to indicate Pre-Diabetic risk
    if xgb_prob[0] >= 0.6:
        st.error("⚠️ The model predicts: **DIABETIC**")
        st.markdown(f"**Probability:** {xgb_prob[0]:.2f}")
    elif xgb_prob[0] <= 0.4:
        st.success("✅ The model predicts: **NON-DIABETIC**")
        st.markdown(f"**Probability:** {xgb_prob[0]:.2f}")
    else:
        # Use fuzzy logic to further assess risk
        risk_sim.input['HbA1c_level'] = HbA1c_level
        risk_sim.input['bmi'] = bmi
        risk_sim.input['blood_glucose_level'] = blood_glucose_level
        risk_sim.compute()
        fuzzy_score = risk_sim.output['diabetes_risk']
        
        # Interpret fuzzy risk score:
        # If fuzzy score is high, suggest pre-diabetic risk.
        if fuzzy_score >= 0.6:
            st.warning("⚠️ The model indicates a **Pre-Diabetic Risk** (High risk).")
        elif fuzzy_score >= 0.4:
            st.info("ℹ️ The model indicates a **Pre-Diabetic Risk** (Moderate risk).")
        else:
            st.success("✅ The model indicates a **Pre-Diabetic Risk** (Low risk).")
            
        st.markdown(f"**XGB Probability:** {xgb_prob[0]:.2f} | **Fuzzy Risk Score:** {fuzzy_score:.2f}")

# ------------------ DEBUG OPTIONS ------------------
if st.checkbox("🔎 Show Debug Info"):
    st.write("Encoded Features:")
    st.write(input_data_encoded)
    st.write("Scaled Features:")
    st.write(input_scaled)
