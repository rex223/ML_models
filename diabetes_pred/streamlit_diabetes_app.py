import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
# For fuzzy logic:
import skfuzzy as fuzz
import skfuzzy.control as ctrl
# For SHAP:
import shap


# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved XGBoost model, scaler, training features, and fuzzy system
try:
    with open(os.path.join(BASE_DIR, "xgb_model.pkl"), "rb") as file:
        xgb_model = pickle.load(file)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as file:
        scaler = pickle.load(file)
    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as file:
        training_features = pickle.load(file)  
    with open(os.path.join(BASE_DIR, "hba1c_min_max.pkl"), "rb") as file:
        hba1c_min, hba1c_max = pickle.load(file)
    with open(os.path.join(BASE_DIR, "fuzzy.pkl"), "rb") as file:
        risk_ctrl = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error loading model, scaler, or fuzzy system: {e}")
    st.stop()

# Initialize the fuzzy control system simulation
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# ------------------ STREAMLIT UI ------------------
st.title(" Diabetes Prediction & Pre-Diabetes Risk Assessment")
st.write("Enter your details below to assess your diabetes risk:")

# --- CATEGORICAL INPUTS ---
hypertension_ui = st.selectbox("⚕️ Hypertension", ["Yes", "No"])
heart_disease_ui = st.selectbox("❤️ Heart Disease", ["Yes", "No"])
gender_ui = st.selectbox("🚻 Gender", ["Male", "Female"])
smoking_history_ui = st.selectbox("🚬 Smoking History", ["never", "current", "former", "No Info"])

# --- NUMERICAL INPUTS ---
age = st.number_input("📅 Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("🩸 HbA1c Level", min_value=3.0, max_value=15.0, value=6.5)
blood_glucose_level = st.number_input("🧪 Blood Glucose Level", min_value=50, max_value=300, value=126)

# Input validation
if HbA1c_level < 3.0 or HbA1c_level > 15.0:
    st.warning("⚠️ HbA1c Level should be between 3.0 and 15.0. Please adjust your input.")
if blood_glucose_level < 50 or blood_glucose_level > 300:
    st.warning("⚠️ Blood Glucose Level should be between 50 and 300. Please adjust your input.")
if bmi < 10.0 or bmi > 50.0:
    st.warning("⚠️ BMI should be between 10.0 and 50.0. Please adjust your input.")

# Map categorical inputs to the format used during training
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
    "HbA1c_level": [HbA1c_level],  # Try without normalization first
    "blood_glucose_level": [blood_glucose_level]
}

input_data = pd.DataFrame(input_dict)
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])


# Debug: Raw input data
st.write("**Debug Info (Raw Input Data):**")
st.write(input_data)

# Try with normalization (comment out if not used during training)
normalized_hba1c = (input_data['HbA1c_level'] - hba1c_min) / (hba1c_max - hba1c_min)
input_data['HbA1c_level'] = normalized_hba1c

# Debug: After HbA1c normalization (if applied)
# st.write("**Debug Info (After HbA1c Normalization):**")
# st.write(f"HbA1c Min: {hba1c_min}, HbA1c Max: {hba1c_max}")
# st.write(f"Normalized HbA1c: {normalized_hba1c[0]:.4f}")
# st.write(input_data)
# Debug: Print all consequents in the fuzzy system
# st.write("**Debug Info (Fuzzy System Consequents):**")
# for consequent in risk_ctrl.consequents:
#     st.write(f"Consequent: {consequent.label}")

# Set fixed categories for consistent one-hot encoding
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])
input_data["gender"] = pd.Categorical(input_data["gender"], categories=["Female", "Male", "Other"])
input_data["smoking_history"] = pd.Categorical(input_data["smoking_history"], categories=["No Info", "current", "former", "never"])


# One-hot encode categorical features
categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Drop any columns in input_data_encoded that are not in training_features
# input_data_encoded = input_data_encoded.loc[:, input_data_encoded.columns.isin(training_features)]

# Add missing columns with zeros and align with training_features
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[training_features]


# Scale only the continuous features
continuous_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = [col for col in input_data_encoded.columns if col not in continuous_cols]


# ------------------ MODEL PREDICTION ------------------
if st.button("🔍 Predict"):
    # Get XGBoost prediction probabilities for class 1
    xgb_prob = xgb_model.predict_proba(input_data_encoded)[:, 1]
    
    # Debug: Raw prediction output
    st.write("**Debug Info (Model Prediction):**")
    st.write(f"Raw XGBoost Probabilities (Class 0, Class 1): {xgb_model.predict_proba(input_data_encoded)[0]}")
    st.write(f"Probability of Class 1 (Diabetic): {xgb_model.predict_proba(input_data_encoded)}")


    # Use XGBoost probability for primary classification
    if xgb_prob[0] >= 0.150:  # Use the training threshold of 0.146
        st.error("⚠️ The model predicts that u are: **DIABETIC**")
        st.markdown(f"**Probability:** {xgb_prob[0]:.4f}")
        st.markdown("Based on your inputs, you may have diabetes. It is strongly recommended to consult a healthcare professional for a comprehensive evaluation and appropriate medical advice.")
    elif xgb_prob[0] <= 0.002:  # Not Diabetic threshold
        st.success("✅ The model predicts that you are: **NON-DIABETIC**")
        st.markdown(f"**Probability:** {xgb_prob[0]:.4f}")
        st.markdown("Based on your inputs, you are unlikely to have diabetes. However, continue maintaining a healthy lifestyle, monitor your health regularly, and consult a doctor if you experience any symptoms or have concerns.")
    else:
        risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    # Set the inputs
    try:
        # Convert Series to float if necessary
        risk_sim.input['hba1c_level'] = float(normalized_hba1c.iloc[0]) if isinstance(normalized_hba1c, pd.Series) else float(normalized_hba1c)
        risk_sim.input['bmi'] = float(bmi)
        risk_sim.input['blood_glucose_level'] = float(blood_glucose_level)
        risk_sim.input['hypertension'] = float(hypertension_val)
        
        # Convert categorical input to integer for fuzzy logic
        smoking_val = 1 if smoking_history_ui in ["current", "former"] else 0
        risk_sim.input['smoking'] = float(smoking_val)

    except Exception as e:
        st.error(f"Error setting fuzzy inputs: {e}")
        st.stop()

    # # Debug: Print membership degrees
    # st.write("**Debug Info (Membership Degrees):**")
    # antecedents_dict = {ant.label: ant for ant in antecedents_list}  # Convert list to dictionary

    # for key, antecedent in antecedents_dict.items():
    #     try:
    #         input_value = risk_sim.input[key]
    #         st.write(f"{key} ({input_value}):")
    #         for label in antecedent.terms:
    #             membership = fuzz.interp_membership(antecedent.universe, antecedent[label].mf, input_value)
    #             st.write(f"  {label}: {membership:.4f}")
    #     except Exception as e:
    #         st.error(f"Error computing membership for {key}: {e}")


    # Compute the fuzzy output
    try:
        risk_sim.compute()
        st.write("**Debug Info (Fuzzy System Outputs):**")
        st.write(risk_sim.output)
        
        fuzzy_score = risk_sim.output.get("diabetes_risk", 50.0)  # Default to 50.0 if computation fails
    except Exception as e:
        st.error(f"Error in fuzzy system computation: {e}")
        st.stop()

    # Interpret the Pre-Diabetic risk
    if xgb_prob[0] >= 0.15:
        st.warning("⚠️ The model indicates a **Pre-Diabetic Risk** (High risk).")
        st.markdown("You are at high risk of developing diabetes. Consider making lifestyle changes such as improving your diet, increasing physical activity, and consulting a doctor for further guidance.")
    elif xgb_prob[0] >= 0.05:
        st.info("ℹ️ The model indicates a **Pre-Diabetic Risk** (Moderate risk).")
        st.markdown("You are at moderate risk of developing diabetes. Monitor your health closely, adopt preventive measures, and consider consulting a doctor for advice.")
    else:
        st.success("✅ The model indicates a **Pre-Diabetic Risk** (Low risk).")
        st.markdown("You are at low risk of developing diabetes. Continue maintaining a healthy lifestyle and monitor your health regularly.")

                
        st.markdown(f"**XGB Probability:** {xgb_prob[0]:.4f} | **Fuzzy Risk Score:** {fuzzy_score:.4f}")

# ------------------ DEBUG OPTIONS ------------------

# How input data is encoded?
if st.checkbox("🔎 Show Additional Debug Info"):
    st.write("**Encoded Features:**")
    st.write(input_data_encoded)
    
# Use SHAP to explain the prediction?
if st.checkbox("Know the weightage of each feature in the prediction?"):
    shap_values = explainer.shap_values(input_data_encoded)
    st.write("**SHAP Values (Contribution of Each Feature tPrediction):**")
    shap_df = pd.DataFrame({
        "Feature": training_features,
        "SHAP Value": shap_values[0]
    })
    st.write(shap_df)
# Debug: Print all consequents in the fuzzy system
if st.checkbox("Consequents name?"):
    st.write("**Debug Info (Fuzzy System Consequents):**")
    for consequent in risk_ctrl.consequents:
        st.write(f"Consequent: {consequent.label}")       
# Debug: Print fuzzy rules
if st.checkbox("Fuzzy rules"):
    st.write("**Debug Info (Fuzzy System Rules):**")
    for rule in risk_ctrl.rules:
        st.write(f"Rule: {rule}")
# ------------- FOOTER ------------------
st.markdown("""
    ---
    **Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")