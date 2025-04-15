import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import shap

# Set page configuration
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 36px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 20px;
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #3498db;
    }
    
    /* Dark mode for main header */
    [data-theme="dark"] .main-header {
        color: blue;
        border-bottom: 2px solid #3498db;
    }
    
    .subheader {
        font-size: 24px;
        font-weight: 500;
        color: #34495e;
        margin: 20px 0 10px 0;
    }
    
    /* Dark mode for subheader */
    [data-theme="dark"] .subheader {
        color: #dcdde1;
    }
    
    /* Base styling for result boxes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    /* Light mode result boxes */
    .diabetic-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #212121;
    }
    
    .prediabetic-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        color: #212121;
    }
    
    .healthy-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #212121;
    }
    
    /* Dark mode result boxes */
    [data-theme="dark"] .diabetic-box {
        background-color: rgba(244, 67, 54, 0.2);
        border-left: 5px solid #f44336;
        color: #f5f5f5;
    }
    
    [data-theme="dark"] .prediabetic-box {
        background-color: rgba(255, 193, 7, 0.2);
        border-left: 5px solid #ffc107;
        color: #f5f5f5;
    }
    
    [data-theme="dark"] .healthy-box {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4caf50;
        color: #f5f5f5;
    }
    
    .metric-label {
        font-weight: 600;
        color: #34495e;
    }
    
    /* Dark mode metric label */
    [data-theme="dark"] .metric-label {
        color: #dcdde1;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #2196f3;
        color: #212121;
    }
    
    /* Dark mode info box */
    [data-theme="dark"] .info-box {
        background-color: rgba(33, 150, 243, 0.2);
        border-left: 5px solid #2196f3;
        color: #f5f5f5;
    }
    
    /* Link styling for both modes */
    .stApp a {
        color: #3498db;
    }
    
    .stApp a:hover {
        color: #2c3e50;
    }
    
    [data-theme="dark"] .stApp a {
        color: #3498db;
    }
    
    [data-theme="dark"] .stApp a:hover {
        color: #8bc4ea;
    }
    
    .input-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #212121;
    }
    
    /* Dark mode input section */
    [data-theme="dark"] .input-section {
        background-color: rgba(255, 255, 255, 0.05);
        color: #f5f5f5;
    }
    
    .footnote {
        font-size: 12px;
        color: #7f8c8d;
        text-align: center;
        margin-top: 50px;
    }
    
    /* Dark mode footnote */
    [data-theme="dark"] .footnote {
        color: #bdc3c7;
    }
    
    .highlight {
        font-weight: bold;
        color: #3498db;
    }
    
    /* Dark mode highlight */
    [data-theme="dark"] .highlight {
        color: #3498db;
    }
    
    /* Styled radio buttons and selectors */
    div[data-testid="stSelectbox"] label {
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* Dark mode for select boxes */
    [data-theme="dark"] div[data-testid="stSelectbox"] label {
        color: #ecf0f1;
    }
    
    /* Custom header for sections */
    .section-header {
        background-color: #3498db;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: 500;
        margin: 20px 0 10px 0;
    }
    
    /* Dark mode section header */
    [data-theme="dark"] .section-header {
        background-color: #2980b9;
    }
    
    /* Debug sections */
    .debug-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        border: 1px dashed #95a5a6;
        margin: 10px 0;
        color: #212121;
    }
    
    /* Dark mode debug box */
    [data-theme="dark"] .debug-box {
        background-color: rgba(240, 240, 240, 0.1);
        border: 1px dashed #bdc3c7;
        color: #f5f5f5;
    }
    
    /* Prediction button */
    .stButton button {
        background-color: #2c3e50;
        color: white;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 50px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #3498db;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Loading spinner */
    div[data-testid="stSpinner"] {
        margin: 20px 0;
    }
    
    /* Progress bars */
    div[data-testid="stProgressBar"] {
        margin: 10px 0;
    }
    
    /* ===== ENHANCED DARK MODE FIXES ===== */
    
    /* Main title and description fixes for dark mode */
    [data-theme="dark"] .stApp h1,
    [data-theme="dark"] .stApp h2,
    [data-theme="dark"] .stApp h3 {
        color: #ecf0f1 !important;
        font-weight: 600;
    }
    
    [data-theme="dark"] .stApp p,
    [data-theme="dark"] .stApp li,
    [data-theme="dark"] .stApp label,
    [data-theme="dark"] .stApp div {
        color: #ecf0f1;
    }
    
    /* White boxes in dark mode */
    [data-theme="dark"] .stTextInput input,
    [data-theme="dark"] .stNumberInput input,
    [data-theme="dark"] .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ecf0f1;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Fix for sidebar text in dark mode */
    [data-theme="dark"] .stSidebar .sidebar-content p,
    [data-theme="dark"] .stSidebar .sidebar-content h1,
    [data-theme="dark"] .stSidebar .sidebar-content h2,
    [data-theme="dark"] .stSidebar .sidebar-content h3,
    [data-theme="dark"] .stSidebar .sidebar-content div {
        color: #ecf0f1 !important;
    }
    
    /* White container backgrounds in dark mode */
    [data-theme="dark"] .stTabs div[data-baseweb="tab-panel"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    /* Ensure expanders and other containers are visible */
    [data-theme="dark"] .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Ensure form labels are visible */
    [data-theme="dark"] div[data-testid="stForm"] label,
    [data-theme="dark"] div[data-testid="stVerticalBlock"] label {
        color: #ecf0f1 !important;
    }
    
    /* Better visibility for radio buttons and checkboxes */
    [data-theme="dark"] div[data-testid="stRadio"] label,
    [data-theme="dark"] div[data-testid="stCheckbox"] label {
        color: #ecf0f1 !important;
    }
    
    /* Header section specifically for the Diabetes app */
    [data-theme="dark"] h1:contains("Diabetes Risk Assessment") {
        color: #3498db !important; 
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Blue info boxes with better visibility */
    [data-theme="dark"] div[data-testid="stAlert"] {
        background-color: rgba(33, 150, 243, 0.15);
        color: #ecf0f1;
        border-left-color: #3498db;
    }
    
    /* Special styling for white box containers in your screenshot */
    [data-theme="dark"] .element-container .stMarkdown div.stMarkdown {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Make all default text in the app visible in dark mode */
    [data-theme="dark"] .stApp {
        color: #ecf0f1;
    }
    
    /* Better contrast for tabs in dark mode */
    [data-theme="dark"] button[role="tab"] {
        background-color: rgba(52, 152, 219, 0.1);
        color: #ecf0f1 !important;
    }
    
    [data-theme="dark"] button[role="tab"][aria-selected="true"] {
        background-color: rgba(52, 152, 219, 0.3);
        border-bottom: 2px solid #3498db;
    }

    /* Styling for the HbA1c Level and other clinical parameters */
    [data-theme="dark"] div:has(> div > .stMarkdown:contains("HbA1c Level")) .stMarkdown {
        color: #ecf0f1 !important;
    }
    
    /* Styling for icons and their text */
    [data-theme="dark"] .css-1kyxreq span.e16nr0p33,
    [data-theme="dark"] .css-1kyxreq span.e16nr0p30 {
        color: #ecf0f1 !important;
    }
    
    /* ===== FIXES FOR BLUE HIGHLIGHTED TEXT ===== */
    
    /* Fix for blue highlighted headers and values with better contrast */
    /* These are the blue elements visible in your screenshot */
    
    /* Headers like HbA1c Level, Blood Glucose, BMI, etc. */
    .stApp [style*="background-color: rgb("] {
        background-color: #1a5fb4 !important; /* Darker blue for better contrast */
        color: white !important;
        font-weight: 600 !important;
        padding: 6px 10px !important;
        border-radius: 4px !important;
    }
    
    /* Make sure text in blue sections is visible in both light and dark modes */
    [style*="background-color: rgb(65, 105, 225)"],
    [style*="background-color: rgb(30, 144, 255)"],
    [style*="background-color: rgb(100, 149, 237)"] {
        background-color: #1a5fb4 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Fix specifically for the status indicators (like High/Diabetic, etc.) */
    [style*="background-color:"] {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Additional contrast fix for any blue backgrounds with text */
    .stApp [style*="color: white"] {
        text-shadow: 0 1px 1px rgba(0,0,0,0.5) !important; /* Add shadow for legibility */
    }
    
    /* Age Factor, Cardiovascular Risk, etc. headers */
    .stApp div:has(> div > .stMarkdown:contains("Age Factor")),
    .stApp div:has(> div > .stMarkdown:contains("Cardiovascular Risk")) {
        color: white !important;
    }
    
    /* Specific fix for "No" values in Hypertension and Heart Disease */
    .stApp div:has(> div > .stMarkdown:contains("No")) div {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Risk level labels */
    .stApp div:has(> div > .stMarkdown:contains("Risk Level")) {
        color: white !important;
    }
    
    /* Fix for "Combined Risk: Low" and similar status indicators */
    div:has(> div > .stMarkdown:contains("Combined Risk")) {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ MODEL LOADING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# App layout - Header
st.markdown('<div class="main-header">🩺 Diabetes Risk Assessment & Prediction</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
This clinical tool helps assess your risk of diabetes based on clinical and lifestyle factors.
For accurate results, please provide precise information. This tool should be used in consultation
with healthcare professionals.
</div>
""", unsafe_allow_html=True)

# Create a sidebar for information
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/diabetes-awareness-month-abstract-concept-illustration_335657-3831.jpg", use_container_width=True)
    st.markdown('<div class="subheader">About This Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    This diabetes risk assessment tool uses machine learning to predict:
    - Diabetes status (Diabetic/Non-diabetic)
    - Pre-diabetes risk level
    
    The prediction is based on:
    - Clinical measurements (HbA1c, Blood glucose)
    - Health conditions (Hypertension, Heart disease)
    - Personal factors (Age, BMI, Gender)
    - Lifestyle factors (Smoking history)
    
    **References:**
    - American Diabetes Association Guidelines
    - World Health Organization Diabetes Criteria
    """)
    
    st.markdown('<div class="subheader">Risk Factors</div>', unsafe_allow_html=True)
    st.markdown("""
    **High-risk categories:**
    - HbA1c ≥ 6.5%
    - Fasting blood glucose ≥ 126 mg/dL
    - BMI ≥ 30
    - Age > 45 years
    - Family history of diabetes
    - Hypertension or heart disease
    - Sedentary lifestyle
    """)

# Load the saved XGBoost model, scaler, training features, and fuzzy system
try:
    with st.spinner("Loading clinical prediction models..."):
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

    # Initialize the fuzzy control system simulation
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    
except FileNotFoundError as e:
    st.error(f"Error loading clinical models: {e}")
    st.stop()

# ------------------ STREAMLIT UI ------------------
# Create three columns for better layout
col1, col2 = st.columns(2)

# Input sections
with col1:
    st.markdown('<div class="section-header">Clinical Parameters</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        HbA1c_level = st.slider("🩸 HbA1c Level (%)", 
                             min_value=3.0, max_value=15.0, value=6.5, step=0.1,
                             help="HbA1c represents average blood glucose over past 3 months. Normal: <5.7%, Pre-diabetic: 5.7-6.4%, Diabetic: ≥6.5%")
        
        blood_glucose_level = st.slider("🧪 Blood Glucose Level (mg/dL)", 
                                    min_value=50, max_value=300, value=126, step=1,
                                    help="Fasting blood glucose level. Normal: <100 mg/dL, Pre-diabetic: 100-125 mg/dL, Diabetic: ≥126 mg/dL")
        
        bmi = st.slider("⚖️ BMI (kg/m²)", 
                     min_value=10.0, max_value=50.0, value=25.0, step=0.1,
                     help="Body Mass Index. Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30")
        
        age = st.slider("📅 Age (years)", 
                     min_value=18, max_value=100, value=40, step=1,
                     help="Risk increases with age, particularly after 45 years")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Health & Lifestyle Factors</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        hypertension_ui = st.radio("⚕️ Hypertension (High Blood Pressure)", 
                                ["No", "Yes"], index=0, horizontal=True,
                                help="High blood pressure increases diabetes risk")
        
        heart_disease_ui = st.radio("❤️ Heart Disease", 
                                 ["No", "Yes"], index=0, horizontal=True,
                                 help="Pre-existing heart conditions may increase diabetes risk")
        
        gender_ui = st.radio("🚻 Gender", 
                          ["Male", "Female"], index=0, horizontal=True,
                          help="Some diabetes risk factors vary by gender")
        
        smoking_history_ui = st.selectbox("🚬 Smoking History", 
                                       ["never", "former", "current", "No Info"], index=0,
                                       help="Smoking can affect insulin sensitivity and diabetes risk")
        st.markdown('</div>', unsafe_allow_html=True)

# Input validation with styled warnings
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
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
}

input_data = pd.DataFrame(input_dict)
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])

# Add normalization
normalized_hba1c = (input_data['HbA1c_level'] - hba1c_min) / (hba1c_max - hba1c_min)
input_data['HbA1c_level'] = normalized_hba1c

# Set fixed categories for consistent one-hot encoding
input_data["hypertension"] = pd.Categorical(input_data["hypertension"], categories=["0", "1"])
input_data["heart_disease"] = pd.Categorical(input_data["heart_disease"], categories=["0", "1"])
input_data["gender"] = pd.Categorical(input_data["gender"], categories=["Female", "Male", "Other"])
input_data["smoking_history"] = pd.Categorical(input_data["smoking_history"], categories=["No Info", "current", "former", "never"])

# One-hot encode categorical features
categorical_columns = ["gender", "hypertension", "heart_disease", "smoking_history"]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Add missing columns with zeros and align with training_features
for col in training_features:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[training_features]

# Scale only the continuous features
continuous_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = [col for col in input_data_encoded.columns if col not in continuous_cols]

# ------------------ MODEL PREDICTION ------------------
st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

# Create a styled prediction button
if st.button("🔍 Run Clinical Assessment", help="Click to analyze your diabetes risk based on the provided information"):
    with st.spinner("Analyzing clinical parameters..."):
        # Add a slight delay for UX
        import time
        time.sleep(0.5)
        
        # Get XGBoost prediction probabilities for class 1
        xgb_prob = xgb_model.predict_proba(input_data_encoded)[:, 1]
        
        # Show clinical metrics
        st.markdown('<div class="subheader">Clinical Assessment Results</div>', unsafe_allow_html=True)
        
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.metric(label="XGBoost Probability", value=f"{xgb_prob[0]:.2%}")
        
        # Risk level thresholds
        if xgb_prob[0] >= 0.146:  # Diabetic
            risk_level = "High Risk"
            risk_color = "#f44336"
        elif xgb_prob[0] >= 0.05:  # Pre-diabetic
            risk_level = "Moderate Risk"
            risk_color = "#ff9800"
        else:  # Low risk
            risk_level = "Low Risk"
            risk_color = "#4caf50"
            
        with col_metrics2:
            st.metric(label="Risk Level", value=risk_level)
            
        with col_metrics3:
            # Try to compute fuzzy score
            try:
                risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
                # Set the inputs
                risk_sim.input['hba1c_level'] = float(normalized_hba1c.iloc[0]) if isinstance(normalized_hba1c, pd.Series) else float(normalized_hba1c)
                risk_sim.input['bmi'] = float(bmi)
                risk_sim.input['blood_glucose_level'] = float(blood_glucose_level)
                risk_sim.input['hypertension'] = float(hypertension_val)
                
                # Convert categorical input to integer for fuzzy logic
                smoking_val = 1 if smoking_history_ui in ["current", "former"] else 0
                risk_sim.input['smoking'] = float(smoking_val)
                
                # Compute
                risk_sim.compute()
                fuzzy_score = risk_sim.output.get("diabetes_risk", 50.0)
                st.metric(label="Fuzzy Risk Score", value=f"{fuzzy_score:.1f}")
            except Exception as e:
                st.metric(label="Fuzzy Risk Score", value="N/A")

        # Create risk visualization
        st.markdown("### Risk Visualization")
        progress_color = risk_color
        st.progress(float(xgb_prob[0]), text=f"Diabetes Risk: {xgb_prob[0]:.2%}")
        
        # Main prediction result with styling based on outcome
        if xgb_prob[0] >= 0.146:  # Use the training threshold of 0.146
            st.markdown(f"""
            <div class="result-box diabetic-box">
                <h3>⚠️ Assessment Result: <span style="color: #f44336;">HIGH RISK FOR DIABETES</span></h3>
                <p>Based on your clinical parameters, you may have diabetes or be at high risk for developing it.</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Consult a healthcare professional immediately for proper diagnosis</li>
                    <li>Consider comprehensive blood tests including fasting glucose and oral glucose tolerance test</li>
                    <li>Begin monitoring blood glucose levels regularly</li>
                    <li>Discuss medication options with your healthcare provider</li>
                    <li>Make significant lifestyle changes including diet and exercise</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif xgb_prob[0] >= 0.05:  # Pre-diabetic threshold
            st.markdown(f"""
            <div class="result-box prediabetic-box">
                <h3>ℹ️ Assessment Result: <span style="color: #ff9800;">MODERATE RISK (PRE-DIABETIC)</span></h3>
                <p>Based on your clinical parameters, you show signs of pre-diabetes, a condition that often precedes type 2 diabetes.</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Schedule an appointment with your healthcare provider</li>
                    <li>Consider routine blood tests to monitor glucose levels</li>
                    <li>Implement lifestyle modifications:</li>
                    <ul>
                        <li>Reduce refined carbohydrate and sugar intake</li>
                        <li>Increase physical activity (aim for 150 minutes per week)</li>
                        <li>Achieve and maintain a healthy weight</li>
                        <li>Monitor blood pressure and cholesterol</li>
                    </ul>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Low risk
            st.markdown(f"""
            <div class="result-box healthy-box">
                <h3>✅ Assessment Result: <span style="color: #4caf50;">LOW RISK</span></h3>
                <p>Based on your clinical parameters, you currently have a low risk of diabetes.</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Continue maintaining a healthy lifestyle</li>
                    <li>Have routine check-ups every 1-3 years</li>
                    <li>Focus on preventive measures:</li>
                    <ul>
                        <li>Balanced diet rich in fruits, vegetables, and whole grains</li>
                        <li>Regular physical activity</li>
                        <li>Maintain healthy weight</li>
                        <li>Avoid or quit smoking</li>
                        <li>Limit alcohol consumption</li>
                    </ul>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Key risk factors visualization
        st.markdown("### Key Risk Factors")
        
        # Create columns for risk factor visualization
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        # HbA1c risk level
        with col_risk1:
            hba1c_orig = HbA1c_level  # Original value before normalization
            if hba1c_orig >= 6.5:
                hba1c_status = "High (Diabetic)"
                hba1c_color = "#f44336"
            elif hba1c_orig >= 5.7:
                hba1c_status = "Elevated (Pre-diabetic)"
                hba1c_color = "#ff9800"
            else:
                hba1c_status = "Normal"
                hba1c_color = "#4caf50"
                
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; text-align: center;">
                <h4>HbA1c Level</h4>
                <div style="font-size: 24px; font-weight: bold; color: {hba1c_color};">{hba1c_orig}%</div>
                <div style="color: {hba1c_color};">{hba1c_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Blood glucose risk level
        with col_risk2:
            if blood_glucose_level >= 126:
                bg_status = "High (Diabetic)"
                bg_color = "#f44336"
            elif blood_glucose_level >= 100:
                bg_status = "Elevated (Pre-diabetic)"
                bg_color = "#ff9800"
            else:
                bg_status = "Normal"
                bg_color = "#4caf50"
                
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; text-align: center;">
                <h4>Blood Glucose</h4>
                <div style="font-size: 24px; font-weight: bold; color: {bg_color};">{blood_glucose_level} mg/dL</div>
                <div style="color: {bg_color};">{bg_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # BMI risk level
        with col_risk3:
            if bmi >= 30:
                bmi_status = "Obese"
                bmi_color = "#f44336"
            elif bmi >= 25:
                bmi_status = "Overweight"
                bmi_color = "#ff9800"
            elif bmi >= 18.5:
                bmi_status = "Normal"
                bmi_color = "#4caf50"
            else:
                bmi_status = "Underweight"
                bmi_color = "#2196f3"
                
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; text-align: center;">
                <h4>BMI</h4>
                <div style="font-size: 24px; font-weight: bold; color: {bmi_color};">{bmi:.1f} kg/m²</div>
                <div style="color: {bmi_color};">{bmi_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional risk factors
        st.markdown("### Additional Risk Factors")
        
        col_add1, col_add2 = st.columns(2)
        
        with col_add1:
            # Age risk
            age_risk = "High" if age > 45 else "Moderate" if age > 35 else "Low"
            age_color = "#f44336" if age_risk == "High" else "#ff9800" if age_risk == "Moderate" else "#4caf50"
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                <h4>Age Factor</h4>
                <div>Age: <strong>{age} years</strong></div>
                <div>Risk Level: <span style="color: {age_color};">{age_risk}</span></div>
                <div><small>Risk increases with age, particularly after 45</small></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_add2:
            # Combined cardiovascular risk
            cv_risk = "High" if hypertension_ui == "Yes" and heart_disease_ui == "Yes" else \
                       "Moderate" if hypertension_ui == "Yes" or heart_disease_ui == "Yes" else "Low"
            cv_color = "#f44336" if cv_risk == "High" else "#ff9800" if cv_risk == "Moderate" else "#4caf50"
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                <h4>Cardiovascular Risk</h4>
                <div>Hypertension: <strong>{hypertension_ui}</strong> | Heart Disease: <strong>{heart_disease_ui}</strong></div>
                <div>Combined Risk: <span style="color: {cv_color};">{cv_risk}</span></div>
                <div><small>Cardiovascular conditions increase diabetes risk</small></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced Analytics Section
        st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Allow user to expand SHAP analysis
        with st.expander("📊 View Feature Importance Analysis"):
            st.markdown("### Feature Contribution Analysis (SHAP)")
            st.markdown("""
            This analysis shows how each factor contributes to your diabetes risk prediction.
            Positive values increase risk, negative values decrease risk.
            """)
            
            try:
                # Calculate SHAP values
                shap_values = explainer.shap_values(input_data_encoded)
                
                # Create a more readable DataFrame
                readable_features = {
                    'age': 'Age',
                    'bmi': 'BMI',
                    'HbA1c_level': 'HbA1c Level',
                    'blood_glucose_level': 'Blood Glucose',
                    'gender_Male': 'Male Gender',
                    'hypertension_1': 'Hypertension',
                    'heart_disease_1': 'Heart Disease',
                    'smoking_history_current': 'Current Smoker',
                    'smoking_history_former': 'Former Smoker',
                    'smoking_history_never': 'Never Smoked'
                }
                
                # Create a more readable SHAP DataFrame
                shap_df = pd.DataFrame({
                    "Feature": [readable_features.get(f, f) for f in training_features],
                    "SHAP Value": shap_values[0],
                    "Absolute Impact": abs(shap_values[0])
                })
                
                # Sort by absolute impact
                shap_df = shap_df.sort_values("Absolute Impact", ascending=False)
                
                # Display top factors
                st.markdown("#### Top Factors Influencing Your Risk")
                
                # Create three columns for visualization
                col_shap1, col_shap2 = st.columns([2, 1])
                
                with col_shap1:
                    # Create a horizontal bar chart
                    for _, row in shap_df.iloc[:5].iterrows():
                        feature = row['Feature']
                        value = row['SHAP Value']
                        impact = "Increases" if value > 0 else "Decreases"
                        color = "#f44336" if value > 0 else "#4caf50"
                        
                        # Calculate width as percentage of maximum (normalized to 80% of column width)
                        max_value = shap_df["Absolute Impact"].max()
                        width_pct = abs(value) / max_value * 80
                        
                        # Create styled bar
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="margin-bottom: 5px;"><strong>{feature}</strong> {impact} risk</div>
                            <div style="display: flex; align-items: center;">
                                <div style="width: 50%; text-align: right; padding-right: 10px;">
                                    {"" if value > 0 else f"<div style='background-color: {color}; height: 20px; margin-left: {100-width_pct}%; width: {width_pct}%;'></div>"}
                                </div>
                                <div style="width: 0; height: 20px; border-right: 2px solid #333;"></div>
                                <div style="width: 50%; text-align: left; padding-left: 10px;">
                                    {"<div style='background-color: {color}; height: 20px; width: {width_pct}%;'></div>" if value > 0 else ""}
                                </div>
                            </div>
                            <div style="text-align: center; font-size: 12px; color: #666;">{abs(value):.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_shap2:
                    st.markdown("#### Interpretation")
                    
                    # Get top positive and negative factors
                    top_positive = shap_df[shap_df["SHAP Value"] > 0].iloc[:2]
                    top_negative = shap_df[shap_df["SHAP Value"] < 0].iloc[:2]
                    
                    if not top_positive.empty:
                        st.markdown("**Highest Risk Factors:**")
                        for _, row in top_positive.iterrows():
                            st.markdown(f"• {row['Feature']}")
                    
                    if not top_negative.empty:
                        st.markdown("**Protective Factors:**")
                        for _, row in top_negative.iterrows():
                            st.markdown(f"• {row['Feature']}")
                            
                    # Overall recommendation based on top factors
                    st.markdown("**Recommendation:**")
                    if not top_positive.empty:
                        st.markdown(f"Focus on managing your {top_positive.iloc[0]['Feature']} for maximum risk reduction.")
                
            except Exception as e:
                st.error(f"Could not generate SHAP analysis: {e}")
                
        # Debug options (hidden in production)
        with st.expander("🔧 Technical Details (for developers)"):
            st.markdown('<div class="debug-box">', unsafe_allow_html=True)
            st.markdown("### Technical Information")
            
            # Display raw model outputs
            st.write("**Raw XGBoost Probabilities:**", xgb_model.predict_proba(input_data_encoded)[0])
            
            # Add tabs for different technical information
            debug_tab1, debug_tab2, debug_tab3 = st.tabs(["Model Inputs", "Fuzzy System", "Raw Data"])
            
            with debug_tab1:
                st.write("**Encoded Features (Model Input):**")
                st.dataframe(input_data_encoded)
                
                st.write("**Feature Normalization:**")
                st.write(f"HbA1c Min: {hba1c_min}, HbA1c Max: {hba1c_max}")
                st.write(f"Normalized HbA1c: {normalized_hba1c[0]:.4f}")
            
            with debug_tab2:
                st.write("**Fuzzy System Outputs:**")
                try:
                    st.write(risk_sim.output)
                    
                    st.write("**Fuzzy System Consequents:**")
                    for consequent in risk_ctrl.consequents:
                        st.write(f"Consequent: {consequent.label}")
                    
                    st.write("**Fuzzy System Rules:**")
                    for i, rule in enumerate(risk_ctrl.rules):
                        st.write(f"Rule {i+1}: {rule}")
                except Exception as e:
                    st.write("Fuzzy system could not be fully evaluated.")
            
            with debug_tab3:
                st.write("**Raw Input Data:**")
                st.dataframe(input_data)
                
                st.write("**SHAP Values (All Features):**")
                try:
                    shap_df = pd.DataFrame({
                        "Feature": training_features,
                        "SHAP Value": shap_values[0]
                    })
                    st.dataframe(shap_df.sort_values("SHAP Value", ascending=False))
                except:
                    st.write("SHAP values could not be computed.")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ ADDITIONAL INFORMATION SECTIONS ------------------

# Educational Information
with st.expander("📚 Learn About Diabetes"):
    st.markdown('<div class="subheader">Understanding Diabetes</div>', unsafe_allow_html=True)
    
    # Create tabs for different educational content
    tab1, tab2, tab3, tab4 = st.tabs(["Diabetes Types", "Risk Factors", "Prevention", "Management"])
    
    with tab1:
        st.markdown("""
        ### Types of Diabetes
        
        **Type 1 Diabetes**
        - Autoimmune condition where the body attacks insulin-producing cells
        - Usually develops in childhood or adolescence
        - Requires insulin therapy for life
        
        **Type 2 Diabetes**
        - Most common form (90-95% of cases)
        - Body becomes resistant to insulin or doesn't produce enough
        - Often linked to lifestyle factors and genetics
        - Can often be managed with lifestyle changes and medication
        
        **Gestational Diabetes**
        - Develops during pregnancy
        - Usually resolves after childbirth
        - Increases risk of developing type 2 diabetes later
        
        **Prediabetes**
        - Blood glucose levels higher than normal but not high enough for diabetes diagnosis
        - Early intervention can prevent progression to type 2 diabetes
        """)
        
    with tab2:
        st.markdown("""
        ### Risk Factors
        
        **Non-modifiable Risk Factors:**
        - Age (risk increases after 45)
        - Family history of diabetes
        - Ethnicity (higher in certain populations)
        - History of gestational diabetes
        - Polycystic ovary syndrome
        
        **Modifiable Risk Factors:**
        - Overweight or obesity (especially abdominal obesity)
        - Physical inactivity
        - Unhealthy diet (high in processed foods, sugars)
        - Smoking
        - High blood pressure
        - Abnormal cholesterol levels
        - Stress and sleep problems
        """)
        
    with tab3:
        st.markdown("""
        ### Prevention Strategies
        
        **Diet Recommendations:**
        - Increase fiber intake (fruits, vegetables, whole grains)
        - Choose whole foods over processed foods
        - Limit added sugars and refined carbohydrates
        - Choose healthy fats (olive oil, avocados, nuts)
        - Control portion sizes
        
        **Physical Activity:**
        - Aim for 150 minutes of moderate exercise weekly
        - Include both aerobic exercise and strength training
        - Break up sedentary time with short activity breaks
        
        **Lifestyle Changes:**
        - Maintain a healthy weight or lose 5-7% of body weight if overweight
        - Quit smoking
        - Limit alcohol consumption
        - Manage stress through mindfulness, yoga, or other techniques
        - Get adequate sleep (7-8 hours nightly)
        """)
        
    with tab4:
        st.markdown("""
        ### Management Approaches
        
        **Monitoring:**
        - Regular blood glucose monitoring
        - HbA1c testing every 3-6 months
        - Regular medical check-ups
        
        **Medication:**
        - Oral medications (metformin, sulfonylureas, etc.)
        - Injectable medications (GLP-1 agonists)
        - Insulin therapy when needed
        
        **Self-Care:**
        - Consistent carbohydrate counting
        - Regular physical activity
        - Stress management
        - Proper foot care
        - Eye examinations
        
        **Support:**
        - Diabetes education programs
        - Support groups
        - Working with healthcare team (doctor, dietitian, diabetes educator)
        """)

# Recommendations based on risk level
with st.expander("🗓 Personalized Recommendations"):
    st.markdown('<div class="subheader">Next Steps Based on Risk Level</div>', unsafe_allow_html=True)
    
    # Create three columns for different risk levels
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; height: 100%;">
            <h3 style="color: #4caf50;">Low Risk</h3>
            <p><strong>Recommended Testing:</strong></p>
            <ul>
                <li>Blood glucose screening every 3 years</li>
                <li>Annual physical examination</li>
                <li>Regular blood pressure checks</li>
            </ul>
            <p><strong>Lifestyle Focus:</strong></p>
            <ul>
                <li>Maintain healthy weight</li>
                <li>Regular physical activity</li>
                <li>Balanced diet</li>
                <li>Avoid smoking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rec2:
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; height: 100%;">
            <h3 style="color: #ff9800;">Moderate Risk</h3>
            <p><strong>Recommended Testing:</strong></p>
            <ul>
                <li>Blood glucose screening annually</li>
                <li>HbA1c test every 6 months</li>
                <li>Lipid profile annually</li>
                <li>Regular blood pressure monitoring</li>
            </ul>
            <p><strong>Lifestyle Focus:</strong></p>
            <ul>
                <li>Aim for 5-7% weight loss if overweight</li>
                <li>150 minutes of physical activity weekly</li>
                <li>Reduce carbohydrate intake</li>
                <li>Stress management techniques</li>
                <li>Consider working with a dietitian</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rec3:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; height: 100%;">
            <h3 style="color: #f44336;">High Risk</h3>
            <p><strong>Recommended Testing:</strong></p>
            <ul>
                <li>Immediate consultation with healthcare provider</li>
                <li>Comprehensive diabetes screening</li>
                <li>HbA1c test every 3 months</li>
                <li>Regular blood glucose monitoring</li>
                <li>Kidney function tests</li>
                <li>Eye examination</li>
            </ul>
            <p><strong>Lifestyle Focus:</strong></p>
            <ul>
                <li>Structured weight management program</li>
                <li>Diabetes education program</li>
                <li>Medication as prescribed</li>
                <li>Carbohydrate counting</li>
                <li>Regular exercise with medical approval</li>
                <li>Consider diabetes prevention program</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footnote">
    <p><strong>Disclaimer:</strong> This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
    <p>Created with ❤️ for better health outcomes | Last updated: April 2025</p>
</div>
""", unsafe_allow_html=True)