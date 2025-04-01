import streamlit as st
import pandas as pd
import numpy as np
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load the trained XGBoost model and imputer
try:
    with open('xgboost_high_risk_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('imputer_final.pkl', 'rb') as f:
        imputer = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: {str(e)}. Please ensure the model and imputer files are in the correct directory.")
    st.stop()

# Define the features used in the XGBoost model
features = ['monthly_temperature', 'pig_vaccinated', 'area', 'state_total_population', 'monthly_case_to_pop_ratio', 'pop_density']

# State data (population and area for each state)
state_data = {
    'Andhra Pradesh': {'population': 49577103, 'area': 162970},
    'Arunachal Pradesh': {'population': 1383727, 'area': 83743},
    'Assam': {'population': 31205576, 'area': 78438},
    'Bihar': {'population': 104099452, 'area': 94163},
    'Chandigarh': {'population': 1055450, 'area': 114},
    'Chhattisgarh': {'population': 25545198, 'area': 135192},
    'Goa': {'population': 1458545, 'area': 3702},
    'Gujarat': {'population': 60439692, 'area': 196244},
    'Haryana': {'population': 25351462, 'area': 44212},
    'Himachal Pradesh': {'population': 6864602, 'area': 55673},
    'Jharkhand': {'population': 32988134, 'area': 79716},
    'Karnataka': {'population': 61095297, 'area': 191791},
    'Kerala': {'population': 33406061, 'area': 38852},
    'Madhya Pradesh': {'population': 72626809, 'area': 308252},
    'Maharashtra': {'population': 112374333, 'area': 307713},
    'Manipur': {'population': 2855794, 'area': 22327},
    'Meghalaya': {'population': 2966889, 'area': 22429},
    'Mizoram': {'population': 1097206, 'area': 21081},
    'Nagaland': {'population': 1978502, 'area': 16579},
    'Odisha': {'population': 41974218, 'area': 155707},
    'Puducherry': {'population': 1247953, 'area': 490},
    'Punjab': {'population': 27743338, 'area': 50362},
    'Rajasthan': {'population': 68548437, 'area': 342239},
    'Sikkim': {'population': 610577, 'area': 7096},
    'Tamil Nadu': {'population': 72147030, 'area': 130060},
    'Telangana': {'population': 35003674, 'area': 112077},
    'Tripura': {'population': 3673917, 'area': 10486},
    'Uttar Pradesh': {'population': 199812341, 'area': 243290},
    'Uttarakhand': {'population': 10086292, 'area': 53483},
    'West Bengal': {'population': 91276115, 'area': 88752},
}

# Define fuzzy logic system with updated ranges
monthly_temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'monthly_temperature')
pop_density = ctrl.Antecedent(np.arange(0, 5001, 50), 'pop_density')
pig_vaccinated = ctrl.Antecedent(np.arange(0, 100001, 1000), 'pig_vaccinated')
state_total_population = ctrl.Antecedent(np.arange(0, 200000001, 1000000), 'state_total_population')
monthly_case_to_pop_ratio = ctrl.Antecedent(np.arange(0, 1e-4, 1e-6), 'monthly_case_to_pop_ratio')  # Adjusted range

# Output variable
outbreak_risk = ctrl.Consequent(np.arange(0, 101, 1), 'outbreak_risk')

# Membership functions for inputs
monthly_temperature.automf(3, names=['poor', 'average', 'good'])
pop_density.automf(3, names=['poor', 'average', 'good'])
pig_vaccinated.automf(3, names=['poor', 'average', 'good'])
state_total_population.automf(3, names=['poor', 'average', 'good'])
monthly_case_to_pop_ratio.automf(3, names=['poor', 'average', 'good'])

# Membership functions for output
outbreak_risk['no_risk'] = fuzz.trimf(outbreak_risk.universe, [0, 0, 20])
outbreak_risk['low'] = fuzz.trimf(outbreak_risk.universe, [10, 30, 50])
outbreak_risk['high'] = fuzz.trimf(outbreak_risk.universe, [40, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(monthly_temperature['poor'] & pop_density['good'] & monthly_case_to_pop_ratio['good'], outbreak_risk['high'])
rule2 = ctrl.Rule(pig_vaccinated['poor'] & state_total_population['good'], outbreak_risk['high'])
rule3 = ctrl.Rule(monthly_temperature['good'] & pop_density['poor'] & monthly_case_to_pop_ratio['poor'] & pig_vaccinated['good'], outbreak_risk['no_risk'])
rule4 = ctrl.Rule(pig_vaccinated['good'], outbreak_risk['low'])
rule5 = ctrl.Rule(monthly_temperature['average'] & monthly_case_to_pop_ratio['average'], outbreak_risk['low'])
rule6 = ctrl.Rule(pop_density['average'] & pig_vaccinated['average'], outbreak_risk['low'])
rule7 = ctrl.Rule(monthly_case_to_pop_ratio['good'], outbreak_risk['high'])  # New rule

# Create control system
outbreak_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
outbreak_sim = ctrl.ControlSystemSimulation(outbreak_ctrl)

# Streamlit app
st.title("Influenza Outbreak Prediction System (Hybrid XGBoost + Fuzzy Logic)")

# Input fields for features
st.header("Input Features")
state = st.selectbox("Select State", options=list(state_data.keys()), index=list(state_data.keys()).index('West Bengal'))
monthly_temperature_val = st.number_input("Monthly Temperature (°C)", min_value=0.0, max_value=40.0, value=20.0)
pig_vaccinated_val = st.number_input("Number of Pigs Vaccinated", min_value=0.0, max_value=100000.0, value=5000.0)
cases_reported_val = st.number_input("Monthly Cases Reported in the State", min_value=0.0, max_value=1000000.0, value=100.0)

# Get state-specific data
state_info = state_data.get(state, {'population': 0, 'area': 0})
state_total_population_val = state_info['population']
area_val = state_info['area']

# Compute pop_density and monthly_case_to_pop_ratio
pop_density_val = state_total_population_val / area_val if area_val > 0 else 0.0
monthly_case_to_pop_ratio_val = cases_reported_val / state_total_population_val if state_total_population_val > 0 else 0.0

# Display computed values
st.write(f"State Total Population: {state_total_population_val:,}")
st.write(f"Area (km²): {area_val:,}")
st.write(f"Computed Population Density (people per km²): {pop_density_val:.2f}")
st.write(f"Computed Monthly Case-to-Population Ratio (cases per person): {monthly_case_to_pop_ratio_val:.6f}")

# Initialize session state for risk scores if not already set
if "fuzzy_risk_score" not in st.session_state:
    st.session_state.fuzzy_risk_score = None
if "xgb_risk_score" not in st.session_state:
    st.session_state.xgb_risk_score = None
if "final_risk_score" not in st.session_state:
    st.session_state.final_risk_score = None
if "risk_label" not in st.session_state:
    st.session_state.risk_label = None
# Debug checkbox (defined before prediction sections)
debug = st.checkbox("Show Debug Values")

# Display debug scores if they exist
if debug:
    if "fuzzy_risk_score" in st.session_state and st.session_state.fuzzy_risk_score is not None:
        st.write(f"Fuzzy Loic Risk Score: {st.session_state.fuzzy_risk_score:.2f}")
        st.write(f"XGBoost Risk Score: {st.session_state.xgb_risk_score:.2f}")
        st.write(f"Combined Risk Score: {st.session_state.final_risk_score:.2f}")
    else:
        st.write("No prediction scores available. Please click 'Predict Outbreak' first.")

# Prediction button
if st.button("Predict Outbreak"):
    # Input validation
    if any(x < 0 for x in [monthly_temperature_val, pig_vaccinated_val, area_val, state_total_population_val, monthly_case_to_pop_ratio_val, cases_reported_val]):
        st.error("Input values cannot be negative.")
        st.stop()

    # Prepare input for XGBoost model
    input_data = pd.DataFrame({
        'monthly_temperature': [monthly_temperature_val],
        'pig_vaccinated': [pig_vaccinated_val],
        'area': [area_val],
        'state_total_population': [state_total_population_val],
        'monthly_case_to_pop_ratio': [monthly_case_to_pop_ratio_val],
        'pop_density': [pop_density_val],
        'name': [state]
    })

    # One-hot encode the 'name' column to match the model's expected features
    input_data_encoded = pd.get_dummies(input_data, columns=['name'], prefix='name')

    # Ensure all expected one-hot encoded columns are present
    expected_columns = model.get_booster().feature_names
    input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

    # Apply imputer to the input data
    try:
        input_data_imputed = pd.DataFrame(imputer.transform(input_data_encoded), columns=input_data_encoded.columns)
    except Exception as e:
        st.error(f"Error in preprocessing input data: {str(e)}")
        st.stop()

    # Get XGBoost prediction
    try:
        xgb_prob = model.predict_proba(input_data_imputed)[:, 1][0]  # Probability of high risk
        xgb_risk_score = xgb_prob * 100  # Scale to 0-100
    except Exception as e:
        st.error(f"Error in XGBoost prediction: {str(e)}")
        st.stop()

    # Get Fuzzy Logic prediction
    try:
        outbreak_sim.input['monthly_temperature'] = monthly_temperature_val
        outbreak_sim.input['pop_density'] = pop_density_val
        outbreak_sim.input['pig_vaccinated'] = pig_vaccinated_val
        outbreak_sim.input['state_total_population'] = state_total_population_val
        outbreak_sim.input['monthly_case_to_pop_ratio'] = monthly_case_to_pop_ratio_val
        outbreak_sim.compute()
        fuzzy_risk_score = outbreak_sim.output['outbreak_risk']
    except Exception as e:
        st.error(f"Error in Fuzzy Logic computation: {str(e)}")
        st.stop()

# Combined score
    final_risk_score_wb = (xgb_risk_score + fuzzy_risk_score) / 2

    # Define three risk levels for West Bengal
    if final_risk_score_wb < 5:
        risk_label_wb = "No Risk"
        risk_color_wb = "green"
        risk_message_wb = "The risk of an influenza outbreak in West Bengal is very low."
    elif final_risk_score_wb < 20:
        risk_label_wb = "Low Risk"
        risk_color_wb = "yellow"
        risk_message_wb = "The risk of an influenza outbreak in West Bengal is low. Monitor conditions."
    else:
        risk_label_wb = "High Risk"
        risk_color_wb = "red"
        risk_message_wb = "The risk of an influenza outbreak in West Bengal is high. Immediate action may be needed."

    st.markdown(f"#### Outbreak Risk Classification: <span style='color:{risk_color_wb}; font-weight:bold;'>{risk_label_wb}</span>", unsafe_allow_html=True)
    st.write(risk_message_wb)
    