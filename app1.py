# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load models, scaler, and encoder (WITH PATHS)
model_path = r"C:\Users\nag15\OneDrive\Desktop\speech_to_text_app\best_xgb_model.pkl"  # Correct Path
scaler_path = r"C:\Users\nag15\OneDrive\Desktop\speech_to_text_app\scaler.pkl"      # Correct Path
encoder_path = r"C:\Users\nag15\OneDrive\Desktop\speech_to_text_app\onehot_encoder.pkl" # Correct Path


# Load models, scaler, and encoder
with open('best_xgb_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('onehot_encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

st.title("Machine Failure Prediction")

# Input fields (as before)
air_temp = st.number_input("Air Temperature [K]", value=30.0)
process_temp = st.number_input("Process Temperature [K]", value=40.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=3000)
torque = st.number_input("Torque [Nm]", value=50.0)
tool_wear = st.number_input("Tool Wear [min]", value=100)
twf = st.number_input("TWF", value=0, min_value=0, max_value=1)  # Corrected input type
hdf = st.number_input("HDF", value=0, min_value=0, max_value=1)
pwf = st.number_input("PWF", value=0, min_value=0, max_value=1)
osf = st.number_input("OSF", value=0, min_value=0, max_value=1)
rnf = st.number_input("RNF", value=0, min_value=0, max_value=1)

# Type input (using selectbox)
type_input = st.selectbox("Type", ["Type_L", "Type_M", "Type_H"])
type_df = pd.DataFrame({'Type': [type_input]})
encoded_type = encoder.transform(type_df[['Type']])
encoded_df = pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(['Type']))

# Feature engineering (same as training)
temp_diff = process_temp - air_temp
power_consumption = torque * rot_speed
tool_wear_interaction = tool_wear * rot_speed

# Create feature array
features = np.array([process_temp, air_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf,
                   temp_diff, power_consumption, tool_wear_interaction])

# Concatenate one-hot encoded features
features = np.concatenate([features, encoded_df.values[0]])
features = features.reshape(1, -1)

# Scale features
scaled_features = scaler.transform(features)

if st.button("Predict"):
    prediction_proba = best_model.predict_proba(scaled_features)[:, 1]
    prediction = (prediction_proba > 0.3).astype(int)  # You might want to adjust this threshold

    if prediction == 1:
        st.write("Prediction: Machine Failure (1)")
        st.write(f"Probability: {prediction_proba[0]:.2f}")
    else:
        st.write("Prediction: No Failure (0)")
        st.write(f"Probability: {prediction_proba[0]:.2f}")