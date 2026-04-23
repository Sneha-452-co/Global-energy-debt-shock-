import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# -------------------------------
# Load files safely
# -------------------------------
def load_file(path):
    if not os.path.exists(path):
        st.error(f" Missing file: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------------
# Prediction function
# -------------------------------
def predict_co2(country, year, coal, oil, gas, renew):

    scaler = load_file("models/scaler.pkl")
    model = load_file("models/model.pkl")
    encoder = load_file("models/encoder.pkl")

    if scaler is None or model is None or encoder is None:
        return None

    try:
        country_enc = encoder.transform([country])[0]
    except:
        st.error(" Country not in training data")
        return None

    # IMPORTANT: Must match training features EXACTLY
    data = pd.DataFrame({
        "Country": [country_enc],
        "Year": [year],
        "coal_consumption": [coal],
        "oil_consumption": [oil],
        "gas_consumption": [gas],
        "renewables_consumption": [renew]
    })

    try:
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        return pred
    except Exception as e:
        st.error(f" Prediction error: {e}")
        return None


# -------------------------------
# UI
# -------------------------------
st.title(" CO₂ Emission Predictor")

# Check models folder
if not os.path.exists("models"):
    st.error("'models' folder not found. Run training code first.")
else:
    st.success(" Models loaded successfully")

# Inputs
country = st.selectbox("Country", ["India", "United States", "China", "UK"])
year = st.number_input("Year", 1965, 2023, 2020)

coal = st.number_input("Coal Consumption", 0.0)
oil = st.number_input("Oil Consumption", 0.0)
gas = st.number_input("Gas Consumption", 0.0)
renew = st.number_input("Renewables Consumption", 0.0)

# Predict
if st.button("Predict CO₂ Emission"):
    scaler_path='/workspaces/Global-energy-debt-shock-/notebook/scaler.pkl'
    model_path='/workspaces/Global-energy-debt-shock-/notebook/model.pkl'

    result = predict_co2(country, year, coal, oil, gas, renew)

    if result is not None:
        st.success(f" Predicted CO₂: {result:.2f} MtCO₂")
        st.progress(min(result / 10000, 1.0))
    else:
        st.error(" Prediction failed")