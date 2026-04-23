import pickle
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# Prediction Function
# -------------------------------
def predict_co2(country, year, coal, oil, gas, renewables, scaler_path, model_path, encoder_path):
    try:
        # Load scaler
        with open(scaler_path, 'rb') as f1:
            scaler = pickle.load(f1)

        # Load model
        with open(model_path, 'rb') as f2:
            model = pickle.load(f2)

        # Load encoder
        with open(encoder_path, 'rb') as f3:
            encoder = pickle.load(f3)

        # Encode country
        country_encoded = encoder.transform([country])[0]

        # Create input dataframe
        data = {
            'Country': [country_encoded],
            'Year': [year],
            'coal_consumption': [coal],
            'oil_consumption': [oil],
            'gas_consumption': [gas],
            'renewables_consumption': [renewables]
        }

        X_new = pd.DataFrame(data)

        # Scale input
        X_scaled = scaler.transform(X_new)

        # Prediction
        prediction = model.predict(X_scaled)[0]

        return prediction

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="CO2 Emission Predictor", layout="centered")

st.title("🌍 Global Energy CO₂ Predictor")

st.markdown("Predict CO₂ emissions based on energy consumption")

# Inputs
country = st.text_input("Country (example: India, USA, China)", "India")
year = st.number_input("Year", min_value=1965, max_value=2023, value=2020)

coal = st.number_input("Coal Consumption", min_value=0.0, step=1.0)
oil = st.number_input("Oil Consumption", min_value=0.0, step=1.0)
gas = st.number_input("Gas Consumption", min_value=0.0, step=1.0)
renewables = st.number_input("Renewables Consumption", min_value=0.0, step=1.0)

# Button
if st.button("Predict CO₂ Emission"):

    scaler_path = "/workspaces/Global-energy-debt-shock-/notebook/models/scaler.pkl"
    model_path = "/workspaces/Global-energy-debt-shock-/notebook/models/model.pkl"
    encoder_path = ""

    result = predict_co2(
        country, year, coal, oil, gas, renewables,
        scaler_path, model_path, encoder_path
    )

    if result is not None:
        st.success(f"Predicted CO₂ Emission: {result:.2f} MtCO₂")

        # Progress bar (normalized for UI)
        st.progress(min(result / 10000, 1.0))

    else:
        st.error("Prediction failed. Check model files or inputs.")