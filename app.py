import pickle
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Prediction Function
# -----------------------------
def predict_value(country, year, iso_alpha, iso_num, region, subregion,
                  opec, eu, oecd, cis, var, scaler_path, model_path):

    try:
        # Load scaler
        with open('/workspaces/Global-energy-debt-shock-/notebook/models/scaler.pkl', 'rb') as f1:
            scaler = pickle.load(f1)

        # Load model
        with open('/workspaces/Global-energy-debt-shock-/notebook/models/model.pkl', 'rb') as f2:
            model = pickle.load(f2)

        # Create input dataframe (MUST match training columns)
        input_data = pd.DataFrame([{
            'Country': country,
            'Year': year,
            'ISO3166_alpha3': iso_alpha,
            'ISO3166_numeric': iso_num,
            'Region': region,
            'SubRegion': subregion,
            'OPEC': opec,
            'EU': eu,
            'OECD': oecd,
            'CIS': cis,
            'Var': var
        }])

        # ⚠️ IMPORTANT: Encoding (if used during training)
        # Example:
        # encoder = pickle.load(open("encoder.pkl", "rb"))
        # input_data = encoder.transform(input_data)

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        return prediction[0]

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Energy / CO2 Predictor")

country = st.text_input("Country")
year = st.number_input("Year", 1900, 2100, 2020)

iso_alpha = st.text_input("ISO Alpha Code (e.g., IND)")
iso_num = st.number_input("ISO Numeric Code", 0)

region = st.text_input("Region")
subregion = st.text_input("SubRegion")

opec = st.selectbox("OPEC", [0, 1])
eu = st.selectbox("EU", [0, 1])
oecd = st.selectbox("OECD", [0, 1])
cis = st.selectbox("CIS", [0, 1])

var = st.text_input("Variable (e.g., Oil, Gas, Coal)")

if st.button("Predict"):
    scaler_path = "notebook/scaler.pkl"
    model_path = "notebook/model.pkl"

    result = predict_value(
        country, year, iso_alpha, iso_num,
        region, subregion, opec, eu, oecd, cis, var,
        scaler_path, model_path
    )

    if result is not None:
        st.success(f"Predicted Value: {result}")
    else:
        st.error("Prediction failed")