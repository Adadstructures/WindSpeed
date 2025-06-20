import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

model = joblib.load("final_model.joblib")
imputer = joblib.load("final_imputer.joblib")
scaler = joblib.load("scaler_wind_analysis.pkl")
features = joblib.load("selected_features.pkl")
features_scaled = joblib.load("features_scaled.pkl")
label_encoder_country = joblib.load("label_encoder_country.pkl")
label_encoder_city = joblib.load("label_encoder_city.pkl")


# === Feature Input Ranges ===
feature_ranges = {
    'tavg': (21.7, 34.0),
    'tmin': (10.8, 28.0),
    'tmax': (24.0, 56.0),
    'wdir': (0.0, 360.0),
    'pres': (1006.3, 1020.2),
    'month': (1, 12),
    'day': (1, 31)
}

# === Human-readable labels for inputs ===
feature_labels = {
    'tavg': 'Average Temperature (°C)',
    'tmin': 'Minimum Temperature (°C)',
    'tmax': 'Maximum Temperature (°C)',
    'wdir': 'Wind Direction (°)',
    'pres': 'Pressure (hPa)'
}

# === Country and City Mapping ===
country_city_map = {
    'Cameroon coastal wind speed': ['Dizangue', 'Douala'],
    'Equatorial Guinea': ['Malabo'],
    'Ghana': ['Accra'],
    'Nigeria': ['Lagos']
}

# === Streamlit UI ===
st.title("Wind Speed Predictor")

# === Date Selection (Allows Future Dates) ===
date_input = st.date_input("Select Date", value=datetime.today())
month = date_input.month
day = date_input.day

# === Country and City Selection ===
selected_country = st.selectbox("Select Country", list(country_city_map.keys()))
filtered_cities = country_city_map[selected_country]
selected_city = st.selectbox("Select City", filtered_cities)

# === Weather Feature Inputs (Descriptive Labels) ===
inputs = {}
for feature in ['tavg', 'tmin', 'tmax', 'wdir', 'pres']:
    min_val, max_val = feature_ranges[feature]
    label = feature_labels.get(feature, feature)
    inputs[feature] = st.number_input(
        f"{label} [{min_val}–{max_val}]", 
        min_value=float(min_val), 
        max_value=float(max_val), 
        value=float((min_val + max_val) / 2)
    )

# Add derived features
inputs['month'] = month
inputs['day'] = day
inputs['city_encoded'] = label_encoder_city.transform([selected_city])[0]
inputs['country_encoded'] = label_encoder_country.transform([selected_country])[0]

# === Predict Button ===
if st.button("Predict Wind Speed"):
    input_df = pd.DataFrame([inputs])
    X_selected = input_df[features]

    # Impute missing values
    X_imputed = imputer.transform(X_selected)

    # Predict in scaled form
    y_scaled_pred = model.predict(X_imputed)

    # Inverse scale wind speed only
    features_scaled_required = ['tavg', 'tmin', 'tmax', 'wdir', 'pres', 'wspd']
    dummy_df = pd.DataFrame(columns=features_scaled_required)
    dummy_df.loc[0] = 0  # initialize with zeros

    for feat in ['tavg', 'tmin', 'tmax', 'wdir', 'pres']:
        dummy_df.at[0, feat] = input_df[feat].values[0]
    dummy_df.at[0, 'wspd'] = y_scaled_pred[0]

    inv = scaler.inverse_transform(dummy_df)
    wspd_unscaled = inv[0, features_scaled_required.index('wspd')]

    st.success(f"Predicted Wind Speed: **{wspd_unscaled:.2f} m/s**")

   
