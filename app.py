import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load Model, Scaler, Encoder
# ==========================

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# Default mean values for auto-fill
# ==========================
MEAN_VALUES = {
    'Crown_Width_East_West': 5.486180,
    'Slope': 22.198898,
    'Soil_TN': 0.510635,
    'Soil_TP': 0.255100,
    'Soil_AP': 0.251220,
    'Soil_AN': 0.249344,
    'Menhinick_Index': 1.762232,
    'Gleason_Index': 2.963965,
    'Fire_Risk_Index': 0.509207
}

FEATURE_ORDER = [
    'DBH', 'Tree_Height', 'Crown_Width_North_South',
    'Crown_Width_East_West', 'Slope', 'Elevation', 'Temperature',
    'Humidity', 'Soil_TN', 'Soil_TP', 'Soil_AP', 'Soil_AN',
    'Menhinick_Index', 'Gleason_Index', 'Disturbance_Level',
    'Fire_Risk_Index'
]


# ==========================
# Streamlit Page Setup
# ==========================

st.set_page_config(
    page_title="Forest Health Prediction",
    page_icon="🌳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2d6a4f;'>🌳 Forest Tree Health Prediction Using KNN</h1>
    <p style='text-align: center; font-size: 18px; color: #1b4332;'>
        Aplikasi ini memprediksi status kesehatan pohon berdasarkan karakteristik pohon dan kondisi ekologis menggunakan algoritma KNN.
    </p>
    <hr style="border: 1px solid #95d5b2;">
    """,
    unsafe_allow_html=True
)

# ==========================
# User Inputs (7 main features)
# ==========================

st.subheader("Masukkan Data Pohon")

DBH = st.number_input("DBH (Diameter at Breast Height)", min_value=0.0, step=0.1)
Tree_Height = st.number_input("Tree Height (m)", min_value=0.0, step=0.1)
Crown_NS = st.number_input("Crown Width (North-South)", min_value=0.0, step=0.1)
Elevation = st.number_input("Elevation (m)", min_value=0.0, step=1.0)
Temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
Disturbance_Level = st.selectbox("Disturbance Level", [0, 1, 2], help="0 = Low, 1 = Medium, 2 = High")

# ==========================
# Predict Button
# ==========================

if st.button("🔍 Predict Health Status"):
    
    # Create input data with 16 features
    input_data = pd.DataFrame([{
        'DBH': DBH,
        'Tree_Height': Tree_Height,
        'Crown_Width_North_South': Crown_NS,
        'Elevation': Elevation,
        'Temperature': Temperature,
        'Humidity': Humidity,
        'Disturbance_Level': Disturbance_Level,
        # Auto-filled features
        'Crown_Width_East_West': MEAN_VALUES['Crown_Width_East_West'],
        'Slope': MEAN_VALUES['Slope'],
        'Soil_TN': MEAN_VALUES['Soil_TN'],
        'Soil_TP': MEAN_VALUES['Soil_TP'],
        'Soil_AP': MEAN_VALUES['Soil_AP'],
        'Soil_AN': MEAN_VALUES['Soil_AN'],
        'Menhinick_Index': MEAN_VALUES['Menhinick_Index'],
        'Gleason_Index': MEAN_VALUES['Gleason_Index'],
        'Fire_Risk_Index': MEAN_VALUES['Fire_Risk_Index']
    }])

    # Pastikan kolom & urutan sama seperti saat training
    input_data = input_data.reindex(columns=FEATURE_ORDER).fillna(0)

    # Scale input
    scaled = scaler.transform(input_data)


    # Predict
    prediction = model.predict(scaled)
    label = label_encoder.inverse_transform(prediction)[0]

    # ==========================
    # Output Formatting
    # ==========================

    colors = {
        "Healthy": "#2d6a4f",
        "Very Healthy": "#1b4332",
        "Unhealthy": "#d00000",
        "Sub-healthy": "#ff8800"
    }

    icons = {
        "Healthy": "🌿",
        "Very Healthy": "🌳",
        "Unhealthy": "🔥",
        "Sub-healthy": "🍂"
    }

    st.markdown(
        f"""
        <div style='text-align:center; padding:20px; border-radius:10px; 
            background-color:{colors.get(label, "#74c69d")}; color:white;'>
            <h2>Hasil Prediksi: {icons.get(label, "🌳")} {label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

