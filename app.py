import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load Model and Encoders
# -------------------------------

model = pickle.load(open("best_random_forest_model.pkl", "rb"))

le_weather = pickle.load(open("label_encoder_Weather.pkl", "rb"))
le_traffic = pickle.load(open("label_encoder_Traffic_Level.pkl", "rb"))
le_time = pickle.load(open("label_encoder_Time_of_Day.pkl", "rb"))
le_vehicle = pickle.load(open("label_encoder_Vehicle_Type.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚴",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------

st.markdown(
    "<h1 style='text-align:center;color:white;'>Food Delivery Time Predictor</h1>",
    unsafe_allow_html=True
)

st.write("Predict how long a food delivery will take based on order conditions.")

st.divider()

# -------------------------------
# Input Fields
# -------------------------------

distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

weather = st.selectbox(
    "Weather Condition",
    le_weather.classes_
)

traffic = st.selectbox(
    "Traffic Level",
    le_traffic.classes_
)

time_of_day = st.selectbox(
    "Time of Day",
    le_time.classes_
)

vehicle = st.selectbox(
    "Vehicle Type",
    le_vehicle.classes_
)

prep_time = st.number_input(
    "Food Preparation Time (minutes)",
    min_value=1,
    max_value=60,
    value=15
)

experience = st.number_input(
    "Courier Experience (years)",
    min_value=0,
    max_value=20,
    value=2
)

st.divider()

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Delivery Time"):

    weather_encoded = le_weather.transform([weather])[0]
    traffic_encoded = le_traffic.transform([traffic])[0]
    time_encoded = le_time.transform([time_of_day])[0]
    vehicle_encoded = le_vehicle.transform([vehicle])[0]

    features = np.array([[
        distance,
        weather_encoded,
        traffic_encoded,
        time_encoded,
        vehicle_encoded,
        prep_time,
        experience
    ]])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Delivery Time: **{round(prediction)} minutes** 🚀")