import streamlit as st
import pandas as pd
from tensorflow import keras

model = keras.models.load_model("best_model.h5", compile=False)

st.title("Car CO2 Emissions Prediction")

engine_size = st.number_input("Engine Size (L)", 0.5, 8.0, 2.0, 0.1)
cylinders = st.number_input("Number of Cylinders", 2, 16, 4, 1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])  
fuel_city = st.number_input("Fuel Consumption City (L/100km)", 1.0, 30.0, 8.0, 0.1)
fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100km)", 1.0, 30.0, 6.0, 0.1)
fuel_comb = st.number_input("Fuel Consumption Combined (L/100km)", 1.0, 30.0, 7.0, 0.1)
fuel_comb_mpg = st.number_input("Fuel Consumption Combined (MPG)", 1.0, 100.0, 35.0, 0.1)

fuel_type_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3}
fuel_type_encoded = fuel_type_map[fuel_type]

X_new = pd.DataFrame([[engine_size, cylinders, fuel_type_encoded,
                       fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg]],
                     columns=["ENGINESIZE", "CYLINDERS", "FUELTYPE",
                              "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY",
                              "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG"])

if st.button("Predict"):
    prediction = model.predict(X_new)
    st.success(f"Predicted CO2 Emissions: {float(prediction[0][0]):.2f} g/km")
