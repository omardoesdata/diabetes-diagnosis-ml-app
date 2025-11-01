import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model("diabetes_nn_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Medical Diabetes Prediction App")

# Input fields for user
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    # Arrange inputs
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    result = (prediction[0][0] > 0.5)

    if result:
        st.error("High chance of Diabetes")
    else:
        st.success("Low chance of Diabetes")
