import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler



 #Load model
model = load_model("../models/diabetes_nn_model.keras")

# Load data from DB to fit scaler
engine = create_engine("sqlite:///../data/medical.db")
df = pd.read_sql("SELECT * FROM patients", engine)

X = df.drop(columns=["outcome"])
scaler = StandardScaler()
scaler.fit(X)

# ---- USER INPUT ----
print("Enter patient medical data:")

preg = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
user_data_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_data_scaled)
pred_class = (prediction > 0.5).astype(int)[0][0]

if pred_class == 1:
    print("\n👉 **High risk of Diabetes** — Consult a doctor.")
else:
    print("\n **Low risk of Diabetes**")

print(f"Model Confidence: {prediction[0][0]:.4f}")