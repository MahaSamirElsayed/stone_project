import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# Load model & encoders
# ======================
model = joblib.load("gradient_boosting_model.joblib")
le_target = joblib.load("target_encoder.joblib")

cat_features = ['Sex','Urine_Crystals','Diabetes','Urine_culture','Organism','Bacteruria']

st.set_page_config(page_title="Urinary Stone Prediction", layout="centered")
st.title("🪨 Urinary Stone Composition Prediction")

st.write("Enter patient clinical data to predict stone composition")

# ======================
# User Inputs
# ======================
Sex = st.selectbox("Sex", ["Male", "Female"])
Stone_Burden = st.number_input("Stone Burden", min_value=0.0)
age = st.number_input("Age", min_value=0, max_value=120)
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0)
Urine_Crystals = st.selectbox("Urine Crystals", ["Yes", "No"])
Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
Urine_PH = st.number_input("Urine pH", min_value=4.0, max_value=9.0)
HU = st.number_input("Stone Density (HU)", min_value=0)
Urine_culture = st.selectbox("Urine Culture", ["Positive", "Negative"])
Organism = st.text_input("Organism", "None")
Bacteruria = st.selectbox("Bacteruria", ["Yes", "No"])
Urine_WBC = st.number_input("Urine WBC", min_value=0)

# ======================
# Prediction
# ======================
if st.button("Predict"):
    input_df = pd.DataFrame({
        'Sex':[Sex],
        'Stone_Burden':[Stone_Burden],
        'age':[age],
        'BMI':[BMI],
        'Urine_Crystals':[Urine_Crystals],
        'Diabetes':[Diabetes],
        'Urine_PH':[Urine_PH],
        'HU':[HU],
        'Urine_culture':[Urine_culture],
        'Organism':[Organism],
        'Bacteruria':[Bacteruria],
        'Urine_WBC':[Urine_WBC]
    })

    # Encode categorical
    for col in cat_features:
        input_df[col] = input_df[col].astype(str).factorize()[0]

    pred = model.predict(input_df)[0]
    pred_label = le_target.inverse_transform([pred])[0]

    st.success(f"Predicted Stone Composition: **{pred_label}**")
