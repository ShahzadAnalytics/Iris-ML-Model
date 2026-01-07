import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# Load trained model and label encoder
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# App title
# -----------------------------
st.title("🌸 Iris Flower Prediction App")

# -----------------------------
# User Inputs (MAIN PAGE)
# -----------------------------
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Iris Species"):
    
    # Prepare input
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Decode class
    predicted_species = label_encoder.inverse_transform(prediction)[0]
    
    # -----------------------------
    # Display Results
    # -----------------------------
    st.success(f"🌼 Predicted Iris Species: **{predicted_species}**")
    
    # Probability table
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=label_encoder.classes_
    )
    
    st.subheader("Prediction Probability")
    st.write(proba_df)
