import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
model = joblib.load("model.pkl")
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🌸 Iris Flower Prediction App")

st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# ✅ Use DataFrame (IMPORTANT)
features = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

prediction = model.predict(features)
species_name = le.inverse_transform(prediction)[0]

prediction_proba = model.predict_proba(features)
proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)

st.subheader("Prediction")
st.success(f"🌼 Predicted Iris Species: **{species_name}**")

st.subheader("Prediction Probability")
st.write(proba_df)
