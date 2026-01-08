import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# ------------------ Load model and label encoder ------------------
model = joblib.load("model.pkl")
le = pickle.load(open("label_encoder.pkl", "rb"))

# ------------------ App title ------------------
st.title("🌸 Iris Flower Prediction App")
st.write("""
This app predicts the **Iris flower species** based on sepal and petal measurements.
""")

# ------------------ Sidebar: User input ------------------
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    
    # Dictionary for displaying user input
    data = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    }
    
    # Features for prediction
features = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)


features, input_data = user_input_features()

# ------------------ Make prediction ------------------
prediction = model.predict(features)
species_name = le.inverse_transform(prediction)[0]  # Convert numeric to species
prediction_proba = model.predict_proba(features)

# ------------------ Display results ------------------
st.subheader("Input Features")
st.write(input_data)

st.subheader("Prediction")
st.write(f"🌼 The predicted Iris species is **{species_name}**")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
st.write(proba_df)

