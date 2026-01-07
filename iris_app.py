import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model and label encoder
# -----------------------------
try:
    model = joblib.load("model.joblib")
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or label encoder file not found. Please check your paths.")
    st.stop()

# -----------------------------
# App title
# -----------------------------
st.title("🌸 Iris Flower Prediction App")
st.write("""
This app predicts the **Iris flower species** using sepal and petal measurements.
""")

# -----------------------------
# Sidebar: User input features
# -----------------------------
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    
    data = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    }
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features, data

features, input_data = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
try:
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    predicted_class = le.inverse_transform(prediction)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# -----------------------------
# Display results
# -----------------------------
st.subheader("Input Features")
st.write(input_data)

st.subheader("Prediction")
st.write(f"🌼 The predicted Iris species is **{predicted_class[0]}**")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
st.write(proba_df)
