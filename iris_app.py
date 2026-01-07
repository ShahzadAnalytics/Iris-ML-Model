import streamlit as st
import pickle
import numpy as np
import pandas as pd
# Load trained model and label encoder
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("iris_model.pkl", "rb"))


# App title
st.title("🌸 Iris Flower Prediction App")
st.write("""
This app predicts the **Iris flower species** using sepal and petal measurements.
# """)

# User inputs
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features, data

features, input_data = user_input_features()

# Prediction
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)

# Display results
st.subheader("Input Features")
st.write(input_data)

st.subheader("Prediction")
st.write(f"🌼 The predicted Iris species is **{prediction[0]}**")

st.subheader("Prediction Probability")

st.write(prediction_proba)


