import streamlit as st
import pickle
import joblib
import pandas as pd

# ------------------ Load model & encoder ------------------
model = joblib.load("model.pkl")
le = pickle.load(open("label_encoder.pkl", "rb"))

# ------------------ App title ------------------
st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements and click **Predict** to see the result.")

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1
)

sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5
)

petal_length = st.sidebar.number_input(
    "Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4
)

petal_width = st.sidebar.number_input(
    "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2
)

# ------------------ Predict Button ------------------
predict_btn = st.sidebar.button("🔍 Predict")

# ------------------ Prediction ------------------
if predict_btn:
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )

    prediction = model.predict(features)
    species = le.inverse_transform(prediction)[0]

    prediction_proba = model.predict_proba(features)
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)

    st.subheader("Prediction Result")
    st.success(f"🌼 Predicted Iris Species: **{species}**")

    st.subheader("Prediction Probability")
    st.write(proba_df)

else:
    st.info("👈 Enter values in the sidebar and click **Predict**")
