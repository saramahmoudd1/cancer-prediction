import streamlit as st
import pandas as pd

from src.preprocessing import load_data
from src.predict import (
    predict_existing_patient,
    predict_uploaded_csv
)

st.set_page_config(
    page_title="Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Cancer Prediction using Gene Expression")

tab1, tab2 = st.tabs(
    [
        "Demo",
        "Predict New Patient"
    ]
)

# -------------------------
# Demo
# -------------------------

with tab1:

    st.header("Predict Existing Patient")

    data = load_data()

    samples = data["Sample"].tolist()

    selected = st.selectbox(
        "Select Sample",
        samples
    )

    if st.button("Predict Sample"):

        prediction, confidence, true = predict_existing_patient(selected)

        st.success(
            f"Prediction : {prediction}"
        )

        if confidence is not None:

            st.info(
                f"Confidence : {confidence*100:.2f}%"
            )

        st.write(
            f"True Label : {true}"
        )

        if prediction == true:

            st.success("✅ Correct Prediction")

        else:

            st.error("❌ Wrong Prediction")

# -------------------------
# Upload
# -------------------------

with tab2:

    st.header("Predict New Patient")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.write(df.head())

        if st.button("Predict New Patient"):

            prediction, confidence = predict_uploaded_csv(uploaded)

            st.success(
                f"Prediction : {prediction}"
            )

            if confidence is not None:

                st.info(
                    f"Confidence : {confidence*100:.2f}%"
                )