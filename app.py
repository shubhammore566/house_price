import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("🏠 House Price Prediction System")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("📄 Dataset")

    st.dataframe(df)

    # Graph
    st.subheader("📊 Area vs Price Graph")

    chart_data = df[["area", "price"]]

    chart_data = chart_data.set_index("area")

    st.line_chart(chart_data)

    # Prediction Inputs
    st.subheader("🏠 Predict House Price")

    area = st.number_input("Area")

    bedrooms = st.number_input("Bedrooms")

    bathrooms = st.number_input("Bathrooms")

    floors = st.number_input("Floors")

    parking = st.number_input("Parking")

    age = st.number_input("Age")

    if st.button("Predict Price"):

        # Simple Prediction Formula
        prediction = (
            area * 3000 +
            bedrooms * 500000 +
            bathrooms * 300000 +
            floors * 200000 +
            parking * 150000 -
            age * 10000
        )

        st.success(
            f"Predicted House Price: ₹ {prediction:,.2f}"
        )

    # Correlation
    st.subheader("📈 Correlation Matrix")

    st.dataframe(df.corr())
