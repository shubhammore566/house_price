import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# Title
st.title("🏠 Smart House Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Limit Large Dataset
    if len(df) > 1000:
        df = df.head(1000)

    # Dataset
    st.subheader("📄 Dataset")

    st.dataframe(df.head(20))

    # Dataset Info
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])

    col2.metric("Columns", df.shape[1])

    # Numeric Columns
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Correlation
    st.subheader("📈 Correlation Matrix")

    st.dataframe(
        df[numeric_cols].corr()
    )

    # Scatter Plot
    st.subheader("📊 Scatter Plot")

    if len(numeric_cols) >= 2:

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_cols,
            key="x"
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_cols,
            key="y"
        )

        chart_data = df[[x_axis, y_axis]].dropna()

        st.scatter_chart(chart_data)

    # Prediction Section
    st.subheader("🏠 Predict House Price")

    area = st.number_input(
        "Area",
        min_value=0.0,
        value=1000.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0,
        value=2
    )

    floors = st.number_input(
        "Floors",
        min_value=0,
        value=1
    )

    parking = st.number_input(
        "Parking",
        min_value=0,
        value=1
    )

    age = st.number_input(
        "Age",
        min_value=0,
        value=5
    )

    # Predict Button
    if st.button("Predict Price"):

        prediction = (
            area * 3000 +
            bedrooms * 500000 +
            bathrooms * 300000 +
            floors * 200000 +
            parking * 100000 -
            age * 10000
        )

        st.success(
            f"🏷 Predicted Price : ₹ {prediction:,.2f}"
        )
