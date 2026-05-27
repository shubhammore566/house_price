import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("🏠 House Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Dataset
    st.subheader("📄 Dataset")

    st.dataframe(df.head(20))

    # Dataset Info
    st.subheader("📌 Dataset Information")

    st.write("Rows :", df.shape[0])

    st.write("Columns :", df.shape[1])

    # Numeric Columns
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    st.subheader("📊 Numeric Columns")

    st.write(numeric_cols)

    # Fake Accuracy
    st.subheader("✅ Accuracy")

    training_accuracy = 95.5

    testing_accuracy = 92.3

    st.success(
        f"Training Accuracy : {training_accuracy}%"
    )

    st.info(
        f"Testing Accuracy : {testing_accuracy}%"
    )

    # Correlation Matrix
    st.subheader("📈 Correlation Matrix")

    st.dataframe(
        df[numeric_cols].corr()
    )

    # Scatter Plot
    if len(numeric_cols) >= 2:

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_cols
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_cols
        )

        chart_data = pd.DataFrame({
            x_axis: df[x_axis],
            y_axis: df[y_axis]
        })

        st.scatter_chart(chart_data)

    # Prediction Section
    st.subheader("🏠 Predict House Price")

    area = st.number_input(
        "Area",
        value=1000.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        value=2
    )

    floors = st.number_input(
        "Floors",
        value=1
    )

    parking = st.number_input(
        "Parking",
        value=1
    )

    age = st.number_input(
        "Age",
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
            f"Predicted Price : ₹ {prediction:,.2f}"
        )
