import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("House Price Prediction App")

# Upload CSV File
uploaded_file = st.file_uploader(
    "Upload House Dataset CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("Dataset")
    st.write(df.head())

    # Show Shape
    st.subheader("Dataset Shape")
    st.write(df.shape)

    # Show Columns
    st.subheader("Columns")
    st.write(df.columns)

    # Features
    X = df[["Bedrooms", "Size_sqft", "Age_years"]]

    # Target
    y = df["Price_Lakhs_INR"]

    # Train Model
    model = LinearRegression()

    model.fit(X, y)

    st.success("Model Trained Successfully")

    # User Inputs
    st.subheader("Enter House Details")

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=2
    )

    size_sqft = st.number_input(
        "Size_sqft",
        min_value=100,
        value=1000
    )

    age_years = st.number_input(
        "Age_years",
        min_value=0,
        value=5
    )

    # Prediction Button
    if st.button("Predict House Price"):

        prediction = model.predict(
            [[bedrooms, size_sqft, age_years]]
        )

        st.success(
            f"Predicted House Price: {prediction[0]:.2f} Lakhs INR"
        )
