import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("House Price Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("Dataset")
    st.write(df.head())

    # Features and Target
    X = df[["Bedrooms", "Size_sqft", "Age_years"]]

    y = df["Price_Lakhs_INR"]

    # Train Model
    model = LinearRegression()

    model.fit(X, y)

    st.success("Model Trained Successfully")

    # User Inputs
    st.subheader("Enter House Details")

    bedrooms = st.number_input("Bedrooms", min_value=1)

    size_sqft = st.number_input("Size_sqft", min_value=100)

    age_years = st.number_input("Age_years", min_value=0)

    # Prediction
    if st.button("Predict Price"):

        prediction = model.predict(
            [[bedrooms, size_sqft, age_years]]
        )

        st.success(
            f"Predicted Price: {prediction[0]:.2f} Lakhs INR"
        )
