import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("House Price Prediction App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("Dataset")
    st.write(df.head())

    # Show Shape
    st.subheader("Dataset Shape")
    st.write(df.shape)

    # User Input
    st.subheader("Enter House Details")

    bedrooms = st.number_input("Bedrooms", min_value=1)

    size_sqft = st.number_input("Size_sqft", min_value=100)

    age_years = st.number_input("Age_years", min_value=0)

    # Prediction
    if st.button("Predict Price"):

        prediction = model.predict([[bedrooms, size_sqft, age_years]])

        st.success(
            f"Predicted House Price: {prediction[0]:.2f} Lakhs INR"
        )
