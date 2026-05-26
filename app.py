import streamlit as st
import numpy as np

st.title("🏠 House Price Prediction")

st.write("Enter House Details")

# Inputs
area = st.number_input("Area (sqft)", min_value=100)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

# Prediction Button
if st.button("Predict Price"):

    # Simple formula
    price = (
        area * 3000 +
        bedrooms * 500000 +
        bathrooms * 300000
    )

    st.success(f"Predicted House Price: ₹ {price:,.2f}")
