import streamlit as st
import numpy as np
import pandas as pd

# Page Title
st.title("🏠 House Price Prediction System")

st.write("Enter House Details")

# Inputs
area = st.number_input("Area (sqft)", min_value=100)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

# Predict Button
if st.button("Predict Price"):

    # Price Formula
    price = (
        area * 3000 +
        bedrooms * 500000 +
        bathrooms * 300000
    )

    # Success Message
    st.success(f"Predicted House Price: ₹ {price:,.2f}")

    # Create DataFrame for Graph
    data = pd.DataFrame({
        'Feature': ['Area', 'Bedrooms', 'Bathrooms'],
        'Value': [area, bedrooms, bathrooms]
    })

    # Bar Chart
    st.subheader("📊 House Features Graph")

    st.bar_chart(data.set_index('Feature'))

    # Line Chart
    st.subheader("📈 Price Trend")

    trend_data = pd.DataFrame({
        'Area': [500, 1000, 1500, 2000, 2500, 3000],
        'Price': [1500000, 3000000, 4500000, 6000000, 7500000, 9000000]
    })

    st.line_chart(trend_data.set_index('Area'))
