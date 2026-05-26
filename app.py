import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("housing.csv")

# Features
X = df[['area', 'bedrooms', 'bathrooms']]

# Target
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Title
st.title("🏠 House Price Prediction")

st.write("Enter House Details")

# Inputs
area = st.number_input("Area", min_value=100)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

# Predict Button
if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
