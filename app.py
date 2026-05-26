import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("🏠 House Price Prediction")

st.write("Enter house details below:")

# Inputs
area = st.number_input("Area (sqft)", min_value=100)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

# Prediction
if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")