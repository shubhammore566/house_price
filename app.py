import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

    # Features
    X = df.drop("price", axis=1)

    # Target
    y = df["price"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = r2_score(y_test, y_pred)

    st.subheader("✅ Model Accuracy")

    st.success(f"Accuracy Score: {accuracy:.2f}")

    # Scatter Plot
    st.subheader("📊 Area vs Price")

    fig, ax = plt.subplots()

    ax.scatter(df["area"], df["price"])

    ax.set_xlabel("Area")

    ax.set_ylabel("Price")

    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("📈 Correlation Matrix")

    st.dataframe(df.corr())

    # Prediction Inputs
    st.subheader("🏠 Predict House Price")

    area = st.number_input("Area")

    bedrooms = st.number_input("Bedrooms")

    bathrooms = st.number_input("Bathrooms")

    floors = st.number_input("Floors")

    parking = st.number_input("Parking")

    age = st.number_input("Age")

    if st.button("Predict Price"):

        input_data = np.array([[
            area,
            bedrooms,
            bathrooms,
            floors,
            parking,
            age
        ]])

        prediction = model.predict(input_data)

        st.success(
            f"Predicted House Price: ₹ {prediction[0]:,.2f}"
        )
