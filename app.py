import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Title
st.title("🏠 Smart House Price Prediction System")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Any House Price CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("📄 Dataset")

    st.dataframe(df)

    # Dataset Info
    st.subheader("📌 Dataset Information")

    st.write("Rows :", df.shape[0])

    st.write("Columns :", df.shape[1])

    # Select Numeric Columns
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Check Numeric Columns
    if len(numeric_cols) < 2:

        st.error(
            "Dataset must contain at least 2 numeric columns"
        )

    else:

        # Select Target Column
        target_column = st.selectbox(
            "Select Target Column (Price)",
            numeric_cols
        )

        # Features
        feature_columns = [
            col for col in numeric_cols
            if col != target_column
        ]

        X = df[feature_columns]

        y = df[target_column]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train Model
        model = LinearRegression()

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = r2_score(y_test, y_pred) * 100

        # Accuracy
        st.subheader("✅ Model Accuracy")

        st.success(f"Accuracy : {accuracy:.2f}%")

        # Scatter Chart
        st.subheader("📊 Scatter Plot")

        x_axis = st.selectbox(
            "Select X-axis",
            feature_columns
        )

        scatter_data = pd.DataFrame({
            x_axis: df[x_axis],
            target_column: df[target_column]
        })

        st.scatter_chart(
            scatter_data,
            x=x_axis,
            y=target_column
        )

        # Correlation Matrix
        st.subheader("📈 Correlation Matrix")

        st.dataframe(df[numeric_cols].corr())

        # Prediction Section
        st.subheader("🏠 Predict House Price")

        user_input = []

        for col in feature_columns:

            value = st.number_input(
                f"Enter {col}",
                value=0.0
            )

            user_input.append(value)

        # Predict Button
        if st.button("Predict Price"):

            prediction = model.predict(
                [user_input]
            )

            st.success(
                f"Predicted Price : ₹ {prediction[0]:,.2f}"
            )
