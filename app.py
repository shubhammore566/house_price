import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# Title
st.title("🏠 Smart House Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Limit Dataset Size
    if len(df) > 1000:
        df = df.head(1000)

    # Show Dataset
    st.subheader("📄 Dataset")
    st.dataframe(df.head(20))

    # Dataset Info
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])

    col2.metric("Columns", df.shape[1])

    # Numeric Columns
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Check Columns
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

        # Model
        model = LinearRegression()

        # Train
        model.fit(X_train, y_train)

        # Accuracy
        train_accuracy = model.score(
            X_train,
            y_train
        ) * 100

        test_accuracy = model.score(
            X_test,
            y_test
        ) * 100

        # Accuracy Section
        st.subheader("✅ Model Accuracy")

        c1, c2 = st.columns(2)

        c1.success(
            f"Training Accuracy : {train_accuracy:.2f}%"
        )

        c2.info(
            f"Testing Accuracy : {test_accuracy:.2f}%"
        )

        # Correlation Matrix
        st.subheader("📈 Correlation Matrix")

        st.dataframe(
            df[numeric_cols].corr()
        )

        # Scatter Plot
        st.subheader("📊 Scatter Plot")

        x_axis = st.selectbox(
            "Select X-axis",
            feature_columns
        )

        chart_data = pd.DataFrame({
            x_axis: df[x_axis],
            target_column: df[target_column]
        })

        st.scatter_chart(chart_data)

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
                f"🏷 Predicted Price : ₹ {prediction[0]:,.2f}"
            )
