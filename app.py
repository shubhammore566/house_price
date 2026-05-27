import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

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

    # Dataset Information
    st.subheader("📌 Dataset Information")

    rows = df.shape[0]
    cols = df.shape[1]

    st.write(f"Rows: {rows}")
    st.write(f"Columns: {cols}")

    # Show Columns
    st.subheader("📋 Column Names")
    st.write(df.columns)

    # Handle Missing Values
    df = df.dropna()

    # Convert total_sqft to numeric
    df["total_sqft"] = pd.to_numeric(
        df["total_sqft"],
        errors="coerce"
    )

    df = df.dropna()

    # Scatter Plot
    st.subheader("📊 Scatter Plot (Area vs Price)")

    scatter_data = pd.DataFrame({
        "area": df["total_sqft"],
        "price": df["price"]
    })

    st.scatter_chart(
        scatter_data,
        x="area",
        y="price"
    )

    # Encode Categorical Columns
    le = LabelEncoder()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

    # Features and Target
    X = df.drop("price", axis=1)

    y = df["price"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = r2_score(y_test, y_pred) * 100

    # Correlation Matrix
    st.subheader("📈 Correlation Matrix")

    st.dataframe(df.corr(numeric_only=True))

    # Accuracy
    st.subheader("✅ Model Accuracy")

    st.success(
        f"Accuracy: {accuracy:.2f}%"
    )

    # Dataset Quality
    if accuracy > 90:

        st.info("🔥 Excellent Dataset")

    elif accuracy > 70:

        st.info("👍 Good Dataset")

    else:

        st.warning("⚠️ Poor Dataset")

    # Prediction Section
    st.subheader("🏠 Predict House Price")

    total_sqft = st.number_input(
        "Total Square Feet"
    )

    bath = st.number_input(
        "Bathrooms"
    )

    balcony = st.number_input(
        "Balcony"
    )

    # Predict Button
    if st.button("Predict Price"):

        input_data = np.array([
            [
                0,
                0,
                0,
                0,
                total_sqft,
                bath,
                balcony
            ]
        )

        prediction = model.predict(
            input_data
        )

        st.success(
            f"Predicted House Price: ₹ {prediction[0]:,.2f}"
        )
