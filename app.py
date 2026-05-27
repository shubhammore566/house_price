import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load Model
model = pickle.load(open('house_model.pkl', 'rb'))

st.title("House Price Prediction App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Dataset Information")
    st.write(df.describe())

    # Heatmap
    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap='YlGnBu',
        ax=ax
    )

    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Scatter Plot")

    fig2, ax2 = plt.subplots()

    ax2.scatter(df.iloc[:,1], df.iloc[:,-1])

    ax2.set_xlabel("Size")
    ax2.set_ylabel("Price")

    st.pyplot(fig2)

    # User Input
    st.subheader("Enter House Details")

    bedrooms = st.number_input("Bedrooms", min_value=1)

    size = st.number_input("Size Sqft", min_value=100)

    age = st.number_input("Age Years", min_value=0)

    # Prediction
    if st.button("Predict Price"):

        prediction = model.predict([[bedrooms, size, age]])

        st.success(f"Predicted House Price : {prediction[0]:.2f} Lakhs INR")
