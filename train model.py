import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("housing.csv")

# Features
X = df[["Bedrooms", "Size_sqft", "Age_years"]]

# Target
y = df["Price_Lakhs_INR"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully")
