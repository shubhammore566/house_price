import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("housing.csv")

# Features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")