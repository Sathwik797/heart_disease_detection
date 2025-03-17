import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart.csv")

# Splitting features and target
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable (1 = Disease, 0 = No Disease)
# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
