# model_trainer.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("dataset/mental_health.csv")

# Step 2: Drop unnecessary columns
df = df.drop(columns=["comments", "state", "Timestamp"])

# Step 3: Drop rows with missing target
df = df.dropna(subset=["treatment"])

# Step 4: Fill other missing values (categorical only)
df.fillna("Unknown", inplace=True)

# Step 5: Filter out invalid ages
df = df[(df["Age"] >= 10) & (df["Age"] <= 100)]

# Step 6: Create LabelEncoder dictionary
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 7: Prepare features and target
X = df.drop(columns=["treatment"])
y = df["treatment"]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 10: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/treatment_predictor.pkl")
print("\n✅ Model saved successfully to models/treatment_predictor.pkl")

# Step 12: Save all LabelEncoders
encoder_dir = "models/encoders"
os.makedirs(encoder_dir, exist_ok=True)

for col, le in label_encoders.items():
    file_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
    joblib.dump(le, file_path)
    print(f"✅ Saved encoder for column: {col}")

import os
import joblib

# ✅ Make sure the directory exists
os.makedirs("models/encoders", exist_ok=True)

# ✅ Save each LabelEncoder one by one
for col, le in label_encoders.items():
    path = f"models/encoders/{col}_encoder.pkl"
    joblib.dump(le, path)

# ✅ Optional: Save all in one dictionary as well
joblib.dump(label_encoders, "models/encoders/label_encoders.pkl")

print("✅ All LabelEncoders saved to models/encoders/")
