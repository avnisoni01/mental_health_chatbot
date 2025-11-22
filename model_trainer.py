import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("data/mental_health.csv")

print("File loaded successfully!")
print("Columns available:", df.columns)

# Clean dataset
df = df.dropna()
df = df[df['text'].str.strip() != ""]
df = df[df['label'].str.strip() != ""]

print("Columns after cleaning:", df.columns)

# Train-test split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split complete!")

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

print("Vectorization complete!")

# Model training
model = LinearSVC()
model.fit(X_train_vec, y_train)

print("Model training complete!")

# Save model + vectorizer
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n🎉 All done! Model and vectorizer saved successfully!")
print("Saved in: models/model.pkl and models/vectorizer.pkl")
