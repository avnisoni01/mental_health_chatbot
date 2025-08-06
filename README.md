# 🧠 Mental Health Support Chatbot

A smart and emotionally intelligent **mental health chatbot** that interacts with users, detects sentiment, and predicts if they may require mental health treatment using a machine learning model.

---

## 💡 Project Overview

This chatbot guides users through a series of mental health-related questions. It analyzes their emotional tone using **VADER Sentiment Analysis** and uses a **Machine Learning classifier** trained on real-world mental health survey data to predict whether the user may need treatment.

---

## ⚙️ Features

- 🧠 **Sentiment Detection** using VADER
- 📊 **ML-based prediction** for mental health treatment
- 🧾 **Personalized summary report** for each session
- 💬 **Conversation memory** (remembers user's name)
- 🔁 **Looping interaction** with option to restart
- 🎨 Clean UI with emojis and clear prompts
- 🔒 No API used — works completely offline
- 📂 Trained on mental health dataset from Kaggle

---

## 🧪 Tech Stack

- **Python**
- `scikit-learn` – Model training and prediction
- `nltk` – Sentiment analysis (VADER)
- `pandas` – Data handling
- `joblib` – Model and encoder saving
- **VS Code / Jupyter Notebook** – Development environment

---

## 🗂️ Project Structure

mental_health_chatbot/
│
├── model_trainer.py # Trains ML model and saves encoders
├── chatbot.py # Main chatbot script
├── nltk_download.py # Downloads VADER sentiment lexicon
│
├── models/
│ ├── treatment_predictor.pkl # Trained model
│ └── encoders/
│ └── label_encoders.pkl # Saved LabelEncoders
│
├── README.md # Project info and instructions
└── requirements.txt # Python packages needed


---

## 🚀 How to Run the Chatbot

```bash
# Step 1: Create a virtual environment (Optional but recommended)
python -m venv chatbot-env

# Step 2: Activate the environment
# Windows
chatbot-env\Scripts\activate

# Step 3: Install required libraries
pip install -r requirements.txt

# Step 4: Download VADER Lexicon
python nltk_download.py

# Step 5: Train the model (only once)
python model_trainer.py

# Step 6: Run the chatbot
python chatbot.py

👤 Author
Name: avni soni

Project: mental health chatbot

GitHub: avnisoni01

