Mental Health Support Chatbot

A smart and emotionally intelligent mental health chatbot that interacts with users, detects sentiment, and predicts whether they may require mental health treatment using a machine learning model.

Project Overview

This chatbot guides users through a series of mental health-related questions. It analyzes emotional tone using VADER Sentiment Analysis and uses a Machine Learning classifier trained on real-world mental health survey data to predict if the user may need treatment.

Features

Sentiment Detection using VADER

ML-based Prediction for mental health treatment

Personalized Summary Report for each session

Conversation Memory (remembers user's name)

Looping Interaction with option to restart

Clean UI with emojis and clear prompts

Works offline – no API required

Trained on real mental health dataset from Kaggle

Tech Stack

Python

scikit-learn – Model training and prediction

nltk – Sentiment analysis (VADER)

pandas – Data handling

joblib – Model and encoder saving

VS Code / Jupyter Notebook – Development environment

Project Structure
mental_health_chatbot/
├── model_trainer.py         # Trains ML model and saves encoders
├── chatbot.py               # Main chatbot script
├── nltk_download.py         # Downloads VADER sentiment lexicon
├── assessment_chatbot.py    # Alternative assessment version
├── create_dataset.py        # Generates dataset file
├── models/
│   ├── model.pkl            # Trained ML model
│   └── vectorizer.pkl       # Saved vectorizer
├── data/
│   └── mental_health.csv    # Dataset used for training
├── README.md                # Project info and instructions
└── requirements.txt         # Required Python packages

How to Run the Chatbot
# Step 1: Clone the repository
git clone https://github.com/avnisoni01/mental_health_chatbot.git
cd mental_health_chatbot

# Step 2: Create a virtual environment (Optional but recommended)
python -m venv chatbot-env

# Step 3: Activate the environment
# Windows
chatbot-env\Scripts\activate
# Mac/Linux
source chatbot-env/bin/activate

# Step 4: Install required libraries
pip install -r requirements.txt

# Step 5: Download VADER Lexicon
python nltk_download.py

# Step 6: Train the model (only required once)
python model_trainer.py

# Step 7: Run the chatbot
python chatbot.py

Example Interaction
Starting a new mental health assessment...

How are you feeling today?
You: sad
Do you often feel stressed or anxious?
You: yes a lot
How is your sleep quality recently?
You: poor
Do you feel supported by friends or family?
You: sometimes
Are you struggling with anything mentally these days?
You: yes

----- FINAL ASSESSMENT REPORT -----

Your combined responses:
sad yes a lot poor sometimes yes

Sentiment Score:
{'neg': 0.45, 'neu': 0.30, 'pos': 0.25, 'compound': -0.65}

ML Result: You may benefit from seeking mental health treatment.
Mood: Negative — Emotional support recommended.

Author

Avni Soni

GitHub: avnisoni01

Project: Mental Health Chatbot