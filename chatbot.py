import joblib
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load model and encoders
model = joblib.load("models/treatment_predictor.pkl")
encoders = joblib.load("models/encoders/label_encoders.pkl")

# Load Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Questions dictionary
questions = {
    "Age": "🎂 What is your age?",
    "Gender": "🧑‍🤝‍🧑 What is your gender? (Male/Female/Other)",
    "Country": "🌍 Which country do you live in?",
    "self_employed": "💼 Are you self-employed? (Yes/No)",
    "family_history": "👨‍👩‍👧 Do you have a family history of mental illness? (Yes/No)",
    "work_interfere": "📊 How often does mental health interfere with your work? (Never/Sometimes/Often/Rarely)",
    "no_employees": "🏢 How many employees are in your company? (1-5/6-25/26-100/100-500/500+)",
    "remote_work": "🏠 Do you work remotely? (Yes/No)",
    "tech_company": "💻 Is your company a tech company? (Yes/No)",
    "benefits": "🎁 Does your employer provide mental health benefits? (Yes/No/Don't know)",
    "care_options": "🩺 Do you know the care options for mental health at your workplace? (Yes/No/Not sure)",
    "wellness_program": "🧘‍♂️ Does your employer offer a wellness program? (Yes/No/Don't know)",
    "seek_help": "🗣️ Does your workplace encourage seeking help for mental health? (Yes/No/Don't know)",
    "anonymity": "🙈 Is anonymity protected when seeking mental health treatment? (Yes/No/Don't know)",
    "leave": "🛌 How easy is it to take mental health leave? (Very easy/Somewhat easy/Somewhat difficult/Very difficult/Don't know)",
    "mental_health_consequence": "🤯 Would discussing mental health affect your career? (Yes/No/Maybe)",
    "phys_health_consequence": "🤒 Would discussing physical health affect your career? (Yes/No/Maybe)",
    "coworkers": "👥 Would you talk to coworkers about mental health? (Yes/No/Some of them)",
    "supervisor": "👔 Would you talk to your supervisor about mental health? (Yes/No/Some of them)",
    "mental_health_interview": "🎙️ Would you discuss mental health in a job interview? (Yes/No/Maybe)",
    "phys_health_interview": "🏥 Would you discuss physical health in a job interview? (Yes/No/Maybe)",
    "mental_vs_physical": "⚖️ Is mental health as important as physical health? (Yes/No/Don't know)",
    "obs_consequence": "👁️‍🗨️ Have you seen consequences for discussing mental health at work? (Yes/No)"
}

def collect_user_input():
    print("👋 Hello! I'm your Mental Health Support Chatbot.")
    name = input("💬 What's your name?\n> ").strip().title()
    print(f"\nHi {name}! Let's begin. Type 'exit' at any time to quit.\n")

    user_inputs = {}
    for field, question in questions.items():
        answer = input(f"{question}\n> ").strip()
        if answer.lower() == "exit":
            print("👋 Goodbye! Take care.")
            exit()
        user_inputs[field] = answer

    return name, user_inputs

def preprocess_inputs(user_inputs):
    df = pd.DataFrame([user_inputs])
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def encode_inputs(df):
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                print(f"⚠️ Unknown value for '{col}'. Please use one of the known options.")
                raise
    return df

def analyze_sentiment(user_inputs):
    combined_text = " ".join(user_inputs.values())
    sentiment = sentiment_analyzer.polarity_scores(combined_text)
    return sentiment

def summarize_report(name, prediction, sentiment):
    print("\n🧾 Summary Report:")
    print("-" * 40)
    print(f"👤 Name: {name}")
    print(f"🧠 Sentiment Score: {sentiment}")
    if prediction == 1:
        print("🔍 Prediction: May need treatment.")
    else:
        print("✅ Prediction: Seems mentally well.")
    print("-" * 40)

def main_loop():
    while True:
        name, user_inputs = collect_user_input()

        print("\n🔄 Analyzing your responses...")
        input_df = preprocess_inputs(user_inputs)
        sentiment = analyze_sentiment(user_inputs)

        if sentiment['compound'] <= -0.5:
            print("😟 You seem to have negative emotions. You're not alone. ❤️")
        elif sentiment['compound'] >= 0.5:
            print("😊 You seem positive! Great to see that. 🌟")
        else:
            print("😐 Neutral tone detected. Let's explore more.")

        # Encode and predict
        try:
            input_df = encode_inputs(input_df)
        except:
            print("❌ Couldn't predict due to unknown input. Please try again.")
            continue

        prediction = model.predict(input_df)[0]

        # Show result
        if prediction == 1:
            print("\n📢 You may benefit from seeking mental health treatment.")
            print("💡 It's okay to ask for help — consider a therapist or support group.")
        else:
            print("\n✅ You seem mentally well. Keep up your good habits!")

        # Show summary
        summarize_report(name, prediction, sentiment)

        again = input("\n🔁 Would you like to assess again? (yes/no):\n> ").strip().lower()
        if again != "yes":
            print("👋 Thank you for using the Mental Health Support Chatbot. Stay well!")
            break

if __name__ == "__main__":
    main_loop()
