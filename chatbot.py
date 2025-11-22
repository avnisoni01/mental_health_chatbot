import joblib
import nltk
from sentiment_analysis import analyze_sentiment

# Download VADER (only first time)
nltk.download('vader_lexicon')

# Load model + vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("🤖 Mental Health Assessment Chatbot")
print("I will ask you a few questions. Type 'exit' to stop anytime.\n")

# Questions to ask user
questions = [
    "How are you feeling today?",
    "Do you often feel stressed or anxious?",
    "How is your sleep quality recently?",
    "Do you feel supported by friends or family?",
    "Are you struggling with anything mentally these days?"
]

def collect_answers():
    answers = []
    for q in questions:
        user_input = input(f"{q}\nYou: ")

        if user_input.lower() == "exit":
            print("👋 Take care! Ending assessment.")
            exit()

        answers.append(user_input)

    return answers

def analyze(answers):
    combined_text = " ".join(answers)

    # Vectorize for ML model
    X = vectorizer.transform([combined_text])
    prediction = model.predict(X)[0]

    # Sentiment scoring
    sentiment = analyze_sentiment(combined_text)

    return prediction, sentiment, combined_text

def show_report(prediction, sentiment, combined_text):
    print("\n🧾 ----- FINAL ASSESSMENT REPORT -----")

    print("\n📝 Your combined responses:")
    print(f"{combined_text}\n")

    print("📊 Sentiment Score:")
    print(sentiment)

    # ML prediction
    if prediction == 1:
        print("\n🔍 **ML Result:** You may need mental health support.")
        print("💡 It's okay to feel this way. Consider talking to a therapist or loved one.")
    else:
        print("\n✅ **ML Result:** You seem mentally stable overall.")

    # Sentiment interpretation
    comp = sentiment["compound"]
    if comp <= -0.5:
        print("😞 Mood: Negative — You may be feeling low.")
    elif comp >= 0.5:
        print("😊 Mood: Positive — You seem in a good emotional state.")
    else:
        print("😐 Mood: Neutral — Your emotional tone is balanced.")

    print("\n-------------------------------------\n")


# MAIN LOOP
while True:
    print("🧠 Starting a new mental health assessment...\n")

    answers = collect_answers()
    prediction, sentiment, combined_text = analyze(answers)

    show_report(prediction, sentiment, combined_text)

    again = input("🔁 Do you want to take the assessment again? (yes/no): ").strip().lower()
    if again != "yes":
        print("👋 Thank you for using the Mental Health Chatbot. Stay safe! ❤️")
        break

    print("\n-------------------------------------\n")
