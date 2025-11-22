from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Questions for assessment
questions = [
    "How are you feeling today?",
    "Do you often feel stressed or anxious?",
    "How is your sleep quality recently?",
    "Do you feel supported by friends or family?",
    "Are you struggling with anything mentally these days?"
]

# Keywords that indicate negative mental health
negative_keywords = [
    "sad", "depressed", "anxious", "stress", "stressed", "bad", "not good",
    "terrible", "no", "not at all", "alone", "lonely", "angry", "upset",
    "tired", "exhausted", "hurt", "scared", "worried"
]

analyzer = SentimentIntensityAnalyzer()

def calculate_risk_score(answers):
    score = 0
    for ans in answers:
        text = ans.lower()
        for word in negative_keywords:
            if word in text:
                score += 1
                break
    return score

def get_risk_level(score):
    if score <= 1:
        return "Stable", "🙂 You seem mostly okay, but monitor yourself."
    elif score <= 3:
        return "Mild Issues", "😟 You are showing some signs of stress. Take care."
    else:
        return "High Risk", "⚠️ You seem to be struggling a lot. It may help to talk to someone."

def start_assessment():
    print("\n🧠 Starting a new mental health assessment...\n")

    answers = []

    for q in questions:
        print(q)
        ans = input("You: ")
        answers.append(ans)

    print("\n🧾 ----- FINAL ASSESSMENT REPORT -----\n")

    # Show all answers
    print("📝 Your combined responses:")
    print(" ".join(answers))
    print()

    # Sentiment analysis
    combined = " ".join(answers)
    sentiment = analyzer.polarity_scores(combined)
    print("📊 Sentiment Score:")
    print(sentiment)
    print()

    # Risk scoring system
    risk_score = calculate_risk_score(answers)
    level, message = get_risk_level(risk_score)

    print(f"⚡ Risk Level: {level}")
    print(f"{message}\n")

    print("-------------------------------------")

    again = input("\n🔁 Do you want to take the assessment again? (yes/no): ")
    if again.lower() == "yes":
        start_assessment()
    else:
        print("👋 Take care! You can come back anytime.\n")

if __name__ == "__main__":
    start_assessment()
