import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download once
nltk.download('vader_lexicon')

# create analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return analyzer.polarity_scores(text)


