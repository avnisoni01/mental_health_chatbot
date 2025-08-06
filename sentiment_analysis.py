from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Returns sentiment score: positive, negative, or neutral
    """
    scores = sid.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
