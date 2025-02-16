import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
try:
    reviews_df = pd.read_csv("E:/NLP(Sentimate analysis project)/245_1.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    reviews_df = pd.DataFrame(columns=['reviews.text', 'sentiment'])

# Prepare the training data
X = reviews_df['reviews.text'].fillna("")
y = reviews_df.get('sentiment', pd.Series([1] * len(reviews_df)))  # Default neutral if sentiment column missing

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tfidf, y)

# Define keyword-based sentiment check
positive_words = ["good", "excellent", "amazing", "fantastic", "love", "wonderful", "perfect", "best", "satisfied", "awesome"]
negative_words = ["bad", "terrible", "worst", "awful", "hate", "poor", "disappointed", "horrible", "not worth", "waste", "don't like", "not good", "not happy", "not satisfied", "poor quality", "too expensive"]
neutral_words = ["okay", "fine", "average", "decent", "neutral", "moderate", "fair", "normal", "standard", "acceptable"]

# Function to check sentiment based on keywords
def get_sentiment_from_keywords(review):
    review_lower = review.lower()
    if any(word in review_lower for word in positive_words):
        return "Positive"
    elif any(word in review_lower for word in negative_words):
        return "Negative"
    elif any(word in review_lower for word in neutral_words):
        return "Neutral"
    return None

st.title("Amazon Review Sentiment Analysis")
st.write("Enter a review and get the sentiment prediction!")

user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input:
        # Check sentiment based on keywords
        sentiment = get_sentiment_from_keywords(user_input)
        
        # If no sentiment from keywords, use model prediction
        if sentiment is None:
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_map.get(prediction, "Unknown")
        
        # Display result
        st.markdown(f"### Sentiment: {sentiment}")
        
        # Background color based on sentiment
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "yellow"}
        color = color_map.get(sentiment, "gray")
        st.markdown(f"<div style='background-color:{color}; padding:10px;'>{sentiment} Review</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review!")
