import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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
    return "Neutral"  # Default to neutral if no keywords are found

# Train a simple Random Forest model with placeholder data
vectorizer = TfidfVectorizer(max_features=5000)
sample_reviews = ["good product", "bad experience", "just okay"]
sample_labels = [2, 0, 1]  # Positive = 2, Negative = 0, Neutral = 1
X_tfidf = vectorizer.fit_transform(sample_reviews)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tfidf, sample_labels)

st.title("Amazon Review Sentiment Analysis")
st.write("Enter a review and get the sentiment prediction!")

user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input:
        # Check sentiment based on keywords
        sentiment = get_sentiment_from_keywords(user_input)
        
        # If no sentiment from keywords, use model prediction
        if sentiment == "Neutral":
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_map.get(prediction, "Neutral")
        
        # Display result
        st.markdown(f"### Sentiment: {sentiment}")
        
        # Background color based on sentiment
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "yellow"}
        color = color_map.get(sentiment, "gray")
        st.markdown(f"<div style='background-color:{color}; padding:10px;'>{sentiment} Review</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review!")
