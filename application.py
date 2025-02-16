import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- LOAD MODELS ---
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
    username_model = pickle.load(open("username_model.pkl", "rb"))  # Load the username model
    vectorizer_username = pickle.load(open("vectorizer_username.pkl", "rb"))  # Load the username vectorizer
    models_loaded = True
except FileNotFoundError:
    st.error("Model files not found. Make sure 'vectorizer.pkl', 'sentiment_model.pkl', 'username_model.pkl', and 'vectorizer_username.pkl' are in the same directory as the app.")
    models_loaded = False
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# --- PREDICTION FUNCTION ---
def predict_review(review_text):
    sentiment = 'N/A'
    username = 'Unknown'  # Default username if not predicted
    try:
        # Predict Sentiment
        review_vec = vectorizer.transform([review_text])
        sentiment = sentiment_model.predict(review_vec)[0]

        # Predict Username (if the model is available)
        if username_model and vectorizer_username:
            review_vec_username = vectorizer_username.transform([review_text])
            username = username_model.predict(review_vec_username)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 'Error', 'Error'  # Return error value for both sentiment and username

    return sentiment, username

# --- STREAMLIT APP ---
st.title("Sentiment Analysis with Username Prediction")

# Input Text Area
review_text = st.text_area("Enter your review here:", "")

# Prediction Button
if st.button("Predict"):
    if models_loaded:
        if review_text:
            predicted_sentiment, predicted_username = predict_review(review_text)
            st.write(f"Predicted Sentiment: {predicted_sentiment}")
            st.write(f"Predicted Username: {predicted_username}")
        else:
            st.warning("Please enter a review to predict.")
    else:
        st.error("Models not loaded. Please check the file paths.")
