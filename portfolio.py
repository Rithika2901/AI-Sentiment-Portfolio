import streamlit as st
import tensorflow as tf
import numpy as np
import os
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import openai

# Load trained sentiment model
model_path = "sentiment_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("🚨 Sentiment model not found! Make sure 'sentiment_model.h5' is uploaded.")
    st.stop()

# Load tokenizer (Ensure it's the same used during training)
tokenizer_path = "tokenizer.pickle"
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
else:
    st.warning("⚠️ Tokenizer file not found. Using default tokenizer.")
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

max_sequence_length = 200

# OpenAI API Key (Use environment variable or Streamlit Secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Function to preprocess user input
def preprocess_text(review):
    review = review.lower()
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    tokens = review.split()
    processed_review = " ".join(tokens)

    # Convert to sequence and pad
    review_seq = tokenizer.texts_to_sequences([processed_review])
    review_pad = pad_sequences(review_seq, maxlen=max_sequence_length, padding="post", truncating="post")
    
    return review_pad

# Function to predict sentiment
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prob = model.predict(processed_review)[0][0]
    sentiment = "Positive 😊" if prob > 0.5 else "Negative 😠"
    return sentiment, prob

# Function to generate AI-powered response using GPT-3.5
def generate_gpt_response(sentiment, review):
    prompt = f"""
    The following is a customer review: "{review}"
    The detected sentiment is: {sentiment}

    Based on this sentiment, generate a professional response:
    - If positive, express gratitude.
    - If negative, provide an empathetic apology.
    - If neutral, acknowledge the feedback.

    Keep the response short, professional, and engaging.
    """

    if openai.api_key is None:
        return "🚨 OpenAI API key missing! Set 'OPENAI_API_KEY' in Streamlit Secrets."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "⚠️ Error generating response. Please try again later."

# Streamlit Web App
st.title("🎭 AI-Powered Sentiment Analysis & Response Generator")
st.subheader("Analyze movie reviews and get AI-powered responses!")

# User input for movie review
user_review = st.text_area("Enter your movie review here:")

if st.button("Analyze Sentiment"):
    if user_review:
        sentiment, probability = predict_sentiment(user_review)
        response = generate_gpt_response(sentiment, user_review)

        st.write(f"### 🎯 Predicted Sentiment: {sentiment}")
        st.write(f"**Confidence Score:** {probability:.4f}")
        st.write(f"**🤖 AI Response:** {response}")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

