import streamlit as st
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import joblib
import pandas as pd

# Load label encoder and model
model = load_model("sentiment_model.h5")
le = joblib.load("label_encoder.pkl")

# Load tokenizer using cleaned text (this part may need adjustment)
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["ID", "Entity", "Sentiment", "Text"]
df = df[df["Sentiment"] != "Irrelevant"]
df["clean_text"] = df["Text"].apply(lambda text: re.sub(r"[^a-zA-Z']", " ", str(text)).lower())
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])

def preprocess_input(text):
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    text = re.sub(r"\\s+", " ", text).strip()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    return padded

# Streamlit UI
st.title("Twitter Sentiment Analyzer")
user_input = st.text_area("Enter a tweet to analyze:")

if st.button("Predict Sentiment"):
    if user_input:
        padded = preprocess_input(user_input)
        pred = model.predict(padded)
        label = le.inverse_transform([np.argmax(pred)])
        st.success(f"Predicted Sentiment: {label[0]}")
    else:
        st.warning("Please enter a tweet.")
