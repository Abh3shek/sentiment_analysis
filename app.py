# app.py
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from joblib import load  # For loading model/vectorizer

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Load trained model and vectorizer ---
vectorizer = load('Model/vectorizer.joblib')
mnb = load('Model/model.joblib')

# --- Preprocessing function ---
def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
    words = word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w not in stop_words]  # remove stopwords + stem
    return ' '.join(words)

# --- Streamlit UI ---
st.title("Sentiment Classifier")
st.write("Enter a sentence & get its sentiment prediction.")
st.write("This project was initially trained with movie reviews, but it can also be used to predict sentiments of almost any type of statements.")

user_input = st.text_area("Write Here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to predict!")
    else:
        clean_input = preprocess(user_input)
        review_vec = vectorizer.transform([clean_input])  # transform the input
        prediction = mnb.predict(review_vec)
        
        if prediction[0] == 1:
            st.markdown(f"<span style='color:green'>Predicted Sentiment: Positive</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'>Predicted Sentiment: Negative</span>", unsafe_allow_html=True)
