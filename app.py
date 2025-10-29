import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from joblib import load

# Setup NLTK data path
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required packages if not already present
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "stopwords":
            nltk.data.find("corpora/stopwords")
        else:  # punkt_tab
            nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)


# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load trained model & vectorizer
vectorizer = load('Model/vectorizer.joblib')
mnb = load('Model/model.joblib')

# Preprocessing function
def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special chars
    words = word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w not in stop_words]  # remove stopwords & stem
    return ' '.join(words)

# Streamlit UI
st.title("Sentiment Classifier")
st.write("Enter a sentence & get its sentiment prediction.")
st.write("This project was initially trained with movie reviews, but it can predict sentiments of almost any statement.")

user_input = st.text_area("Write Here:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to predict!")
    else:
        clean_input = preprocess(user_input)
        review_vec = vectorizer.transform([clean_input])
        prediction = mnb.predict(review_vec)
        
        if prediction[0] == 1:
            st.markdown("<span style='color:green'>Predicted Sentiment: Positive</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>Predicted Sentiment: Negative</span>", unsafe_allow_html=True)
