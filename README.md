# Sentiment Classifier

A simple web application built with **Naive-Bayes, nltk & Streamlit** to predict the sentiment of text as **Positive** or **Negative**. Originally trained on movie reviews, but it can handle general text statements as well.

## Features

- Text preprocessing using **NLTK**:
  - Lowercasing
  - Removing HTML tags and special characters
  - Stopword removal
  - Stemming with **Porter Stemmer**
- Sentiment prediction using a pre-trained **Naive Bayes model**
- Real-time predictions via a **user-friendly Streamlit interface**

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Abh3shek/sentiment_analysis.git
   cd sentiment_classifier
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install Dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Ensure the `Model/` contains:

   ```
   - vectorizer.joblib
   - model.joblib
   ```

5. Run:
   `    streamlit run app.py
   `
   Open your browser at the URL provided (usually http://localhost:8501) and enter a text to get the sentiment prediction.
