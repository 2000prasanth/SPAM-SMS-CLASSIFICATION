import streamlit as st
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the WordNet Lemmatizer and stopwords
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model
loaded_model = joblib.load('naive_bayes_model.pkl')  # Replace 'naive_bayes_model.pkl' with the path to your trained Naive Bayes model

# Load the TF-IDF vectorizer used during training
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace 'tfidf_vectorizer.pkl' with the path to your TF-IDF vectorizer pickle file

# Define a function for preprocessing text
def preprocess_text(text):
    text = re.sub('[^A-Za-z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert text to lowercase
    words = word_tokenize(text)  # Tokenize text
    words = [lemma.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize words and remove stopwords
    return ' '.join(words)  # Join processed words back into a string

# Streamlit app
st.title("Text Classifier: Spam or Ham")

text_to_classify = st.text_input("Enter text to classify:", "You've won something")

if st.button("Classify"):
    # Preprocess and transform the text
    cleaned_text = preprocess_text(text_to_classify)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])

    # Make predictions
    prediction = loaded_model.predict(vectorized_text)

    # Convert prediction to human-readable label
    if prediction[0] == 'spam':
        result = "spam"
    else:
        result = "ham"

    st.write(f"The text is classified as: {result}")
    st.write(" \n Developed by Prasanth K V:  2000prasanth@protonmail.com")
