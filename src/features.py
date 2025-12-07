# src/features.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def handle_negation(text):
    return re.sub(r"\bnot (\w+)", r"not_\1", text)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    text = handle_negation(text)

    text = re.sub(r"[^a-zA-Z_ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]

    return " ".join(words)
