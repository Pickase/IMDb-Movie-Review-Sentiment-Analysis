import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def handle_negation(text):
    # Convert "not good" → "not_good"
    return re.sub(r"\bnot (\w+)", r"not_\1", text)


def preprocess_text(text: str) -> str:
    """Full optimized preprocessing for sentiment analysis."""

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove HTML
    text = re.sub(r"<.*?>", "", text)

    # apply negation handler
    text = handle_negation(text)

    # keep letters only
    text = re.sub(r"[^a-zA-Z_ ]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # lemmatize + remove stopwords
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)
