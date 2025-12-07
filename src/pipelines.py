# src/pipelines.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=45000,
            sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC()))
    ])
