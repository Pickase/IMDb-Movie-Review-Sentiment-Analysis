# src/predict.py

import pandas as pd
import joblib
from .config import MODEL_PATH
from .features import preprocess_text

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_single(text):
    model = load_model()
    clean = preprocess_text(text)
    pred = model.predict([clean])[0]
    return pred

def predict_batch(df):
    model = load_model()
    df = df.copy()
    df["clean_review"] = df["review"].apply(preprocess_text)
    df["prediction"] = model.predict(df["clean_review"])
    return df
