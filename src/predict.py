import joblib
import pandas as pd
from .config import MODEL_PATH
from .features import preprocess_text

_model = None


def load_model():
    """Load and cache the model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_single(text: str):
    """Predict sentiment for a single review."""
    model = load_model()
    processed = preprocess_text(text)
    df = pd.DataFrame({"review": [processed]})
    pred = model.predict(df["review"])
    return pred[0]


def predict_batch(df: pd.DataFrame):
    """Predict sentiment for a batch."""
    model = load_model()

    df = df.copy()
    df["clean_review"] = df["review"].apply(preprocess_text)

    preds = model.predict(df["clean_review"])
    df["prediction"] = preds

    return df
