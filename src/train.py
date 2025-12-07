# src/train.py

import pandas as pd
import joblib
from .config import PROCESSED_DATA_PATH, MODEL_PATH, REVIEW_COL, LABEL_COL
from .pipelines import build_pipeline

def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[REVIEW_COL]
    y = df[LABEL_COL]

    model = build_pipeline()

    print("Training model...")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_model()

    print("Training complete.")