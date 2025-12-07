# src/evaluate.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from .config import PROCESSED_DATA_PATH, MODEL_PATH, REVIEW_COL, LABEL_COL
import joblib

def evaluate():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    preds = model.predict(df[REVIEW_COL])

    print("Accuracy:", accuracy_score(df[LABEL_COL], preds))
    print(classification_report(df[LABEL_COL], preds))

if __name__ == "__main__":
    evaluate()
    print("Evaluation complete.")
