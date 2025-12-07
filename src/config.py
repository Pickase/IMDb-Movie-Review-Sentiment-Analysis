# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

# Used only during local training
RAW_DATA_PATH = "YOUR_GOOGLE_DRIVE_LINK"
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed_local.csv")  # local only

REVIEW_COL = "review"
LABEL_COL = "sentiment"
