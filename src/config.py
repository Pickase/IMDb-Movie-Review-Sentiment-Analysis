# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Imdb.xlsx")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

REVIEW_COL = "review"
LABEL_COL = "sentiment"

LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "logs.csv")
