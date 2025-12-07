# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

# Used only during local training
RAW_DATA_PATH = "https://docs.google.com/spreadsheets/d/1gblqnEpfJPCeTX_a5v-4hPKomhF_Ozpq/edit?usp=sharing&ouid=117900499454272180226&rtpof=true&sd=true"
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed_local.csv")  # local only

REVIEW_COL = "review"
LABEL_COL = "sentiment"
