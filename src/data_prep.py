# src/data_prep.py

import pandas as pd
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, REVIEW_COL
from .features import clean_text

def prepare_data():
    df = pd.read_excel(RAW_DATA_PATH)

    print("Cleaning text...")
    df[REVIEW_COL] = df[REVIEW_COL].astype(str).apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Saved processed data at:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    prepare_data()
