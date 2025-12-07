# app/app.py

import streamlit as st
import pandas as pd
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.predict import predict_single, predict_batch
from src.config import LOG_PATH, REVIEW_COL

st.set_page_config(page_title="Sentiment Analysis")

st.title("Sentiment Analysis App")

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Logs"])

# ---------------- TAB 1 --------------------
with tab1:
    st.subheader("Analyze a Single Review")

    text = st.text_area("Enter review text:")

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            label, confidence = predict_single(text)
            st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")

# ---------------- TAB 2 --------------------
with tab2:
    st.subheader("Batch Prediction from CSV")

    file = st.file_uploader("Upload CSV with 'review' column", type=["csv"])

    if file:
        df = pd.read_csv(file)
        result = predict_batch(df)
        st.dataframe(result)

        csv = result.to_csv(index=False).encode()
        st.download_button("Download Results", csv, "predictions.csv")


# ---------------- TAB 3 --------------------
with tab3:
    st.subheader("Logs (local)")

    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs)
    else:
        st.info("No logs found.")
