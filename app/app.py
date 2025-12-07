import streamlit as st
import pandas as pd
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.predict import predict_single, predict_batch

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis App")

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "About"])

# Single Review
with tab1:
    st.subheader("Analyze a Single Review")
    text = st.text_area("Enter review text:")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter text.")
        else:
            pred = predict_single(text)
            st.success(f"Prediction: {pred}")

# Batch Review
with tab2:
    st.subheader("Batch Prediction (CSV)")
    file = st.file_uploader("Upload CSV containing 'review' column", type="csv")

    if file:
        df = pd.read_csv(file)
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            result = predict_batch(df)
            st.dataframe(result)

            st.download_button(
                label="Download Predictions",
                data=result.to_csv(index=False).encode(),
                file_name="batch_predictions.csv"
            )

# About
with tab3:
    st.write("### IMDB Dataset Source:")
    st.write("https://docs.google.com/spreadsheets/d/1gblqnEpfJPCeTX_a5v-4hPKomhF_Ozpq/edit?usp=sharing")
    st.write("This dataset was used only for training. The deployed app does not load the dataset.")
