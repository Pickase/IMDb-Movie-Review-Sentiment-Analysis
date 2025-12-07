import streamlit as st
import pandas as pd
import sys, os

# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import predict_single, predict_batch

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Classify reviews as **Positive** or **Negative** using a trained ML model.")

# -------------------------
# Sidebar Navigation
# -------------------------
mode = st.sidebar.radio("Choose Mode", ["Single Review", "Batch CSV"])

# -------------------------
# MODE 1 — Single Prediction
# -------------------------
if mode == "Single Review":
    st.subheader("Single Review")

    review = st.text_area("Enter your movie review:", height=200)

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review before predicting.")
        else:
            label = predict_single(review)

            # Convert model output
            if label == "positive":
                label = "Positive"
            else:
                label = "Negative"

            st.success(f"Sentiment: **{label}**")

# -------------------------
# MODE 2 — Batch Prediction
# -------------------------
else:
    st.subheader("Batch Review")

    st.write("""
    **Upload a CSV file** containing a column named **`review`**.
    The model will classify each review.
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "review" not in df.columns:
                st.error("CSV must contain a column named 'review'.")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("Analyze Batch"):
                    preds = predict_batch(df["review"].tolist())

                    # Fix label capitalization
                    preds = ["Positive" if p == "positive" else "Negative" for p in preds]

                    df["sentiment"] = preds
                    st.success("Batch Sentiment Analysis Completed!")

                    st.dataframe(df)

                    # Download processed file
                    csv_output = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_output,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
