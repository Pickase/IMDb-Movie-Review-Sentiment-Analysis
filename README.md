# IMDB Movie Review Sentiment Analysis  
A complete end-to-end Machine Learning project to classify IMDB movie reviews as **Positive** or **Negative** using TF-IDF and Support Vector Machine (SVM).

### 🔗 Live Demo  
https://imdb-movie-review-sentiment-analysis-wzwpo3nyn77uobfzud7v8m.streamlit.app/#single-review

---

## 1. Project Overview  
This project builds a sentiment classification system trained on IMDB movie reviews.  
The objective is to convert raw text into meaningful numerical representations using NLP preprocessing and TF-IDF vectorization, then train a robust SVM model for sentiment prediction.

The deployed Streamlit application allows users to:
- Analyze **single movie reviews**
- Upload CSV files for **batch sentiment prediction**

---

## 2. Features  
- Full NLP preprocessing pipeline  
- Text normalization (lowercasing, punctuation removal, contraction fix, etc.)  
- Stopword removal  
- Lemmatization  
- TF-IDF feature extraction  
- Linear SVM classifier  
- Streamlit app with two prediction modes:
  - **Single Review Sentiment Analysis**
  - **Batch CSV Sentiment Analysis**

---

## 3. Project Structure
```bash
imdb-sentiment-analysis/
│
├── app/
│ └── app.py # Streamlit interface
│
├── src/
│ ├── config.py # Paths and global configuration
│ ├── data_prep.py # Raw → Processed dataset (local only)
│ ├── features.py # NLP preprocessing functions
│ ├── pipelines.py # TF-IDF + SVM pipeline
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│ └── predict.py # Single/Batch prediction functions
│
├── data/
│ ├── raw/Imdb.xlsx # Raw dataset (local only)
│ └── processed/ # Processed data (ignored in deployment)
│
├── models/
│ └── sentiment_model.pkl # Final trained model
│
├── requirements.txt
└── README.md

```
---

## 4. Dataset  
The project uses an IMDB movie review dataset containing reviews and their sentiment labels (positive/negative).  
Dataset source (Google Sheets link provided by user):

**https://docs.google.com/spreadsheets/d/1gblqnEpfJPCeTX_a5v-4hPKomhF_Ozpq/edit?usp=sharing**

---

## 5. Training Pipeline  
The training workflow includes:

### **NLP Preprocessing**
- Lowercasing  
- Removing punctuation  
- Removing extra spaces  
- Expanding contractions  
- Removing stopwords  
- Lemmatization  

### **Vectorization**
- TF-IDF with up to n-grams  
- Smooth IDF and sublinear TF scaling  

### **Model**
- Support Vector Machine (LinearSVC)

---

## 6. Performance  
Typical performance achieved:

| Metric | Score |
|--------|--------|
| Accuracy | 88–92% |
| F1 Score | High for both classes |

---

## 7. Streamlit Application  
The deployed web app supports:

### **Single Review Mode**  
Enter a review → model returns **Positive** or **Negative**.

### **Batch Prediction Mode**  
Upload CSV containing a `review` column → app returns a downloadable CSV with sentiment predictions.

---

## 8. How to Run Locally

### **1. Clone Repository**
```bash
git clone <your_repo_link>
cd imdb-sentiment-analysis

2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Train Model (optional)

python -m src.train

5. Run Streamlit App

streamlit run app/app.py

9. Deployment

The project is deployed using Streamlit Cloud.

Only code + model are deployed.
Large datasets remain local and are not required for inference.
10. Future Improvements

    Add probability/confidence scores

    Include visualization dashboards

    Add model comparison (Naive Bayes, Logistic Regression, etc.)

    Integrate database for storing user predictions

11. Author

Pranav Joshi
BSc IT Student | Data Science & Machine Learning Learner
