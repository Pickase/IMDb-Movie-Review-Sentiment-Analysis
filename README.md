# IMDb Movie Review Sentiment Analysis

This repository contains a **from-scratch** implementation of multiple NLP models for classifying IMDb movie reviews as **positive** or **negative**. The project covers data loading, preprocessing, feature engineering (Bag-of-Words, TF-IDF, embeddings), model training (Logistic Regression, Naive Bayes, SVM, Random Forest, LSTM), evaluation, and prediction.

## Table of Contents

- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Feature Engineering](#feature-engineering)  
- [Model Implementation](#model-implementation)  
- [Results](#results)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Insights & Observations](#insights--observations)  
- [License](#license)  

## Dataset

- **Source:** IMDb Movie Reviews  
- **Total Reviews:** 50,000 (25,000 positive, 25,000 negative)  
- **Labels:** `positive`, `negative`  

Split:  
- **Training Set:** 40,000 reviews (80%)  
- **Test Set:** 10,000 reviews (20%)  

## Preprocessing

1. **Text Cleaning**  
   - Remove HTML tags, punctuation, and special characters  
2. **Case Normalization**  
   - Convert all text to lowercase  
3. **Tokenization**  
   - Split each review into word tokens  
4. **Stop-Word Removal**  
   - Filter out common English stop words  
5. **Lemmatization & Stemming**  
   - Normalize tokens to their base forms  

## Feature Engineering

- **Bag-of-Words:** Count occurrences of each token  
- **TF-IDF:** Compute term frequency–inverse document frequency vectors  
- **Embeddings:** Pretrained Word2Vec/GloVe or custom Word2Vec  

Additional features:  
- Review word count  
- Review character count  
- Average word length  

## Model Implementation

| Model                  | Key Hyperparameters                        |
|------------------------|---------------------------------------------|
| Logistic Regression    | C (inverse regularization strength)         |
| Multinomial Naive Bayes| alpha (Laplace smoothing)                  |
| Linear SVM             | C (regularization)                          |
| Random Forest          | n_estimators, max_depth                    |
| LSTM (Keras)           | embedding_dim, units, dropout_rate, epochs |

All models are implemented in **src/models.py**, with training and evaluation orchestrated from **notebooks/Nlp-Project-1-PranavJoshi.ipynb**.

## Results

| Metric     | Logistic Regression | Naive Bayes | SVM   | Random Forest | LSTM  |
|------------|---------------------|-------------|-------|---------------|-------|
| Accuracy   | 0.88                | 0.85        | 0.87  | 0.86          | 0.89  |
| Precision  | 0.89                | 0.86        | 0.88  | 0.87          | 0.90  |
| Recall     | 0.87                | 0.84        | 0.86  | 0.85          | 0.88  |
| F1-Score   | 0.88                | 0.85        | 0.87  | 0.86          | 0.89  |

  
    
  Confusion matrix for the best-performing LSTM model.  


## Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**  
   Open `notebooks/Nlp-Project-1-PranavJoshi.ipynb` to explore the pipeline.  

## Project Structure

```
imdb-sentiment-analysis/
├── data/
│   └── imdb_reviews.csv
├── notebooks/
│   └── Nlp-Project-1-PranavJoshi.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   │   ├── wordcloud_positive.png
│   │   ├── wordcloud_negative.png
│   │   └── confusion_matrix_lstm.png
│   ├── models/
│   │   ├── lstm_model.h5
│   │   └── logistic_regression.pkl
│   └── metrics.csv
├── requirements.txt
└── README.md
```

## Insights & Observations

- **Class Balance:** Perfectly balanced dataset mitigates bias.  
- **Review Length:** Positive reviews average 140 words; negative 90 words.  
- **Word Importance:** “excellent”, “perfect” signal positive; “boring”, “waste” signal negative.  
- **Model Comparison:** LSTM outperforms classical models by ~2% F1-score.
