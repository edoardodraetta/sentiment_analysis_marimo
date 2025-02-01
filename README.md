# IMDb Sentiment Analysis

![Marimo](https://img.shields.io/badge/Marimo-Interactive%20Notebooks-blue)
![LIME](https://img.shields.io/badge/LIME-Explainability-green)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## Project Overview
This project builds a **Sentiment Analysis Model** using the **IMDb dataset**. The main goal is to classify movie reviews as **positive** or **negative**. We use **Natural Language Processing (NLP)** techniques along with **Machine Learning** models to analyze sentiment.

## What is Marimo?
Marimo is a Python-based interactive computing framework that allows users to create and share computational notebooks. Unlike traditional notebooks like Jupyter, Marimo emphasizes modularity, reactivity, and a seamless development experience.

## Project Workflow

1. **Download & Extract** the IMDb dataset.
2. **Load** the dataset into pandas DataFrames.
3. Perform **Exploratory Data Analysis (EDA)** to understand the data.
4. **Build & Train** a sentiment analysis model using a scikit-learn pipeline.
5. **Evaluate** the model on the test data.
6. **Improve the model** using hyperparameter tuning.
7. **Explain** model predictions using LIME.

## Dataset
- **Source:** [Stanford IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- **Structure:**
  - 50,000 movie reviews (25,000 for training, 25,000 for testing)
  - Balanced dataset (equal positive and negative reviews)
  - Reviews stored as raw text files

## Dependencies
To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```

## Implementation Details

### Data Preprocessing
1. **HTML Cleaning:** Remove unwanted HTML tags from the text using `BeautifulSoup`.
2. **Emoji Processing:** Convert emojis into textual representations using the `emoji` library.
3. **TF-IDF Vectorization:** Transform text into numerical features for machine learning.

### Model Pipeline
We implement a **scikit-learn pipeline** consisting of:
- **HTMLCleaner** (Custom Transformer) – Removes HTML tags.
- **EmojiConverter** (Custom Transformer) – Converts emojis to words.
- **TF-IDF Vectorizer** – Converts text into numerical features.
- **Logistic Regression** – Classifies reviews as positive or negative.

### Hyperparameter Tuning
We use **Optuna** to optimize parameters:
- `max_df`, `min_df`, `ngram_range` (TF-IDF Vectorizer)
- `C` (Regularization strength for Logistic Regression)

### Model Evaluation
- **Accuracy Score** – Overall model performance.
- **Classification Report** – Precision, Recall, F1-score.
- **Confusion Matrix** – Breakdown of correct and incorrect classifications.

### Explainability with LIME
- We use **LIME (Local Interpretable Model-agnostic Explanations)** to understand the model’s predictions.
- Generates **word importance scores** for individual reviews.

## Results
- **Baseline Model Accuracy:** ~88%
- **After Hyperparameter Tuning:** ~90%
- **LIME Interpretation:** Model correctly identifies sentiment-heavy words.

## Conclusion
- **Machine Learning-based sentiment analysis is effective** on movie reviews.
- **Hyperparameter tuning improves accuracy** but marginally.
- **LIME helps explain model decisions**, making the classifier more interpretable.
- **Future improvements:** Try deep learning (e.g., LSTMs, Transformers) for more advanced NLP modeling.

## Running the Project
To execute the project:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Marimo notebook:
   ```bash
   marimo run IMDB Sentiment Analysis.py
   ```

