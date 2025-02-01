import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # IMDb Sentiment Analysis

        In this notebook, we:  
        1. **Download & Extract** the IMDb dataset.  
        2. **Load** the dataset into pandas DataFrames.   
        3. Perform **Exploratory Data Analysis (EDA)** to understand the data.  
        4. **Build & Train** a sentiment analysis model using a scikit-learn pipeline.  
        5. **Evaluate** the model on the test data.
        6. Attempt to improve the model using hyperparameter tuning.  
        7. **Explain** model predictions using LIME.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Download and extract IMDb movie reviews dataset""")
    return


@app.cell
def _():
    import os
    from pathlib import Path
    import tarfile
    import requests
    import pandas as pd
    from tqdm import tqdm

    def download(url, dest_path):
        """Download a file from URL and provide a progress bar."""

        response = requests.get(url, stream=True) # send a get request, stream response in chunks
        response.raise_for_status()  # Ensure we catch HTTP errors early
        total = int(response.headers.get('content-length', 0)) # get total content length, default 0
        with open(dest_path, 'wb') as file, tqdm( 
            desc=dest_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                if data:  # filter out keep-alive chunks
                    file.write(data)
                    bar.update(len(data))

    def extract_tarfile(tar_path, extract_path):
        """Extract a tar.gz file to the specified path."""
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extracted to {extract_path}")


    def load_imdb_data(data_dir, subset="train"):
        """
        Load IMDb data from the given directory (e.g., aclImdb/train).
        Returns a DataFrame with columns 'text' and 'sentiment'.
        """
        data_dir = Path(data_dir) / subset
        reviews, sentiments = [], []

        for sentiment, label in zip(['pos', 'neg'], [1, 0]):
            folder = data_dir / sentiment
            if not folder.exists():
                print(f"Warning: {folder} does not exist. Skipping.")
                continue
            # Use glob to find .txt files in the folder
            for file_path in folder.glob("*.txt"):
                try:
                    review = file_path.read_text(encoding="utf-8")
                    reviews.append(review)
                    sentiments.append(label)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

        return pd.DataFrame({"text": reviews, "sentiment": sentiments})
    return (
        Path,
        download,
        extract_tarfile,
        load_imdb_data,
        os,
        pd,
        requests,
        tarfile,
        tqdm,
    )


@app.cell
def _(Path, download, extract_tarfile):
    # Define dataset parameters
    dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = Path("aclImdb_v1.tar.gz")
    extract_path = Path(".")  # current directory
    data_dir = Path("aclImdb")

    # Download and extract dataset if necessary
    if not data_dir.exists():
        if not tar_path.exists():
            print("Dataset not found locally. Downloading...")
            download(dataset_url, tar_path)
        print("Extracting dataset...")
        extract_tarfile(tar_path, extract_path)
    else:
        print("Dataset already exists. Skipping download/extraction.")
    return data_dir, dataset_url, extract_path, tar_path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 2. Load the dataset into pandas dataframes""")
    return


@app.cell
def _(data_dir, load_imdb_data):
    print("Loading training data...")
    train_df = load_imdb_data(data_dir, subset="train")
    print(f"Loaded {len(train_df)} training examples.")

    print("Loading test data...")
    test_df = load_imdb_data(data_dir, subset="test")
    print(f"Loaded {len(test_df)} test examples.")
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Exploratory Data Analysis (EDA)


        In this section, we explore the training data by:  
        - Displaying a sample of reviews.  
        - Examining the distribution of sentiment labels.  
        - Analyzing the distribution of review lengths.
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _(train_df):
    # Display a sample of the training data
    print("Sample of training data:")
    train_df.sample(100)
    return


@app.cell
def _(train_df):
    # Check the distribution of sentiment labels
    print("Sentiment Distribution:")
    print(train_df['sentiment'].value_counts())
    return


@app.cell
def _(plt, sns, train_df):
    # Analyze review lengths (in words)
    train_df['review_length'] = train_df['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df['review_length'], bins=50, kde=True)
    plt.title("Distribution of Review Lengths (in Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.show()
    return


@app.cell
def _(train_df):
    print(train_df['review_length'].describe())
    return


@app.cell
def _(plt, sns, train_df):
    # Boxplot for review lengths by sentiment
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='sentiment', y='review_length', data=train_df)
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.title("Review Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Word Count")
    plt.show()
    return


@app.cell
def _(train_df):
    # Show a sample negative and positive review
    print("Sample Negative Review:")
    print(train_df[train_df['sentiment'] == 0]['text'].iloc[0])
    print()
    print("Sample Positive Review:")
    print(train_df[train_df['sentiment'] == 1]['text'].iloc[0])
    return


@app.cell
def _(test_df, train_df):
    # IMDb reviews may contain html tags:

    import re
    from collections import Counter

    # Regular expression pattern to detect HTML tags
    html_tags = [
        "html", "head", "body", "title", "div", "span", "p", "a", "ul", "li",
        "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "br",
        "hr", "img", "script", "style", "link", "meta"
    ]

    # Build the regex pattern.
    # The pattern explained:
    #   - </?       : Matches either an opening (<) or closing (</) tag.
    #   - (?:...)   : A non-capturing group containing all tag names separated by '|'.
    #   - (?:\s+[^>]*?)? : Optionally matches whitespace and any attribute content (non-greedy).
    #   - /?>       : Matches an optional self-closing slash followed by the closing >.
    pattern_string = r"</?(?:{})(?:\s+[^>]*?)?/?>".format("|".join(html_tags))
    html_tag_pattern = re.compile(pattern_string, re.IGNORECASE)

    # Function to check if text contains HTML tags
    def contains_html(text):
        return bool(html_tag_pattern.search(text))

    # Count the number of reviews with HTML tags
    train_df['has_html'] = train_df['text'].apply(contains_html)
    test_df['has_html'] = test_df['text'].apply(contains_html)
    train_html_count = train_df['has_html'].sum()
    test_html_count = test_df['has_html'].sum()
    print(f"Reviews containing HTML tags (Train): {train_html_count}")
    print(f"Reviews containing HTML tags (Test): {test_html_count}")
    return (
        Counter,
        contains_html,
        html_tag_pattern,
        html_tags,
        pattern_string,
        re,
        test_html_count,
        train_html_count,
    )


@app.cell
def _(Counter, html_tag_pattern, re, train_df):
    # Extract all HTML tags and count occurrences
    def extract_html_tags(text):
        return re.findall(html_tag_pattern, text)

    all_html_tags = []
    for review in train_df['text']:
        all_html_tags.extend(extract_html_tags(review))

    # Count frequency of unique tags
    html_tag_counts = Counter(all_html_tags).most_common(10)

    print("Most common HTML tags in dataset:")
    for tag, count in html_tag_counts:
        print(f"{tag}: {count}")
    return (
        all_html_tags,
        count,
        extract_html_tags,
        html_tag_counts,
        review,
        tag,
    )


@app.cell
def _(test_df, train_df):
    import emoji

    # Function to detect if a text contains emojis
    def contains_emoji(text):
        return bool(emoji.emoji_count(text))

    # Count the number of reviews containing emojis
    train_df['has_emoji'] = train_df['text'].apply(contains_emoji)
    test_df['has_emoji'] = test_df['text'].apply(contains_emoji)

    train_emoji_count = train_df['has_emoji'].sum()
    test_emoji_count = test_df['has_emoji'].sum()

    print(f"Reviews containing emojis (Train): {train_emoji_count}")
    print(f"Reviews containing emojis (Test): {test_emoji_count}")
    return contains_emoji, emoji, test_emoji_count, train_emoji_count


@app.cell
def _(Counter, emoji, train_df):
    # Extract all emojis and count occurrences
    def extract_emojis(text):
        return [char for char in text if char in emoji.EMOJI_DATA]

    all_emojis = []
    for review_ in train_df['text']:
        all_emojis.extend(extract_emojis(review_))

    # Count frequency of unique emojis
    emoji_counts = Counter(all_emojis).most_common(10)

    print("Most common emojis in dataset:")
    for em, count_ in emoji_counts:
        print(f"{em}: {count_}")
    return all_emojis, count_, em, emoji_counts, extract_emojis, review_


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Build and Train a Sentiment Analysis Model

        We create a scikit-learn pipeline that includes:  
        - **HMTL Cleaner** to remove html tags. 
        - **TF-IDF Vectorization** to transform text data.  
        - **Logistic Regression** for classification.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What is TF-IDF?**

        TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It’s a statistical measure used to evaluate how important a word is to a document in a collection (or corpus). Here’s why and how it works:

        1.	**Term Frequency (TF)**: This measures **how often a term appears in a document**. The more a term appears, the more important it might seem in that specific document.  
        2.	**Inverse Document Frequency (IDF)**: This measures **how common or rare a word is across all documents**. Words that appear in many documents (like “the”, “is”, etc.) are less informative and are downweighted, while words that are rare across the corpus but frequent in a specific document are given more importance.  
        3.	**Combined Effect**: By multiplying TF and IDF, TF-IDF assigns a high weight to words that are frequent in a specific document but not frequent in the entire corpus. This helps in emphasizing terms that are more discriminative for that document. In sentiment analysis, such words can be strong indicators of positive or negative sentiment.

        \[
        \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
        \]

        \[
        \text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
        \]

        \[
        \text{IDF}(t) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right)
        \]
        """
    )
    return


@app.cell
def _():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    return (
        LogisticRegression,
        Pipeline,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        confusion_matrix,
    )


@app.cell(hide_code=True)
def _():
    # Define an html cleaner transformer to use in the pipeline

    from bs4 import BeautifulSoup
    from sklearn.base import BaseEstimator, TransformerMixin

    class HTMLCleaner(BaseEstimator, TransformerMixin):
        """Custom transformer to remove HTML tags from text data."""

        def fit(self, X, y=None):
            return self  # No fitting necessary

        def transform(self, X):
            def clean(text):
                # Check if the text appears to contain HTML tags
                if '<' in text and '>' in text:
                    return BeautifulSoup(text, "html.parser").get_text()
                return text
            return X.apply(clean)
    return BaseEstimator, BeautifulSoup, HTMLCleaner, TransformerMixin


@app.cell
def _(HTMLCleaner, LogisticRegression, Pipeline, TfidfVectorizer):
    pipeline = Pipeline([
        ('html_cleaner', HTMLCleaner()),
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
    ])
    pipeline
    return (pipeline,)


@app.cell
def _(pipeline, train_df):
    # Train the model on the training data
    print("Training the sentiment analysis model...")
    pipeline.fit(train_df['text'], train_df['sentiment'])
    print("Model training complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Evaluate the trained model""")
    return


@app.cell(hide_code=True)
def _(accuracy_score, pipeline, test_df):
    # Generate predictions on the test set
    print("Evaluating the model on test data...")
    preds = pipeline.predict(test_df['text'])
    accuracy = accuracy_score(test_df['sentiment'], preds)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    return accuracy, preds


@app.cell(hide_code=True)
def _(classification_report, preds, test_df):
    print("Classification Report:")
    print(classification_report(test_df['sentiment'], preds))
    return


@app.cell(hide_code=True)
def _(confusion_matrix, plt, preds, sns, test_df):
    # Plot the confusion matrix
    cm = confusion_matrix(test_df['sentiment'], preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    return (cm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Tune hyperparameters 

        Our search strategy is built around **Bayesian optimization** using Optuna’s default sampler (which is based on the Tree-structured Parzen Estimator, or TPE). Here’s a detailed breakdown of what that means for our hyperparameter tuning:

        For each trial, Optuna uses its Bayesian optimization strategy to **intelligently sample a combination of these hyperparameters**. The default TPE sampler **builds a probabilistic model of the objective function** and uses it to **choose promising hyperparameter configurations**.

        Our goal is to **maximize the cross-validated accuracy**. Optuna iteratively updates its probabilistic model based on the results of the previous trials, guiding the search towards areas in the hyperparameter space that are likely to yield higher accuracy.

        **TFIDF Hyperparameters**  
        - **max_df:** controls how frequent a word must be before it is ignored
        - **min_df:** controls how rare a word must be before it is ignored
        - **ngram_range:** defines word sequences (n-grams) to consider (e.g. (1,2) = unigrams and bigrams)

        **Logistic Regression Hyperparameters**  
        - **C:** inverse regularization strength (high C = less regularization)
        """
    )
    return


@app.cell(hide_code=True)
def _(
    HTMLCleaner,
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    train_df,
):
    import optuna
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        # Suggest hyperparameters for the TF-IDF vectorizer:
        max_df = trial.suggest_float('tfidf_max_df', 0.5, 1.0)
        min_df = trial.suggest_int('tfidf_min_df', 1, 10)
        ngram_range = trial.suggest_categorical('tfidf_ngram_range', ['1-1', '1-2'])
        
        if ngram_range == '1-1':
            ngram_range = (1,1)
        elif ngram_range == '1-2':
            ngram_range = (1,2)
                
        # Suggest hyperparameters for Logistic Regression:
        C = trial.suggest_float('clf_C', 1e-3, 1e3, log=True)

        # Build the pipeline with the suggested hyperparameters:
        pipeline = Pipeline([
            ('html_cleaner', HTMLCleaner()),
            ('tfidf', TfidfVectorizer(
                stop_words='english', 
                max_df=max_df, 
                min_df=min_df, 
                ngram_range=ngram_range
            )),
            ('clf', LogisticRegression(C=C, solver='saga', max_iter=1000))
        ])

        # Evaluate the pipeline using cross-validation:
        scores = cross_val_score(pipeline, train_df['text'], train_df['sentiment'],
                                 cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    return cross_val_score, objective, optuna


@app.cell
def _(objective, optuna):
    # Create an Optuna study to maximize accuracy:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Accuracy: {:.4f}".format(best_trial.value))
    print("  Best hyperparameters:")
    for param, value in best_trial.params.items():
        print("    {}: {}".format(param, value))
    return best_trial, param, study, value


@app.cell
def _(
    HTMLCleaner,
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    study,
    train_df,
):
    # After tuning, suppose the best parameters were:
    best_params = study.best_trial.params
    if best_params['tfidf_ngram_range'] == '1-1':
        best_ngram_range = (1,1)
    elif best_params['tfidf_ngram_range'] == '1-2':
        best_ngram_range = (1,2)
    # Build the final pipeline using the best parameters:
    final_pipeline = Pipeline([
        ('html_cleaner', HTMLCleaner()),
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_df=best_params['tfidf_max_df'],
            min_df=best_params['tfidf_min_df'],
            ngram_range=best_ngram_range
        )),
        ('clf', LogisticRegression(solver='saga', C=best_params['clf_C'], max_iter=1000))
    ])
    # Train on the full training data:
    final_pipeline.fit(train_df['text'], train_df['sentiment'])
    return best_ngram_range, best_params, final_pipeline


@app.cell(hide_code=True)
def _(accuracy_score, final_pipeline, test_df):
    # Generate predictions on the test set
    print("Evaluating the model on test data...")
    tuned_preds = final_pipeline.predict(test_df['text'])
    tuned_accuracy = accuracy_score(test_df['sentiment'], tuned_preds)
    print(f"Test Accuracy for tuned model: {tuned_accuracy:.4f}\n")
    return tuned_accuracy, tuned_preds


@app.cell(hide_code=True)
def _(classification_report, test_df, tuned_preds):
    print("Classification Report:")
    print(classification_report(test_df['sentiment'], tuned_preds))
    return


@app.cell(hide_code=True)
def _(confusion_matrix, plt, sns, test_df, tuned_preds):
    # Plot the confusion matrix
    cm_tuned = confusion_matrix(test_df['sentiment'], tuned_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    return (cm_tuned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Explanability with LIME

        LIME (Local Interpretable Model-agnostic Explanations) is a technique designed to **help you understand the predictions** of any machine learning classifier by approximating it locally with an interpretable (usually linear) model. 



        """
    )
    return


@app.cell
def _(final_pipeline, pd, test_df):
    """
    Explain model predictions using LIME.

    We select a sample review from the test dataset, generate an explanation 
    for the predicted sentiment, and display the explanation as an interactive
    HTML visualization within the notebook.
    """

    import lime
    import lime.lime_text

    # Create a LimeTextExplainer object.
    # Provide the class names corresponding to the sentiment labels.
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])

    # Choose a sample review from the test set (for example, the first review).
    idx = 0
    sample_text = test_df['text'].iloc[idx]

    # Wrapper function to ensure input format is consistent
    def predict_proba_fixed(text_list):
        # Convert list to pandas Series if needed, then predict probabilities.
        return final_pipeline.predict_proba(pd.Series(text_list))

    # Generate an explanation for the sample review.
    explanation = explainer.explain_instance(
        sample_text,
        predict_proba_fixed,  # Use the wrapped function
        num_features=10,      # Show up to 10 features (words) in the explanation.
        labels=[0, 1]         # Explain both classes: Negative and Positive.
    )
    return (
        explainer,
        explanation,
        idx,
        lime,
        predict_proba_fixed,
        sample_text,
    )


@app.cell
def _(explanation):
    # Print the explanation for each class as a list of feature-weight pairs.
    print("Explanation for Negative sentiment (class 0):")
    for feature, weight in explanation.as_list(label=0):
        print(f"{feature}: {weight:.3f}")
    print("\nExplanation for Positive sentiment (class 1):")
    for feature, weight in explanation.as_list(label=1):
        print(f"{feature}: {weight:.3f}")
    return feature, weight


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Conclusion

        We have **successfully** built a **sentiment analysis model** using the IMDb dataset.  
        - The EDA phase provided insights into the dataset (such as sentiment distribution and review length characteristics).  
        - Our pipeline (Cleaner + TF-IDF + Logistic Regression) **achieved a reasonable accuracy** on the test set.  
        - Hyperparameter tuning brought **marginal improvements** to performance.  
        - LIME explanations provide insight into **how the model interprets** a single example.   
        - Further improvements might include experimenting with **different classification algorithms.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
