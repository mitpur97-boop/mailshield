"""
MailShield - Model Training Script

Trains a Multinomial Naive Bayes spam classifier using TF-IDF features.
Dataset: UCI SMS Spam Collection
Algorithm: Multinomial Naive Bayes with TF-IDF Vectorization

Usage:
    python train.py

The script will:
1. Download or load the UCI SMS Spam Collection dataset
2. Preprocess the text data
3. Split into train/test sets
4. Build a TF-IDF + MultinomialNB pipeline
5. Train the model
6. Evaluate on the test set
7. Save the trained pipeline and metadata
"""

import os
import sys
import json
import zipfile
import io
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_text

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def download_dataset():
    """Download the UCI SMS Spam Collection dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset_file = os.path.join(DATA_DIR, "SMSSpamCollection")
    
    if os.path.exists(dataset_file):
        print(f"Dataset already exists at {dataset_file}")
        return dataset_file
    
    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract the SMSSpamCollection file
            for name in z.namelist():
                if "SMSSpamCollection" in name and not name.endswith("/"):
                    with z.open(name) as source:
                        content = source.read()
                    with open(dataset_file, "wb") as target:
                        target.write(content)
                    print(f"Dataset saved to {dataset_file}")
                    return dataset_file
        
        raise FileNotFoundError("SMSSpamCollection not found in zip archive")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print(f"\nPlease manually download the dataset from:")
        print(f"  {DATASET_URL}")
        print(f"Extract 'SMSSpamCollection' to: {DATA_DIR}")
        sys.exit(1)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate the SMS Spam Collection dataset."""
    print("\nLoading dataset...")
    
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    
    print(f"  Total records loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Drop missing values
    initial_count = len(df)
    df = df.dropna(subset=["label", "message"])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} records with missing values")
    
    # Drop duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["message"])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} duplicate messages")
    
    # Validate labels
    valid_labels = {"ham", "spam"}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.any():
        print(f"  Dropping {invalid_mask.sum()} records with invalid labels")
        df = df[~invalid_mask]
    
    # Convert labels: spam=1, ham=0
    df["label_encoded"] = (df["label"] == "spam").astype(int)
    
    print(f"\n  Dataset summary:")
    print(f"    Total messages: {len(df)}")
    print(f"    Ham (legitimate): {(df['label_encoded'] == 0).sum()}")
    print(f"    Spam: {(df['label_encoded'] == 1).sum()}")
    print(f"    Spam ratio: {df['label_encoded'].mean():.2%}")
    
    return df


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + Multinomial Naive Bayes pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=0.1)),
    ])
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
    }
    
    return metrics, y_pred


def get_top_features(pipeline, n=20):
    """Extract top features (words) for each class from the trained pipeline."""
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = tfidf.get_feature_names_out()
    
    # Log probabilities for each class
    # Class 0 = ham, Class 1 = spam
    log_probs = classifier.feature_log_prob_
    
    # Top spam-indicative features
    spam_log_probs = log_probs[1]  # class 1 = spam
    ham_log_probs = log_probs[0]   # class 0 = ham
    
    # Difference in log probabilities (spam vs ham)
    diff = spam_log_probs - ham_log_probs
    
    top_spam_indices = np.argsort(diff)[-n:][::-1]
    top_ham_indices = np.argsort(diff)[:n]
    
    top_spam_features = [
        {"word": feature_names[i], "score": round(float(diff[i]), 4)}
        for i in top_spam_indices
    ]
    top_ham_features = [
        {"word": feature_names[i], "score": round(float(diff[i]), 4)}
        for i in top_ham_indices
    ]
    
    return top_spam_features, top_ham_features


def main():
    """Main training workflow."""
    print("=" * 60)
    print("MailShield - Spam Classifier Training")
    print("=" * 60)
    
    # Step 1: Download/load dataset
    dataset_path = download_dataset()
    df = load_dataset(dataset_path)
    
    # Step 2: Preprocess text
    print("\nPreprocessing text...")
    df["message_clean"] = df["message"].apply(preprocess_text)
    
    # Remove any empty messages after preprocessing
    df = df[df["message_clean"].str.len() > 0]
    print(f"  Messages after preprocessing: {len(df)}")
    
    # Step 3: Split dataset
    X = df["message_clean"]
    y = df["label_encoded"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Training spam: {y_train.sum()}")
    print(f"  Training ham: {(y_train == 0).sum()}")
    print(f"  Testing spam: {y_test.sum()}")
    print(f"  Testing ham: {(y_test == 0).sum()}")
    
    # Step 4: Build and train pipeline
    print("\nBuilding pipeline...")
    pipeline = build_pipeline()
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("  Training complete!")
    
    # Step 5: Evaluate
    print("\nEvaluating model...")
    metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
    
    print(f"\n  Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1_score']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm['true_negative']}  FP={cm['false_positive']}")
    print(f"    FN={cm['false_negative']}  TP={cm['true_positive']}")
    
    # Step 6: Extract top features
    print("\nExtracting top features...")
    top_spam, top_ham = get_top_features(pipeline)
    print(f"  Top spam words: {[f['word'] for f in top_spam[:10]]}")
    print(f"  Top ham words: {[f['word'] for f in top_ham[:10]]}")
    
    # Step 7: Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("  Model saved!")
    
    # Step 8: Save metadata
    tfidf = pipeline.named_steps["tfidf"]
    metadata = {
        "algorithm": "Multinomial Naive Bayes",
        "feature_extraction": "TF-IDF (Term Frequency-Inverse Document Frequency)",
        "dataset": "UCI SMS Spam Collection",
        "total_samples": len(df),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "training_spam": int(y_train.sum()),
        "training_ham": int((y_train == 0).sum()),
        "testing_spam": int(y_test.sum()),
        "testing_ham": int((y_test == 0).sum()),
        "classes": ["ham (Not Spam)", "spam (Spam)"],
        "tfidf_params": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "sublinear_tf": True,
        },
        "nb_params": {
            "alpha": 0.1,
        },
        "vocabulary_size": len(tfidf.vocabulary_),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "top_spam_features": top_spam,
        "top_ham_features": top_ham,
    }
    
    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Metadata saved!")
    
    # Print classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    
    print("\n" + "=" * 60)
    print("Training complete! Model is ready for deployment.")
    print("=" * 60)
    
    return pipeline, metadata


if __name__ == "__main__":
    main()
