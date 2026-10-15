"""
MailShield - Model Evaluation Script

Loads the trained model and evaluates it, or re-evaluates on the test set.

Usage:
    python evaluate.py
"""

import os
import sys
import json
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def load_model_and_metadata():
    """Load the trained model and its metadata."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run train.py first to train the model.")
        sys.exit(1)
    
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        print("Run train.py first to generate metadata.")
        sys.exit(1)
    
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    return pipeline, metadata


def display_evaluation():
    """Display the model evaluation results."""
    pipeline, metadata = load_model_and_metadata()
    
    print("=" * 60)
    print("MailShield - Model Evaluation Report")
    print("=" * 60)
    
    print(f"\nAlgorithm: {metadata['algorithm']}")
    print(f"Feature Extraction: {metadata['feature_extraction']}")
    print(f"Dataset: {metadata['dataset']}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Training samples: {metadata['training_samples']}")
    print(f"  Testing samples: {metadata['testing_samples']}")
    print(f"  Vocabulary size: {metadata['vocabulary_size']}")
    
    metrics = metadata['metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Ham  Predicted Spam")
    print(f"  Actual Ham       {cm['true_negative']:>6}         {cm['false_positive']:>6}")
    print(f"  Actual Spam      {cm['false_negative']:>6}         {cm['true_positive']:>6}")
    
    print(f"\nTop Spam-Indicative Words:")
    for i, feat in enumerate(metadata.get('top_spam_features', [])[:10], 1):
        print(f"  {i:2d}. {feat['word']:<20} (score: {feat['score']:+.4f})")
    
    print(f"\nTop Ham-Indicative Words:")
    for i, feat in enumerate(metadata.get('top_ham_features', [])[:10], 1):
        print(f"  {i:2d}. {feat['word']:<20} (score: {feat['score']:+.4f})")
    
    # Test a few sample predictions
    print(f"\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    
    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "CONGRATULATIONS! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Reminder: Your dentist appointment is scheduled for 3pm today.",
        "FREE entry in a weekly competition to win an iPad! Text WIN to 80085",
        "Can you pick up some milk on your way home?",
    ]
    
    for msg in test_messages:
        prediction = pipeline.predict([msg.lower()])[0]
        proba = pipeline.predict_proba([msg.lower()])[0]
        label = "SPAM" if prediction == 1 else "NOT SPAM"
        confidence = proba[prediction]
        print(f"\n  Message: \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print(f"  Prediction: {label} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    display_evaluation()
