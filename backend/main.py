"""
MailShield - FastAPI Backend

Provides the REST API for spam classification using a trained
Multinomial Naive Bayes model with TF-IDF features.

Endpoints:
    GET  /          - Welcome message
    GET  /health    - Health check
    GET  /model-info - Model information and metrics
    POST /predict   - Classify a message as spam or not spam
"""

import os
import json
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from ml.preprocess import preprocess_text

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mailshield")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_classifier.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

# Global model references
ml_pipeline = None
model_metadata = None

# Maximum message length (characters)
MAX_MESSAGE_LENGTH = 10000


def load_model():
    """Load the trained ML pipeline and metadata."""
    global ml_pipeline, model_metadata
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Run 'python ml/train.py' to train the model first.")
        return False
    
    try:
        ml_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                model_metadata = json.load(f)
            logger.info(f"Metadata loaded from {METADATA_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            model_metadata = None
    else:
        logger.warning(f"Metadata file not found: {METADATA_PATH}")
        model_metadata = None
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    success = load_model()
    if not success:
        logger.warning("Model not loaded. /predict endpoint will be unavailable.")
    yield


# FastAPI app
app = FastAPI(
    title="MailShield API",
    description="AI-Powered Spam Message Classifier using Multinomial Naive Bayes",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://mailshield-1ctm.onrender.com/"
    ).split(",")
    if origin.strip()
]

logger.info(f"CORS_ORIGINS environment variable: {os.environ.get('CORS_ORIGINS')}")
logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The message text to classify",
    )
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only")
        return stripped


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    spam_probability: float
    ham_probability: float
    message_length: int
    top_contributing_words: list = []


class ModelInfoResponse(BaseModel):
    algorithm: str
    feature_extraction: str
    dataset: str
    total_samples: int
    training_samples: int
    testing_samples: int
    training_spam: int
    training_ham: int
    testing_spam: int
    testing_ham: int
    classes: list
    vocabulary_size: int
    metrics: dict
    top_spam_features: list = []
    top_ham_features: list = []


# Endpoints
@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "application": "MailShield",
        "description": "AI-Powered Spam Message Classifier",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": ml_pipeline is not None,
        "metadata_loaded": model_metadata is not None,
    }


@app.get("/model-info")
async def get_model_info():
    """Return model information and performance metrics."""
    if model_metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model metadata is not available. Train the model first.",
        )
    
    return model_metadata


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify a message as spam or not spam."""
    if ml_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train and load the model first.",
        )
    
    try:
        # Preprocess the message
        clean_message = preprocess_text(request.message)
        
        if not clean_message:
            raise HTTPException(
                status_code=400,
                detail="Message is empty after preprocessing.",
            )
        
        # Get prediction and probabilities
        prediction = ml_pipeline.predict([clean_message])[0]
        probabilities = ml_pipeline.predict_proba([clean_message])[0]
        
        # Class labels: 0 = ham, 1 = spam
        ham_prob = float(probabilities[0])
        spam_prob = float(probabilities[1])
        confidence = float(probabilities[prediction])
        label = "SPAM" if prediction == 1 else "NOT SPAM"
        
        # Get top contributing words for this specific message
        contributing_words = get_contributing_words(clean_message, prediction)
        
        return PredictResponse(
            prediction=label,
            confidence=round(confidence, 4),
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            message_length=len(request.message),
            top_contributing_words=contributing_words,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again.",
        )


def get_contributing_words(message: str, prediction: int, top_n: int = 5) -> list:
    """Identify words that contributed most to the prediction.
    
    Uses TF-IDF weights and Naive Bayes log probabilities to find
    which words in the input message had the strongest influence.
    """
    try:
        tfidf = ml_pipeline.named_steps["tfidf"]
        classifier = ml_pipeline.named_steps["classifier"]
        
        # Transform the message to TF-IDF
        tfidf_vector = tfidf.transform([message])
        feature_names = tfidf.get_feature_names_out()
        
        # Get non-zero features for this message
        non_zero_indices = tfidf_vector.nonzero()[1]
        
        if len(non_zero_indices) == 0:
            return []
        
        # Get log probabilities for the predicted class
        log_probs = classifier.feature_log_prob_[prediction]
        other_log_probs = classifier.feature_log_prob_[1 - prediction]
        
        # Calculate contribution: TF-IDF weight * (log_prob_predicted - log_prob_other)
        contributions = []
        for idx in non_zero_indices:
            tfidf_weight = float(tfidf_vector[0, idx])
            log_prob_diff = log_probs[idx] - other_log_probs[idx]
            contribution = tfidf_weight * log_prob_diff
            if contribution > 0:  # Only positive contributions
                contributions.append({
                    "word": feature_names[idx],
                    "contribution": round(float(contribution), 4),
                })
        
        # Sort by contribution and return top N
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:top_n]
    
    except Exception as e:
        logger.warning(f"Could not extract contributing words: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
