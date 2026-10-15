"""
MailShield - Text Preprocessing Module

Handles text preprocessing for the spam classification pipeline.
Design philosophy: Light preprocessing to preserve spam-indicative features.

Approach:
- Convert to lowercase for consistency
- Normalize whitespace (collapse multiple spaces/newlines)
- Preserve URLs, numbers, special characters as they can be spam indicators
- Do NOT aggressively strip or stem - let TF-IDF handle feature extraction
"""

import re


def preprocess_text(text: str) -> str:
    """Preprocess a single message for the spam classifier.
    
    Args:
        text: Raw message string
        
    Returns:
        Preprocessed message string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace: collapse multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
