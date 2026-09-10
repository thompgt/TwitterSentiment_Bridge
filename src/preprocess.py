"""Dual-track tweet preprocessing pipeline for TwitterSentiment-Bridge.

Provides specialized cleaning strategies:
- `clean_for_transformer`: Preserves emojis, casing emphasis, and maps handles
  to standard tokens expected by Twitter-RoBERTa models.
- `clean_for_baseline`: Optimized for n-gram TF-IDF vectorization and linear models.
"""

import re
from typing import Optional

# Contraction mapping for robust baseline tokenization
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bn't\b": " not",
    r"\b're\b": " are",
    r"\b's\b": " is",
    r"\b'd\b": " would",
    r"\b'll\b": " will",
    r"\b've\b": " have",
    r"\b'm\b": " am",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions to preserve negation signals."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_for_transformer(text: Optional[str]) -> str:
    """Clean tweet text while preserving signals critical to Transformer models.

    - Standardizes Twitter user mentions to '@user' (CardiffNLP standard).
    - Removes raw URLs while preserving contextual text.
    - Compresses character elongation (e.g., 'sooooo happyyyyy' -> 'soo happyy').
    - Preserves emojis, punctuation (?!), and casing emphasis.
    """
    if not text:
        return ""

    text = str(text)

    # Normalize HTML entities
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)

    # Replace user handles with '@user' token expected by CardiffNLP Twitter-RoBERTa
    text = re.sub(r"@\w+", "@user", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE)

    # Compress character repetitions (> 2 consecutive chars -> 2 chars, e.g., 'coooool' -> 'cool')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Clean redundant whitespace while maintaining token separation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_baseline(text: Optional[str]) -> str:
    """Clean tweet text optimized for TF-IDF Vectorization & Logistic Regression.

    - Removes URLs and usernames completely.
    - Expands contractions (can't -> cannot) to preserve negation signals.
    - Removes punctuation and symbols.
    - Converts to lower case.
    """
    if not text:
        return ""

    text = str(text)

    # Expand contractions before stripping punctuation
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE)

    # Remove user @ references and hashtag symbol #
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    # Remove punctuation and non-alphanumeric chars (excluding spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Compress character repetitions
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Lowercase & normalize whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tweet(text: Optional[str], mode: str = "baseline") -> str:
    """Unified entrypoint for tweet cleaning.

    Args:
        text: Raw input tweet string.
        mode: Cleaning strategy, either 'transformer' or 'baseline'.

    Returns:
        Cleaned text string.
    """
    if mode == "transformer":
        return clean_for_transformer(text)
    return clean_for_baseline(text)
