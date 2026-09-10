"""Explainability, token saliency attribution, and model divergence diagnostics for TwitterSentiment-Bridge.

Enables developers and analysts to:
1. Identify which words drove a sentiment prediction (token attribution).
2. Isolate and audit tweets where the Baseline and Transformer disagree (e.g., sarcasm, idioms, slang).
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import HybridSentimentPredictor
from preprocess import clean_for_baseline


class SentimentExplainer:
    """Provides word-level saliency attribution and model disagreement analysis."""

    def __init__(self, predictor: Optional[HybridSentimentPredictor] = None):
        self.predictor = predictor or HybridSentimentPredictor()

    def explain_tokens(self, text: str, top_n: int = 5) -> Dict[str, Any]:
        """Compute token-level saliency weights using the baseline model coefficients.

        Returns words pushing towards positive vs negative sentiment with their relative weights.
        """
        if not self.predictor.baseline:
            return {"error": "Baseline model required for linear token attribution."}

        pipeline = self.predictor.baseline
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]

        cleaned = clean_for_baseline(text)
        tokens = cleaned.split()
        if not tokens:
            return {
                "text": text,
                "cleaned_tokens": [],
                "top_positive_drivers": [],
                "top_negative_drivers": [],
                "all_active_terms": [],
            }

        feature_names = vectorizer.get_feature_names_out()
        vocab = vectorizer.vocabulary_
        coefficients = classifier.coef_[0]  # Positive class weights

        token_contributions: List[Dict[str, Any]] = []

        # Unigrams and bigrams
        ngrams = list(tokens)
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i+1]}")

        for term in ngrams:
            if term in vocab:
                idx = vocab[term]
                weight = float(coefficients[idx])
                token_contributions.append(
                    {
                        "term": term,
                        "weight": round(weight, 4),
                        "polarity": "positive" if weight > 0 else "negative",
                        "impact": abs(round(weight, 4)),
                    }
                )

        # Deduplicate and sort
        unique_contributions = {}
        for tc in token_contributions:
            term = tc["term"]
            if term not in unique_contributions or abs(tc["weight"]) > abs(unique_contributions[term]["weight"]):
                unique_contributions[term] = tc

        sorted_terms = sorted(unique_contributions.values(), key=lambda x: x["weight"], reverse=True)
        top_pos = [t for t in sorted_terms if t["weight"] > 0][:top_n]
        top_neg = sorted([t for t in sorted_terms if t["weight"] < 0], key=lambda x: x["weight"])[:top_n]

        return {
            "text": text,
            "cleaned_tokens": tokens,
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "all_active_terms": sorted_terms,
        }

    def find_disagreements(self, corpus: List[str]) -> List[Dict[str, Any]]:
        """Audit a corpus to surface tweets where Baseline and Transformer predict opposite polarities.

        Ideal for surfacing sarcasm, double negations, and dialectal slang.
        """
        disagreements = []
        for tweet in corpus:
            res_base = self.predictor.predict(tweet, mode="fast")
            res_tf = self.predictor.predict(tweet, mode="accurate")

            # Check if polarities conflict (pos vs neg)
            b_label = res_base["label"]
            t_label = res_tf["label"]

            if (b_label == "positive" and t_label == "negative") or (b_label == "negative" and t_label == "positive"):
                disagreements.append(
                    {
                        "text": tweet,
                        "baseline_prediction": {
                            "label": b_label,
                            "confidence": res_base["score"],
                        },
                        "transformer_prediction": {
                            "label": t_label,
                            "confidence": res_tf["score"],
                        },
                        "divergence_type": f"baseline_{b_label}_vs_transformer_{t_label}",
                    }
                )

        return disagreements


if __name__ == "__main__":
    explainer = SentimentExplainer()
    sample = "The screen is amazing but battery life is horrible and completely unacceptable."
    explanation = explainer.explain_tokens(sample)
    print("Explanation for:", sample)
    print("Top Positive Drivers:", explanation["top_positive_drivers"])
    print("Top Negative Drivers:", explanation["top_negative_drivers"])
