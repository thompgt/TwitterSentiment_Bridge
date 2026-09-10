"""Hugging Face Transformer sentiment analysis integration for TwitterSentiment-Bridge.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest for 3-class sentiment prediction
(negative, neutral, positive) with calibrated probability distributions.
"""

import time
from typing import Dict, List, Union
from transformers import pipeline


class TwitterSentimentTransformer:
    """Wrapper around Hugging Face CardiffNLP Twitter-RoBERTa sentiment model."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1,
    ):
        """Initialize the Hugging Face sentiment pipeline.

        Args:
            model_name: Hugging Face model repository identifier.
            device: Computing device (-1 for CPU, >=0 for CUDA GPU ID).
        """
        self.model_name = model_name
        self.device = device
        # top_k=None ensures all class scores are returned for probability calibration
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            top_k=None,
        )

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """Predict sentiment labels and calibrated probability distributions.

        Args:
            texts: Single string or list of preprocessed text strings.

        Returns:
            List of dicts containing 'label', 'score', 'probabilities', and 'latency_ms'.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        t0 = time.perf_counter()
        raw_results = self.sentiment_pipe(texts)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = round(total_latency_ms / max(len(texts), 1), 2)

        formatted_results = []
        for res in raw_results:
            # res is a list of dicts: [{'label': 'positive', 'score': 0.98}, ...]
            probs = {item["label"].lower(): round(float(item["score"]), 4) for item in res}
            # Ensure standard keys exist
            for expected in ("negative", "neutral", "positive"):
                probs.setdefault(expected, 0.0)

            best_label = max(probs, key=probs.get)
            best_score = probs[best_label]

            formatted_results.append(
                {
                    "label": best_label,
                    "score": best_score,
                    "probabilities": probs,
                    "latency_ms": per_item_latency_ms,
                }
            )

        return formatted_results


if __name__ == "__main__":
    ts = TwitterSentimentTransformer()
    test_tweets = [
        "I love this project! It makes hybrid sentiment analysis so clean. 😍",
        "The server crashed again. This is totally unacceptable.",
        "Flight arrives at 4:30 PM.",
    ]
    predictions = ts.predict(test_tweets)
    for t, p in zip(test_tweets, predictions):
        print(f"Tweet: {t}")
        print(f"  -> Label: {p['label']} (Score: {p['score']}) | Latency: {p['latency_ms']}ms")
        print(f"  -> Probs: {p['probabilities']}\n")
