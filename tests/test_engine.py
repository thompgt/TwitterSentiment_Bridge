"""Unit tests for TwitterSentiment-Bridge engine and preprocessing."""

import os
import sys
import pytest

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import clean_for_baseline, clean_for_transformer, clean_tweet, expand_contractions
from main import HybridSentimentPredictor


class TestPreprocessing:
    """Test suite for dual-track tweet cleaners."""

    def test_clean_for_transformer_preserves_emojis_and_handles(self):
        raw = "@jack Check this out!! 😍🔥 https://t.co/12345"
        cleaned = clean_for_transformer(raw)
        assert "@user" in cleaned
        assert "😍🔥" in cleaned
        assert "http" not in cleaned
        assert "!!" in cleaned

    def test_clean_for_transformer_elongation_compression(self):
        raw = "This is soooooooo coooool"
        cleaned = clean_for_transformer(raw)
        assert cleaned == "This is soo cool"

    def test_clean_for_baseline_strips_punctuation_and_lowercases(self):
        raw = "@jack Can't believe this didn't work! #fail https://t.co/xyz"
        cleaned = clean_for_baseline(raw)
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "cannot" in cleaned
        assert "not" in cleaned
        assert "fail" in cleaned
        assert "http" not in cleaned
        assert cleaned == cleaned.lower()

    def test_expand_contractions(self):
        assert "cannot" in expand_contractions("can't")
        assert "will not" in expand_contractions("won't")
        assert "are" in expand_contractions("they're")

    def test_empty_string_handling(self):
        assert clean_for_transformer("") == ""
        assert clean_for_baseline(None) == ""
        assert clean_tweet("") == ""


class TestHybridPredictor:
    """Test suite for HybridSentimentPredictor."""

    @pytest.fixture(scope="module")
    def predictor(self):
        return HybridSentimentPredictor(confidence_threshold=0.82)

    def test_fast_mode_prediction(self, predictor):
        res = predictor.predict("I love this software so much!", mode="fast")
        assert res["label"] == "positive"
        assert res["score"] > 0.5
        assert "negative" in res["probabilities"]
        assert "positive" in res["probabilities"]
        assert "baseline_lr" in res["model_used"]

    def test_hybrid_mode_confident_tweet(self, predictor):
        res = predictor.predict("Horrible service, worst experience ever.", mode="hybrid")
        assert res["label"] == "negative"
        assert res["score"] >= 0.7
        assert res["mode"] == "hybrid"

    def test_probabilities_sum_to_one(self, predictor):
        res = predictor.predict("Decent product, does the job.", mode="fast")
        total_p = sum(res["probabilities"].values())
        assert abs(total_p - 1.0) < 0.05

    def test_batch_prediction_shape(self, predictor):
        tweets = [
            "Super excited for tomorrow!",
            "Completely broken update.",
        ]
        results = predictor.predict_batch(tweets, mode="fast")
        assert len(results) == 2
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"
