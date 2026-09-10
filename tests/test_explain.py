"""Unit tests for explainability and token attribution."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from explain import SentimentExplainer


class TestExplainability:
    """Test suite for SentimentExplainer."""

    @pytest.fixture(scope="module")
    def explainer(self):
        return SentimentExplainer()

    def test_explain_tokens_finds_positive_and_negative_drivers(self, explainer):
        text = "This application is wonderful, but the slow performance is terrible."
        res = explainer.explain_tokens(text)

        assert "top_positive_drivers" in res
        assert "top_negative_drivers" in res

        pos_terms = [t["term"] for t in res["top_positive_drivers"]]
        neg_terms = [t["term"] for t in res["top_negative_drivers"]]

        assert any("wonderful" in term for term in pos_terms)
        assert any("terrible" in term or "slow" in term for term in neg_terms)

    def test_explain_empty_string(self, explainer):
        res = explainer.explain_tokens("")
        assert res["cleaned_tokens"] == []
        assert res["top_positive_drivers"] == []
        assert res["top_negative_drivers"] == []
