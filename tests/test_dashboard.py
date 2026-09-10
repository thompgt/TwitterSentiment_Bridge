"""Smoke test to verify dashboard module components load cleanly."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


def test_dashboard_imports():
    from app.dashboard import load_predictor
    predictor = load_predictor(0.82)
    assert predictor is not None
    assert predictor.confidence_threshold == 0.82
