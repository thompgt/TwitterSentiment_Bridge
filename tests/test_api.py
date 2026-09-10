"""Integration tests for TwitterSentiment-Bridge FastAPI application."""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root and src are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a test client instance for FastAPI."""
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    """Test suite for REST endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "baseline_loaded" in data

    def test_metadata_endpoint(self, client):
        response = client.get("/api/v1/meta")
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "models" in data
        assert "baseline" in data["models"]
        assert "transformer" in data["models"]

    def test_predict_single_positive(self, client):
        payload = {
            "text": "The new update solved everything! Love the design and speed.",
            "mode": "fast",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "positive"
        assert data["score"] > 0.5
        assert "probabilities" in data
        assert data["probabilities"]["positive"] > data["probabilities"]["negative"]

    def test_predict_single_negative(self, client):
        payload = {
            "text": "Worst purchase of my entire life, broken right out of the box.",
            "mode": "fast",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "negative"
        assert data["score"] > 0.5

    def test_predict_batch(self, client):
        payload = {
            "texts": [
                "Amazing customer service! So fast and polite.",
                "Terrible delay and rude staff.",
            ],
            "mode": "fast",
        }
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_processed"] == 2
        assert len(data["results"]) == 2

    def test_stream_sample(self, client):
        response = client.get("/api/v1/stream/sample?topic=tech&count=3&mode=fast")
        assert response.status_code == 200
        data = response.json()
        assert data["topic"] == "tech"
        assert data["count"] == 3
        assert len(data["items"]) == 3
        for item in data["items"]:
            assert "tweet" in item
            assert "analysis" in item

    def test_predict_invalid_mode(self, client):
        payload = {
            "text": "Hello world",
            "mode": "invalid_mode",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error
