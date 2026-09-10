"""FastAPI application for TwitterSentiment-Bridge.

Provides RESTful endpoints for single predictions, batch analysis,
streaming simulation, and model observability.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)
from batch_processor import BatchSentimentProcessor
from main import HybridSentimentPredictor
from stream_simulator import SocialStreamSimulator

# Global engine instances
predictor: Optional[HybridSentimentPredictor] = None
batch_processor: Optional[BatchSentimentProcessor] = None
stream_simulator: Optional[SocialStreamSimulator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models on startup."""
    global predictor, batch_processor, stream_simulator
    print("[FastAPI] Initializing Hybrid Sentiment Engine...")
    predictor = HybridSentimentPredictor(confidence_threshold=0.82)
    batch_processor = BatchSentimentProcessor(predictor=predictor)
    stream_simulator = SocialStreamSimulator(default_topic="tech")
    print("[FastAPI] Engine ready to serve requests.")
    yield
    print("[FastAPI] Shutting down.")


app = FastAPI(
    title="TwitterSentiment-Bridge API",
    description=(
        "Production-grade hybrid sentiment analysis API bridging sub-millisecond "
        "TF-IDF Logistic Regression baselines with CardiffNLP Twitter-RoBERTa transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web dashboards and local frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> HybridSentimentPredictor:
    """Helper to ensure predictor is initialized."""
    global predictor
    if predictor is None:
        predictor = HybridSentimentPredictor(confidence_threshold=0.82)
    return predictor


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Observability"])
def health_check():
    """Health check endpoint displaying engine and model status."""
    p = get_predictor()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "baseline_loaded": p.baseline is not None,
        "transformer_loaded": p._hf_model is not None,
        "default_threshold": p.confidence_threshold,
        "device": "cpu",
    }


@app.post("/api/v1/predict", response_model=SentimentResponse, tags=["Inference"])
def predict_sentiment(payload: SentimentRequest):
    """Analyze the sentiment of a single tweet string."""
    p = get_predictor()
    try:
        result = p.predict(
            text=payload.text,
            mode=payload.mode,
            confidence_threshold=payload.confidence_threshold,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictRequest):
    """Analyze a batch of tweets with statistical rollups."""
    global batch_processor
    if batch_processor is None:
        batch_processor = BatchSentimentProcessor(predictor=get_predictor())

    try:
        output = batch_processor.process_texts(
            texts=payload.texts,
            mode=payload.mode,
            show_progress=False,
        )
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/api/v1/stream/sample", tags=["Streaming"])
def sample_stream(
    topic: str = Query("tech", regex="^(tech|crypto|aviation|ecommerce)$"),
    count: int = Query(5, ge=1, le=50),
    mode: str = Query("hybrid", regex="^(hybrid|fast|accurate|ensemble)$"),
):
    """Generate a stream sample with live sentiment labels."""
    global stream_simulator
    if stream_simulator is None:
        stream_simulator = SocialStreamSimulator(default_topic=topic)

    p = get_predictor()
    items = []
    for tweet in stream_simulator.stream(topic=topic, max_items=count, interval_seconds=0):
        pred = p.predict(tweet["text"], mode=mode)
        items.append(
            {
                "tweet": tweet,
                "analysis": pred,
            }
        )

    return {
        "topic": topic,
        "count": len(items),
        "mode": mode,
        "items": items,
    }
