"""Pydantic request and response schemas for TwitterSentiment-Bridge REST API."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Payload for single-tweet sentiment analysis."""

    text: str = Field(..., min_length=1, max_length=1000, description="Raw tweet text to analyze")
    mode: Literal["hybrid", "fast", "accurate", "ensemble"] = Field(
        default="hybrid",
        description="Prediction routing strategy",
    )
    confidence_threshold: Optional[float] = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Gating threshold for hybrid escalation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The new update resolved all latency spikes! 🚀 Love this framework.",
                "mode": "hybrid",
                "confidence_threshold": 0.82,
            }
        }
    }


class SentimentResponse(BaseModel):
    """Result schema for sentiment analysis."""

    text: str
    label: str
    score: float
    probabilities: Dict[str, float]
    model_used: str
    mode: str
    routing_reason: Optional[str] = None
    latency_ms: float


class BatchPredictRequest(BaseModel):
    """Payload for batch tweet analysis."""

    texts: List[str] = Field(..., min_length=1, max_length=1000, description="List of tweet strings")
    mode: Literal["hybrid", "fast", "accurate", "ensemble"] = Field(
        default="hybrid",
        description="Prediction routing strategy",
    )
    confidence_threshold: Optional[float] = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for hybrid escalation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Blazingly fast inference and beautiful UI.",
                    "Terrible support, wasted 4 hours of my day.",
                    "The package arrived on Tuesday.",
                ],
                "mode": "hybrid",
            }
        }
    }


class BatchSummary(BaseModel):
    """Statistical aggregate summary for batch predictions."""

    total_processed: int
    total_time_sec: float
    throughput_tweets_per_sec: float
    average_latency_ms: float
    average_confidence: float
    sentiment_distribution: Dict[str, int]
    sentiment_distribution_pct: Dict[str, float]
    model_routing_breakdown: Dict[str, Any]


class BatchPredictResponse(BaseModel):
    """Batch prediction results and analytics rollup."""

    summary: BatchSummary
    results: List[SentimentResponse]


class HealthResponse(BaseModel):
    """System health check and engine readiness."""

    status: str
    version: str
    baseline_loaded: bool
    transformer_loaded: bool
    default_threshold: float
    device: str
