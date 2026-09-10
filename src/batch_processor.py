"""Batch processing engine for TwitterSentiment-Bridge.

Supports high-throughput processing of CSV, JSON, and text datasets with
chunking, progress tracking, statistical aggregations, and export capabilities.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure search paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import HybridSentimentPredictor


class BatchSentimentProcessor:
    """Processes large collections of tweets and generates analytical rollups."""

    def __init__(
        self,
        predictor: Optional[HybridSentimentPredictor] = None,
        confidence_threshold: float = 0.82,
    ):
        """Initialize batch processor with an underlying hybrid predictor."""
        self.predictor = predictor or HybridSentimentPredictor(confidence_threshold=confidence_threshold)

    def process_texts(
        self,
        texts: List[str],
        mode: str = "hybrid",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Process a list of tweet strings with performance tracking."""
        t_start = time.perf_counter()
        results = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing Tweets", unit="batch")

        for i in iterator:
            batch = texts[i : i + batch_size]
            for text in batch:
                pred = self.predictor.predict(text, mode=mode)
                results.append(pred)

        total_elapsed = time.perf_counter() - t_start
        throughput = len(texts) / total_elapsed if total_elapsed > 0 else 0.0

        # Statistical rollups
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        models = [r.get("model_used", "unknown") for r in results]
        latencies = [r.get("latency_ms", 0.0) for r in results]

        distribution = {
            "positive": labels.count("positive"),
            "negative": labels.count("negative"),
            "neutral": labels.count("neutral"),
        }
        total_items = max(len(texts), 1)
        distribution_pct = {k: round((v / total_items) * 100.0, 2) for k, v in distribution.items()}

        model_breakdown = {}
        for m in set(models):
            cnt = models.count(m)
            model_breakdown[m] = {
                "count": cnt,
                "percentage": round((cnt / total_items) * 100.0, 2),
            }

        summary = {
            "total_processed": len(texts),
            "total_time_sec": round(total_elapsed, 3),
            "throughput_tweets_per_sec": round(throughput, 1),
            "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "average_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
            "sentiment_distribution": distribution,
            "sentiment_distribution_pct": distribution_pct,
            "model_routing_breakdown": model_breakdown,
        }

        return {
            "summary": summary,
            "results": results,
        }

    def process_file(
        self,
        input_file: str,
        text_column: Optional[str] = None,
        output_file: Optional[str] = None,
        mode: str = "hybrid",
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Load, process, and optionally write predictions from a CSV, JSON, or text file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        ext = os.path.splitext(input_file)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(input_file)
            col = text_column or next(
                (c for c in df.columns if c.lower() in ("text", "tweet", "content", "message")),
                df.columns[0],
            )
            texts = df[col].astype(str).tolist()
        elif ext == ".json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    col = text_column or next(
                        (c for c in data[0].keys() if c.lower() in ("text", "tweet", "content")),
                        list(data[0].keys())[0],
                    )
                    texts = [str(item[col]) for item in data]
                else:
                    texts = [str(x) for x in data]
            else:
                raise ValueError("JSON file must contain an array of objects or strings.")
        else:
            # Plain text file, one tweet per line
            with open(input_file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

        batch_output = self.process_texts(texts, mode=mode, batch_size=batch_size)

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            out_ext = os.path.splitext(output_file)[1].lower()
            if out_ext == ".json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(batch_output, f, indent=2)
            else:
                # Save as enriched CSV
                records = []
                for item in batch_output["results"]:
                    rec = {
                        "text": item.get("text", ""),
                        "sentiment_label": item.get("label", ""),
                        "confidence_score": item.get("score", 0.0),
                        "prob_negative": item.get("probabilities", {}).get("negative", 0.0),
                        "prob_neutral": item.get("probabilities", {}).get("neutral", 0.0),
                        "prob_positive": item.get("probabilities", {}).get("positive", 0.0),
                        "model_used": item.get("model_used", ""),
                        "latency_ms": item.get("latency_ms", 0.0),
                    }
                    records.append(rec)
                out_df = pd.DataFrame(records)
                out_df.to_csv(output_file, index=False)

        return batch_output


if __name__ == "__main__":
    sample_tweets = [
        "Unbelievable battery life on this laptop! Loving every minute. 🔥",
        "Worst experience ever. Do not purchase from this vendor.",
        "Flight delay was 20 minutes, not a big deal.",
        "The interface feels so sluggish after yesterday's patch.",
        "Shoutout to support team member Alex for solving my billing issue in 2 mins! 👏",
    ]
    processor = BatchSentimentProcessor()
    out = processor.process_texts(sample_tweets, mode="fast")
    print(json.dumps(out["summary"], indent=2))
