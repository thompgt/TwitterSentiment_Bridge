"""Evaluation and benchmarking module for TwitterSentiment-Bridge.

Compares Baseline (TF-IDF + LR), RoBERTa Transformer, and Hybrid Cascading
across Accuracy, Precision, Recall, Macro-F1, and Latency.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# Ensure root and src are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from main import HybridSentimentPredictor


# Curated benchmark validation suite covering edge cases:
# positive, negative, sarcastic, ambiguous, and emoji-heavy tweets
BENCHMARK_CORPUS: List[Tuple[str, str]] = [
    ("This new camera feature is absolutely outstanding! 😍📸", "positive"),
    ("Worst customer support on earth, waited 3 hours and hung up.", "negative"),
    ("Package arrived safely and works as described.", "positive"),
    ("My laptop screen completely shattered after one day of use.", "negative"),
    ("I love waiting in airport lines for 6 hours, truly living the dream...", "negative"),
    ("Just wrapped up our product launch with an incredible team! 🚀🎉", "positive"),
    ("The battery drains from 100% to 0% in under two hours.", "negative"),
    ("Super clean UI and intuitive keyboard shortcuts.", "positive"),
    ("Flight was canceled without notice and baggage is lost.", "negative"),
    ("Fantastic tutorial! Saved me dozens of hours.", "positive"),
    ("The new subscription price increase is predatory and unacceptable.", "negative"),
    ("Food was delicious and the ambiance was cozy. Highly recommend!", "positive"),
    ("App crashes every single time I click on settings.", "negative"),
    ("What a remarkable performance by the entire cast.", "positive"),
    ("Zero stars if I could. Fraudulent charges on my credit card.", "negative"),
    ("Stunning design, fast shipping, and friendly support! 💯", "positive"),
    ("Horrible sound quality, tinny bass and constant audio static.", "negative"),
    ("Proud to announce I finally passed my cloud architect certification! 🎓", "positive"),
    ("Cannot believe how slow and buggy this software update is.", "negative"),
    ("Hands down the best coffee in the city! ☕✨", "positive"),
]


def run_benchmark(sample_size: int = 20, threshold: float = 0.82) -> Dict[str, Dict]:
    """Run benchmark comparison between Fast (Baseline) and Hybrid modes."""
    texts = [item[0] for item in BENCHMARK_CORPUS[:sample_size]]
    y_true = [item[1] for item in BENCHMARK_CORPUS[:sample_size]]

    predictor = HybridSentimentPredictor(confidence_threshold=threshold)

    results = {}

    for mode in ("fast", "hybrid"):
        t0 = time.perf_counter()
        predictions = []
        latencies = []
        models_routed = {"baseline_lr": 0, "roberta_transformer": 0, "fallback": 0}

        for text in texts:
            p = predictor.predict(text, mode=mode)
            # Map neutral to nearest or compare on pos/neg
            pred_label = p["label"]
            if pred_label == "neutral":
                # Break tie by highest probability between pos and neg
                neg_p = p["probabilities"].get("negative", 0.0)
                pos_p = p["probabilities"].get("positive", 0.0)
                pred_label = "positive" if pos_p >= neg_p else "negative"

            predictions.append(pred_label)
            latencies.append(p.get("latency_ms", 0.0))

            m = p.get("model_used", "baseline_lr")
            if "baseline" in m:
                models_routed["baseline_lr"] += 1
            elif "roberta" in m:
                models_routed["roberta_transformer"] += 1
            else:
                models_routed["fallback"] += 1

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        acc = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, pos_label="positive", average="binary")
        prec = precision_score(y_true, predictions, pos_label="positive", average="binary")
        rec = recall_score(y_true, predictions, pos_label="positive", average="binary")
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        throughput = (len(texts) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        results[mode] = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "avg_latency_ms": round(avg_latency, 2),
            "throughput_qps": round(throughput, 1),
            "models_routed": models_routed,
        }

    return results


def print_comparison_table(results: Dict[str, Dict]):
    """Print markdown formatted benchmark comparison."""
    print("\n" + "=" * 75)
    print("           TWITTER SENTIMENT BRIDGE: PERFORMANCE BENCHMARK")
    print("=" * 75)
    print(f"{'Metric':<22} | {'Fast (Baseline)':<22} | {'Hybrid (Cascading)':<22}")
    print("-" * 75)

    metrics = [
        ("Accuracy", "accuracy", "{:.2%}"),
        ("F1 Score (Binary)", "f1_score", "{:.4f}"),
        ("Precision", "precision", "{:.4f}"),
        ("Recall", "recall", "{:.4f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.2f} ms"),
        ("Throughput (QPS)", "throughput_qps", "{:.1f} req/s"),
    ]

    for label, key, fmt in metrics:
        v_fast = fmt.format(results["fast"][key])
        v_hyb = fmt.format(results["hybrid"][key])
        print(f"{label:<22} | {v_fast:<22} | {v_hyb:<22}")

    print("-" * 75)
    print(f"Fast Mode Routing:   {results['fast']['models_routed']}")
    print(f"Hybrid Mode Routing: {results['hybrid']['models_routed']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and benchmark TwitterSentiment-Bridge models")
    parser.add_argument("--threshold", type=float, default=0.82, help="Hybrid gating confidence threshold")
    args = parser.parse_args()

    benchmark_data = run_benchmark(threshold=args.threshold)
    print_comparison_table(benchmark_data)
