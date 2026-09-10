"""TwitterSentiment-Bridge: Hybrid Sentiment Prediction Engine.

Bridges sub-millisecond TF-IDF Logistic Regression baselines with deep
contextual RoBERTa transformers through confidence-gated cascading and ensembles.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import joblib

# Ensure UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src is in search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from preprocess import clean_for_baseline, clean_for_transformer


class HybridSentimentPredictor:
    """Production-grade hybrid sentiment predictor with confidence-gated routing."""

    def __init__(
        self,
        baseline_path: str = "models/baseline_lr.pkl",
        confidence_threshold: float = 0.82,
        load_transformer: bool = True,
    ):
        """Initialize the hybrid predictor.

        Args:
            baseline_path: Path to serialized scikit-learn baseline pipeline.
            confidence_threshold: Confidence score (0.0 to 1.0) required to accept
                the baseline prediction without escalating to the transformer.
            load_transformer: Whether transformer escalation is enabled.
        """
        self.baseline_path = baseline_path
        self.confidence_threshold = confidence_threshold
        self.enable_transformer = load_transformer
        self.baseline = None
        self._hf_model = None
        self._transformer_failed = False

        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load scikit-learn baseline pipeline if present."""
        if os.path.exists(self.baseline_path):
            try:
                self.baseline = joblib.load(self.baseline_path)
            except Exception as e:
                print(f"[Warning] Failed to load baseline at '{self.baseline_path}': {e}", file=sys.stderr)
        else:
            print(
                f"[Info] Baseline model not found at '{self.baseline_path}'. Run src/train_baseline.py to generate it.",
                file=sys.stderr,
            )

    @property
    def hf_model(self):
        """Lazy-load the Hugging Face transformer model on first demand."""
        if self._hf_model is not None:
            return self._hf_model
        if not self.enable_transformer or self._transformer_failed:
            return None

        try:
            print("[Info] Loading Hugging Face Twitter-RoBERTa model (first run may download weights)...", file=sys.stderr)
            from hf_model import TwitterSentimentTransformer
            self._hf_model = TwitterSentimentTransformer()
            print("[Info] Hugging Face Twitter-RoBERTa successfully loaded.", file=sys.stderr)
            return self._hf_model
        except Exception as e:
            self._transformer_failed = True
            print(f"[Warning] Failed to load Hugging Face model: {e}. Cascading will rely on baseline.", file=sys.stderr)
            return None

    def _predict_baseline(self, text: str) -> Dict[str, Any]:
        """Run inference using TF-IDF Logistic Regression baseline."""
        t0 = time.perf_counter()
        cleaned = clean_for_baseline(text)

        if not self.baseline:
            raise RuntimeError("Baseline model is not loaded.")

        probs = self.baseline.predict_proba([cleaned])[0]
        neg_prob, pos_prob = float(probs[0]), float(probs[1])

        margin = abs(pos_prob - neg_prob)
        neutral_prob = max(0.0, 1.0 - (margin * 1.5))
        remainder = max(0.0, 1.0 - neutral_prob)

        if pos_prob >= neg_prob:
            norm_pos = pos_prob / (pos_prob + neg_prob) * remainder
            norm_neg = neg_prob / (pos_prob + neg_prob) * remainder
        else:
            norm_pos = pos_prob / (pos_prob + neg_prob) * remainder
            norm_neg = neg_prob / (pos_prob + neg_prob) * remainder

        p_dist = {
            "negative": round(norm_neg, 4),
            "neutral": round(neutral_prob, 4),
            "positive": round(norm_pos, 4),
        }

        total = sum(p_dist.values()) or 1.0
        p_dist = {k: round(v / total, 4) for k, v in p_dist.items()}

        best_label = max(p_dist, key=p_dist.get)
        confidence = max(neg_prob, pos_prob)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "label": best_label,
            "score": round(confidence, 4),
            "probabilities": p_dist,
            "latency_ms": latency_ms,
            "model_used": "baseline_lr",
        }

    def _predict_transformer(self, text: str) -> Dict[str, Any]:
        """Run inference using Hugging Face Twitter-RoBERTa."""
        if not self.hf_model:
            raise RuntimeError("Hugging Face model is not available.")

        cleaned = clean_for_transformer(text)
        res = self.hf_model.predict([cleaned])[0]
        res["model_used"] = "roberta_transformer"
        return res

    def predict(
        self,
        text: str,
        mode: str = "hybrid",
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment of a single tweet."""
        threshold = confidence_threshold or self.confidence_threshold
        mode = mode.lower()
        t_start = time.perf_counter()

        if mode in ("fast", "baseline"):
            if self.baseline:
                result = self._predict_baseline(text)
                result["text"] = text
                result["mode"] = "fast"
                result["routing_reason"] = "forced_baseline"
                return result
            elif self.hf_model:
                result = self._predict_transformer(text)
                result["text"] = text
                result["mode"] = "fast"
                result["routing_reason"] = "fallback_to_transformer"
                return result
            else:
                raise RuntimeError("No models available.")

        elif mode in ("accurate", "roberta"):
            if self.hf_model:
                result = self._predict_transformer(text)
                result["text"] = text
                result["mode"] = "accurate"
                result["routing_reason"] = "forced_transformer"
                return result
            elif self.baseline:
                result = self._predict_baseline(text)
                result["text"] = text
                result["mode"] = "accurate"
                result["routing_reason"] = "fallback_to_baseline"
                return result
            else:
                raise RuntimeError("No models available.")

        elif mode == "ensemble":
            if self.hf_model and self.baseline:
                r_hf = self._predict_transformer(text)
                r_base = self._predict_baseline(text)

                blended_probs = {}
                for k in ("negative", "neutral", "positive"):
                    blended_probs[k] = round(
                        (r_hf["probabilities"].get(k, 0.0) * 0.65)
                        + (r_base["probabilities"].get(k, 0.0) * 0.35),
                        4,
                    )
                best_label = max(blended_probs, key=blended_probs.get)
                total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "text": text,
                    "label": best_label,
                    "score": blended_probs[best_label],
                    "probabilities": blended_probs,
                    "model_used": "ensemble_blend",
                    "mode": "ensemble",
                    "routing_reason": "weighted_ensemble",
                    "latency_ms": total_latency,
                }
            elif self.hf_model:
                return self.predict(text, mode="accurate")
            else:
                return self.predict(text, mode="fast")

        # Default: HYBRID Cascading
        if self.baseline:
            base_res = self._predict_baseline(text)
            if base_res["score"] >= threshold or not self.enable_transformer:
                base_res["text"] = text
                base_res["mode"] = "hybrid"
                base_res["routing_reason"] = f"baseline_confident (score {base_res['score']} >= {threshold})"
                return base_res

            if self.hf_model:
                try:
                    tf_res = self._predict_transformer(text)
                    tf_res["text"] = text
                    tf_res["mode"] = "hybrid"
                    tf_res["routing_reason"] = f"escalated_to_transformer (baseline score {base_res['score']} < {threshold})"
                    tf_res["baseline_score"] = base_res["score"]
                    tf_res["latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
                    return tf_res
                except Exception as e:
                    print(f"[Warning] Transformer failed during escalation: {e}. Using baseline.", file=sys.stderr)
                    base_res["text"] = text
                    base_res["mode"] = "hybrid"
                    base_res["routing_reason"] = "transformer_exception_fallback"
                    return base_res
            else:
                base_res["text"] = text
                base_res["mode"] = "hybrid"
                base_res["routing_reason"] = "transformer_unavailable_baseline_accepted"
                return base_res

        elif self.hf_model:
            tf_res = self._predict_transformer(text)
            tf_res["text"] = text
            tf_res["mode"] = "hybrid"
            tf_res["routing_reason"] = "transformer_only (baseline unavailable)"
            return tf_res

        return {
            "text": text,
            "label": "unknown",
            "score": 0.0,
            "probabilities": {},
            "model_used": "none",
            "mode": mode,
            "routing_reason": "no_models_loaded",
            "latency_ms": 0.0,
        }

    def predict_batch(
        self,
        texts: List[str],
        mode: str = "hybrid",
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Predict sentiment for a collection of tweets."""
        return [self.predict(t, mode=mode, confidence_threshold=confidence_threshold) for t in texts]


def run_streaming_cli(predictor: HybridSentimentPredictor, topic: str, count: Optional[int], mode: str):
    """Run real-time streaming simulation in CLI."""
    from stream_simulator import SocialStreamSimulator

    simulator = SocialStreamSimulator(default_topic=topic)
    print("=" * 75)
    print(f"  TwitterSentiment-Bridge Real-Time Stream Monitor 📡")
    print(f"  Topic: {topic.upper()} | Mode: {mode.upper()} | Limit: {count or 'Infinite'}")
    print("  Press Ctrl+C to stop streaming.")
    print("=" * 75)

    pos_count = 0
    neg_count = 0
    neu_count = 0
    total = 0

    try:
        for tweet in simulator.stream(topic=topic, max_items=count, interval_seconds=0.7):
            total += 1
            pred = predictor.predict(tweet["text"], mode=mode)
            lbl = pred["label"].upper()

            if lbl == "POSITIVE":
                pos_count += 1
                icon = "🟢 POS"
            elif lbl == "NEGATIVE":
                neg_count += 1
                icon = "🔴 NEG"
            else:
                neu_count += 1
                icon = "🟡 NEU"

            pos_pct = round((pos_count / total) * 100, 1)
            neg_pct = round((neg_count / total) * 100, 1)

            print(f"[{tweet['timestamp'][11:19]}] {icon} ({pred['score']:.2f}) | {tweet['handle']}: {tweet['text']}")
            print(f"    ↳ Engine: {pred['model_used']} | Latency: {pred['latency_ms']}ms | Live Sentiment: {pos_pct}% Pos, {neg_pct}% Neg\n")

    except KeyboardInterrupt:
        print("\nStream halted by user.")

    print(f"\nFinal Stream Totals: {total} tweets (Pos: {pos_count}, Neg: {neg_count}, Neu: {neu_count})")


def main():
    parser = argparse.ArgumentParser(
        description="TwitterSentiment-Bridge: Hybrid RoBERTa + TF-IDF Sentiment Predictor"
    )
    parser.add_argument("--text", type=str, help="Single tweet text to analyze.")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "fast", "accurate", "ensemble"],
        default="hybrid",
        help="Prediction routing strategy (default: hybrid).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Confidence threshold for hybrid escalation (default: 0.82).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable summary.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive sentiment REPL.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to CSV, JSON, or TXT file to process in batch mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Destination path for batch results (CSV or JSON).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Launch simulated real-time streaming feed monitor.",
    )
    parser.add_argument(
        "--topic",
        choices=["tech", "crypto", "aviation", "ecommerce"],
        default="tech",
        help="Topic domain for simulated social stream (default: tech).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of tweets to process in stream mode (default: infinite).",
    )

    args = parser.parse_args()

    predictor = HybridSentimentPredictor(confidence_threshold=args.threshold)

    # 1. Stream Mode
    if args.stream:
        run_streaming_cli(predictor, topic=args.topic, count=args.count, mode=args.mode)
        return

    # 2. Batch Mode
    if args.batch:
        from batch_processor import BatchSentimentProcessor

        processor = BatchSentimentProcessor(predictor=predictor)
        out = processor.process_file(args.batch, output_file=args.output, mode=args.mode)
        if args.json:
            print(json.dumps(out["summary"], indent=2))
        else:
            s = out["summary"]
            print("\n" + "=" * 60)
            print("         BATCH PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Total Tweets:      {s['total_processed']}")
            print(f"Total Time:        {s['total_time_sec']} s")
            print(f"Throughput:        {s['throughput_tweets_per_sec']} tweets/sec")
            print(f"Average Latency:   {s['average_latency_ms']} ms")
            print(f"Average Conf:      {s['average_confidence']:.4f}")
            print(f"Sentiment Split:   {s['sentiment_distribution']}")
            print(f"Model Routing:     {s['model_routing_breakdown']}")
            if args.output:
                print(f"Results Saved To:  {args.output}")
            print("=" * 60 + "\n")
        return

    # 3. Interactive Shell
    if args.interactive or (not args.text and len(sys.argv) == 1):
        print("=" * 65)
        print("  TwitterSentiment-Bridge Interactive Shell 🐦")
        print("  Mode:", args.mode.upper(), "| Threshold:", args.threshold)
        print("  Type 'exit', 'quit', or press Ctrl+C to terminate.")
        print("=" * 65)

        while True:
            try:
                tweet = input("\nEnter tweet: ").strip()
                if not tweet:
                    continue
                if tweet.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                res = predictor.predict(tweet, mode=args.mode)
                if args.json:
                    print(json.dumps(res, indent=2))
                else:
                    print(f"  [Result]      Label: {res['label'].upper()} (Score: {res['score']:.4f})")
                    print(f"  [Engine]      Model: {res['model_used']} | Reason: {res.get('routing_reason', 'N/A')}")
                    print(f"  [Latency]     {res['latency_ms']} ms")
                    print(f"  [Breakdown]   Negative: {res['probabilities'].get('negative', 0):.2f} | "
                          f"Neutral: {res['probabilities'].get('neutral', 0):.2f} | "
                          f"Positive: {res['probabilities'].get('positive', 0):.2f}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break
        return

    # 4. Single-shot prediction
    text = args.text or "This project is really helpful for learning NLP!"
    res = predictor.predict(text, mode=args.mode)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\nTweet: \"{text}\"")
        print(f"Prediction: {res['label'].upper()} (Confidence: {res['score']:.4f})")
        print(f"Model Used: {res['model_used']} ({res.get('routing_reason', '')})")
        print(f"Latency:    {res['latency_ms']} ms")
        print(f"Probabilities: {res['probabilities']}\n")


if __name__ == "__main__":
    main()
