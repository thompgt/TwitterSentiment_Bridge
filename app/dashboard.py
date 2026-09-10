"""Interactive Web Dashboard for TwitterSentiment-Bridge.

Features:
- Single-tweet interactive playground with confidence meters and probability bars.
- Batch CSV dataset uploader with analytical rollups and downloadable enriched data.
- Real-time social stream visualizer with live sentiment tracking across domains.
"""

import io
import os
import sys
import pandas as pd
import streamlit as st

# Add src and root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from batch_processor import BatchSentimentProcessor
from main import HybridSentimentPredictor
from stream_simulator import SocialStreamSimulator

# Set page configuration
st.set_page_config(
    page_title="TwitterSentiment-Bridge",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stMetric {
        background: #151d30;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #24304f;
    }
    .metric-card {
        background: #151d30;
        border: 1px solid #24304f;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-pos {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neg {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neu {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .routing-tag {
        font-family: monospace;
        font-size: 0.8rem;
        color: #93c5fd;
        background: #1e3a8a;
        padding: 3px 8px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor(threshold: float):
    """Cache predictor instance across reruns."""
    return HybridSentimentPredictor(confidence_threshold=threshold)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🐦 TwitterSentiment")
st.sidebar.caption("Hybrid Deep Learning & Linear Engine")

mode = st.sidebar.selectbox(
    "Routing Strategy",
    options=["hybrid", "fast", "accurate", "ensemble"],
    index=0,
    help=(
        "• Hybrid: Cascades high-confidence tweets to TF-IDF baseline, escalates ambiguous to RoBERTa.\n"
        "• Fast: Pure sub-millisecond TF-IDF Logistic Regression.\n"
        "• Accurate: CardiffNLP Twitter-RoBERTa Transformer.\n"
        "• Ensemble: Blended calibrated probability distribution."
    ),
)

threshold = st.sidebar.slider(
    "Confidence Threshold (τ)",
    min_value=0.50,
    max_value=0.99,
    value=0.82,
    step=0.01,
    help="Minimum probability required to accept baseline without escalating to transformer.",
)

predictor = load_predictor(threshold)

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine Readiness**")
base_status = "🟢 Ready" if predictor.baseline is not None else "🔴 Not Loaded"
st.sidebar.write(f"• Baseline LR: {base_status}")
tf_status = "🟢 Ready" if predictor._hf_model is not None else "🟡 On-Demand (Lazy)"
st.sidebar.write(f"• RoBERTa: {tf_status}")

# ==============================================================================
# MAIN INTERFACE TABS
# ==============================================================================
tab_playground, tab_batch, tab_stream = st.tabs(
    ["🎯 Live Playground", "📊 Batch CSV Analyzer", "📡 Real-Time Social Stream"]
)

# ------------------------------------------------------------------------------
# TAB 1: LIVE PLAYGROUND
# ------------------------------------------------------------------------------
with tab_playground:
    st.subheader("Interactive Sentiment Analyzer")
    st.write("Test single tweets with instant model comparison, probability breakdown, and routing telemetry.")

    col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
    preset_tweet = None
    if col_quick1.button("✨ Positive Sample"):
        preset_tweet = "The new battery update is astonishing! Easily lasted 14 hours of video playback. 😍🔥"
    if col_quick2.button("⚠️ Negative Sample"):
        preset_tweet = "Flight got delayed 5 hours with zero explanation and customer service was incredibly rude."
    if col_quick3.button("🤔 Sarcastic / Subtle"):
        preset_tweet = "I just love spending my entire Sunday untangling dependency conflicts... true bliss."
    if col_quick4.button("⚖️ Neutral Sample"):
        preset_tweet = "The quarterly earnings call is scheduled for tomorrow at 2:00 PM EST."

    user_text = st.text_area(
        "Enter Tweet or Social Post:",
        value=preset_tweet or "This hybrid architecture is blazingly fast and accurate! Love the clean interface. 🚀",
        height=100,
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Classifying sentiment..."):
                res = predictor.predict(user_text.strip(), mode=mode, confidence_threshold=threshold)

            # Display top results
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            lbl = res["label"].upper()
            score_pct = f"{res['score'] * 100:.1f}%"

            with res_col1:
                st.metric("Predicted Sentiment", lbl)
            with res_col2:
                st.metric("Confidence Score", score_pct)
            with res_col3:
                st.metric("Engine Used", res["model_used"])
            with res_col4:
                st.metric("Latency", f"{res['latency_ms']} ms")

            st.markdown(f"**Routing Telemetry:** `{res.get('routing_reason', 'N/A')}`")

            # Probability Breakdown
            st.markdown("#### Calibrated Probabilities")
            probs = res.get("probabilities", {})
            p_df = pd.DataFrame(
                {
                    "Class": ["Negative", "Neutral", "Positive"],
                    "Probability": [
                        probs.get("negative", 0.0),
                        probs.get("neutral", 0.0),
                        probs.get("positive", 0.0),
                    ],
                }
            )
            st.bar_chart(p_df.set_index("Class"), color="#3b82f6", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: BATCH CSV ANALYZER
# ------------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Dataset Ingestion & Analytics")
    st.write("Upload a CSV file containing social media posts or analyze our built-in benchmark dataset.")

    uploaded_file = st.file_uploader("Upload CSV (must have 'text', 'tweet', or 'content' column):", type=["csv"])

    use_sample = st.checkbox("Or use built-in 20-tweet validation corpus", value=uploaded_file is None)

    if uploaded_file is not None or use_sample:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            from evaluate import BENCHMARK_CORPUS
            df = pd.DataFrame(BENCHMARK_CORPUS, columns=["text", "ground_truth"])

        st.write(f"Loaded **{len(df)}** rows. Preview:")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("Run Batch Sentiment Classification", type="primary"):
            processor = BatchSentimentProcessor(predictor=predictor)
            col_target = next(
                (c for c in df.columns if c.lower() in ("text", "tweet", "content")),
                df.columns[0],
            )
            texts = df[col_target].astype(str).tolist()

            prog_bar = st.progress(0, text="Processing batch...")
            out = processor.process_texts(texts, mode=mode, show_progress=False)
            prog_bar.progress(100, text="Batch processing complete!")

            s = out["summary"]
            # KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Processed", s["total_processed"])
            kpi2.metric("Throughput", f"{s['throughput_tweets_per_sec']} tweets/s")
            kpi3.metric("Avg Latency", f"{s['average_latency_ms']} ms")
            kpi4.metric("Avg Confidence", f"{s['average_confidence'] * 100:.1f}%")

            # Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Sentiment Distribution")
                dist_df = pd.DataFrame(
                    list(s["sentiment_distribution"].items()),
                    columns=["Sentiment", "Count"],
                )
                st.bar_chart(dist_df.set_index("Sentiment"), use_container_width=True)

            with chart_col2:
                st.markdown("#### Model Routing Breakdown")
                routing_data = [
                    {"Model": k, "Count": v["count"], "Percentage": v["percentage"]}
                    for k, v in s["model_routing_breakdown"].items()
                ]
                st.dataframe(pd.DataFrame(routing_data), use_container_width=True)

            # Exportable CSV
            results_df = pd.DataFrame(
                [
                    {
                        "text": r["text"],
                        "sentiment": r["label"],
                        "confidence": r["score"],
                        "negative_prob": r["probabilities"].get("negative", 0.0),
                        "neutral_prob": r["probabilities"].get("neutral", 0.0),
                        "positive_prob": r["probabilities"].get("positive", 0.0),
                        "model": r["model_used"],
                        "latency_ms": r.get("latency_ms", 0.0),
                    }
                    for r in out["results"]
                ]
            )
            st.markdown("#### Annotated Output Table")
            st.dataframe(results_df, use_container_width=True)

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Annotated CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_annotated_results.csv",
                mime="text/csv",
            )

# ------------------------------------------------------------------------------
# TAB 3: REAL-TIME SOCIAL STREAM
# ------------------------------------------------------------------------------
with tab_stream:
    st.subheader("Simulated Real-Time Social Stream Monitor")
    st.write("Observe live simulated social feeds across domains with real-time hybrid sentiment classification.")

    s_col1, s_col2, s_col3 = st.columns(3)
    stream_topic = s_col1.selectbox("Topic Feed", ["tech", "crypto", "aviation", "ecommerce"], index=0)
    stream_count = s_col2.slider("Number of Posts to Pull", min_value=3, max_value=25, value=8)
    fetch_stream = s_col3.button("📡 Fetch Live Stream Feed", type="primary", use_container_width=True)

    if fetch_stream:
        simulator = SocialStreamSimulator(default_topic=stream_topic)
        tweets = list(simulator.stream(topic=stream_topic, max_items=stream_count, interval_seconds=0))

        pos_cnt = 0
        neg_cnt = 0
        neu_cnt = 0

        stream_cards = []
        for t in tweets:
            p = predictor.predict(t["text"], mode=mode, confidence_threshold=threshold)
            lbl = p["label"].lower()
            if lbl == "positive":
                pos_cnt += 1
                badge_html = '<span class="badge-pos">POSITIVE</span>'
            elif lbl == "negative":
                neg_cnt += 1
                badge_html = '<span class="badge-neg">NEGATIVE</span>'
            else:
                neu_cnt += 1
                badge_html = '<span class="badge-neu">NEUTRAL</span>'

            stream_cards.append(
                f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div><strong>{t['user']}</strong> <span style="color:#9ca3af;">{t['handle']}</span> • <span style="font-size:0.8rem; color:#6b7280;">{t['timestamp'][11:19]} UTC</span></div>
                        <div>{badge_html} <span class="routing-tag">{p['model_used']} ({p['latency_ms']}ms)</span></div>
                    </div>
                    <div style="font-size:1.05rem; margin-bottom:8px;">{t['text']}</div>
                    <div style="font-size:0.8rem; color:#9ca3af;">Score: <strong>{p['score']:.2f}</strong> | Neg: {p['probabilities'].get('negative', 0):.2f} • Neu: {p['probabilities'].get('neutral', 0):.2f} • Pos: {p['probabilities'].get('positive', 0):.2f}</div>
                </div>
                """
            )

        # Stream Aggregate KPIs
        total_p = max(len(tweets), 1)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stream Feed Volume", len(tweets))
        k1.write(f"Domain: **{stream_topic.upper()}**")
        k2.metric("Positive Posts", f"{pos_cnt} ({pos_cnt/total_p*100:.1f}%)")
        k3.metric("Negative Posts", f"{neg_cnt} ({neg_cnt/total_p*100:.1f}%)")
        k4.metric("Neutral Posts", f"{neu_cnt} ({neu_cnt/total_p*100:.1f}%)")

        st.markdown("---")
        st.markdown("#### Live Tweet Stream")
        for card in stream_cards:
            st.markdown(card, unsafe_allow_html=True)
