"""
FixFirst AI Dashboard — Streamlit frontend.

Three views:
  - Overview: High Priority (Needs Work) / Low Priority (Backlog/Stable)
    feature lists — the core "instead of a vague 3.5-star rating" pitch.
  - Trends: per-feature score-over-time chart, so a developer can see if
    a deployment helped or hurt.
  - Reviews: filterable review browser with per-feature sentiment tags.

Run:
    PYTHONPATH=src streamlit run src/fixfirst/dashboard/app.py

All HTTP calls and data shaping live in api_client.py — this file only
renders. That split is what makes api_client.py unit-testable without a
Streamlit runtime.
"""

import sys

sys.path.insert(0, "src")  # allows `streamlit run` from repo root without PYTHONPATH set

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fixfirst.dashboard import api_client
from fixfirst.exceptions.exception import FixFirstException

st.set_page_config(page_title="FixFirst AI", page_icon="🛠️", layout="wide")


def render_sidebar() -> str:
    st.sidebar.title("🛠️ FixFirst AI")
    st.sidebar.caption("Automated feature prioritization from user reviews")

    healthy = api_client.check_api_health()
    if healthy:
        st.sidebar.success("API: connected")
    else:
        st.sidebar.error(f"API unreachable at {api_client.settings.dashboard_api_base_url}")

    return st.sidebar.radio("View", ["Overview", "Trends", "Reviews"])


def render_score_table(rows: list, empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return

    df = pd.DataFrame(rows)
    df["score"] = df["score"].round(3)
    df["negative_ratio"] = (df["negative_ratio"] * 100).round(1).astype(str) + "%"
    df["trend_delta"] = df["trend_delta"].apply(lambda d: f"{d:+.3f}" if pd.notna(d) else "—")

    display_df = df.rename(
        columns={
            "display_name": "Feature",
            "score": "Criticality Score",
            "mention_count": "Mentions",
            "negative_ratio": "% Negative",
            "trend_delta": "Trend Δ",
            "window_start": "Window Start",
            "window_end": "Window End",
        }
    )[["Feature", "Criticality Score", "Mentions", "% Negative", "Trend Δ", "Window Start", "Window End"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_overview() -> None:
    st.header("Overview")
    st.caption("Each feature's most recent scoring window. See the Trends tab for history.")

    try:
        high_priority = api_client.get_criticality_scores(priority="high", limit=20)
        low_priority = api_client.get_criticality_scores(priority="low", limit=20)
    except FixFirstException as e:
        st.error(f"Failed to load criticality scores: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔴 High Priority — Needs Work")
        render_score_table(high_priority, "No scored features yet — run the scoring pipeline first.")
    with col2:
        st.subheader("🟢 Low Priority — Backlog / Stable")
        render_score_table(low_priority, "No scored features yet — run the scoring pipeline first.")


def render_trends() -> None:
    st.header("Trends")

    try:
        features = api_client.get_features()
    except FixFirstException as e:
        st.error(f"Failed to load features: {e}")
        return

    if not features:
        st.info("No features found — check that features_master is seeded.")
        return

    options = {f["display_name"]: f["feature_key"] for f in features}
    selected_display = st.selectbox("Feature", list(options.keys()))
    feature_key = options[selected_display]

    try:
        trend = api_client.get_feature_trend(feature_key)
    except FixFirstException as e:
        st.error(f"Failed to load trend for {feature_key}: {e}")
        return

    if trend is None or not trend["points"]:
        st.info(f"No scoring history yet for {selected_display}.")
        return

    points_df = pd.DataFrame(trend["points"])
    points_df["window_start"] = pd.to_datetime(points_df["window_start"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points_df["window_start"],
            y=points_df["score"],
            mode="lines+markers",
            name="Criticality Score",
            line=dict(color="#d62728"),
        )
    )
    fig.update_layout(
        title=f"Criticality Score Over Time — {selected_display}",
        xaxis_title="Window",
        yaxis_title="Score",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Mention volume per window")
    fig2 = go.Figure(go.Bar(x=points_df["window_start"], y=points_df["mention_count"], marker_color="#1f77b4"))
    fig2.update_layout(height=250, xaxis_title="Window", yaxis_title="Mentions")
    st.plotly_chart(fig2, use_container_width=True)


def render_reviews() -> None:
    st.header("Reviews")

    try:
        features = api_client.get_features()
    except FixFirstException as e:
        st.error(f"Failed to load features: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_options = ["All"] + [f["display_name"] for f in features]
        selected_feature_display = st.selectbox("Feature", feature_options)
    with col2:
        selected_sentiment = st.selectbox("Sentiment", ["All", "negative", "neutral", "positive"])
    with col3:
        selected_source = st.selectbox("Source", ["All", "aware", "google_play", "app_store", "github_issues"])

    feature_key = None
    if selected_feature_display != "All":
        feature_key = next(f["feature_key"] for f in features if f["display_name"] == selected_feature_display)
    sentiment = None if selected_sentiment == "All" else selected_sentiment
    source = None if selected_source == "All" else selected_source

    try:
        result = api_client.get_reviews(feature_key=feature_key, sentiment=sentiment, source=source, limit=25)
    except FixFirstException as e:
        st.error(f"Failed to load reviews: {e}")
        return

    st.caption(f"{result['total']} matching reviews (showing up to {result['limit']})")

    for review in result["items"]:
        with st.container(border=True):
            st.write(review["review_text"])
            tag_cols = st.columns(max(len(review["aspects"]), 1))
            for i, aspect in enumerate(review["aspects"]):
                color = {"negative": "red", "neutral": "gray", "positive": "green"}.get(aspect["sentiment"], "gray")
                tag_cols[i].markdown(f":{color}[{aspect['feature_key']}: {aspect['sentiment']}]")


def main() -> None:
    view = render_sidebar()
    if view == "Overview":
        render_overview()
    elif view == "Trends":
        render_trends()
    elif view == "Reviews":
        render_reviews()


if __name__ == "__main__":
    main()