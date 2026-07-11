"""FixFirst AI dashboard rendered with Streamlit."""

import sys

sys.path.insert(0, "src")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fixfirst.dashboard import api_client
from fixfirst.exceptions.exception import FixFirstException

st.set_page_config(page_title="FixFirst AI", page_icon="🛠️", layout="wide")


def render_sidebar() -> str:
    """Render the dashboard sidebar and return the selected view."""
    try:
        st.sidebar.title("🛠️ FixFirst AI")
        st.sidebar.caption("Automated feature prioritization from user reviews")

        healthy = api_client.check_api_health()
        if healthy:
            st.sidebar.success("API: connected")
        else:
            st.sidebar.error(f"API unreachable at {api_client.settings.dashboard_api_base_url}")

        return st.sidebar.radio("View", ["Overview", "Trends", "Reviews"])
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def render_score_table(rows: list, empty_message: str) -> None:
    """Render a formatted criticality table."""
    try:
        if not rows:
            st.info(empty_message)
            return

        df = pd.DataFrame(rows)
        df["score"] = df["score"].round(3)
        df["negative_ratio"] = (df["negative_ratio"] * 100).round(1).astype(str) + "%"
        df["trend_delta"] = df["trend_delta"].apply(lambda delta: f"{delta:+.3f}" if pd.notna(delta) else "—")

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
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def render_overview() -> None:
    """Render the overview page."""
    try:
        st.header("Overview")
        st.caption("Each feature's most recent scoring window. See the Trends tab for history.")

        high_priority = api_client.get_criticality_scores(priority="high", limit=20)
        low_priority = api_client.get_criticality_scores(priority="low", limit=20)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔴 High Priority — Needs Work")
            render_score_table(high_priority, "No scored features yet — run the scoring pipeline first.")
        with col2:
            st.subheader("🟢 Low Priority — Backlog / Stable")
            render_score_table(low_priority, "No scored features yet — run the scoring pipeline first.")
    except FixFirstException as exc:
        st.error(f"Failed to load criticality scores: {exc}")
    except Exception as exc:
        st.error(f"Failed to render overview: {exc}")


def render_trends() -> None:
    """Render the feature trend page."""
    try:
        st.header("Trends")
        features = api_client.get_features()

        if not features:
            st.info("No features found — check that features_master is seeded.")
            return

        options = {feature["display_name"]: feature["feature_key"] for feature in features}
        selected_display = st.selectbox("Feature", list(options.keys()))
        feature_key = options[selected_display]

        trend = api_client.get_feature_trend(feature_key)
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
    except FixFirstException as exc:
        st.error(f"Failed to load trend data: {exc}")
    except Exception as exc:
        st.error(f"Failed to render trends: {exc}")


def render_reviews() -> None:
    """Render the review browser page."""
    try:
        st.header("Reviews")
        features = api_client.get_features()

        col1, col2, col3 = st.columns(3)
        with col1:
            feature_options = ["All"] + [feature["display_name"] for feature in features]
            selected_feature_display = st.selectbox("Feature", feature_options)
        with col2:
            selected_sentiment = st.selectbox("Sentiment", ["All", "negative", "neutral", "positive"])
        with col3:
            selected_source = st.selectbox("Source", ["All", "aware", "google_play", "app_store", "github_issues"])

        feature_key = None
        if selected_feature_display != "All":
            feature_key = next(feature["feature_key"] for feature in features if feature["display_name"] == selected_feature_display)
        sentiment = None if selected_sentiment == "All" else selected_sentiment
        source = None if selected_source == "All" else selected_source

        result = api_client.get_reviews(feature_key=feature_key, sentiment=sentiment, source=source, limit=25)

        st.caption(f"{result['total']} matching reviews (showing up to {result['limit']})")

        for review in result["items"]:
            with st.container(border=True):
                st.write(review["review_text"])
                tag_cols = st.columns(max(len(review["aspects"]), 1))
                for index, aspect in enumerate(review["aspects"]):
                    color = {"negative": "red", "neutral": "gray", "positive": "green"}.get(aspect["sentiment"], "gray")
                    tag_cols[index].markdown(f":{color}[{aspect['feature_key']}: {aspect['sentiment']}]")
    except FixFirstException as exc:
        st.error(f"Failed to load reviews: {exc}")
    except Exception as exc:
        st.error(f"Failed to render reviews: {exc}")


def main() -> None:
    """Run the dashboard application."""
    view = render_sidebar()
    if view == "Overview":
        render_overview()
    elif view == "Trends":
        render_trends()
    elif view == "Reviews":
        render_reviews()


if __name__ == "__main__":
    main()