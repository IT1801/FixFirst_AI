import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure we can import from src
src_path = str(Path(__file__).resolve().parents[3])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fixfirst.dashboard import api_client

st.set_page_config(page_title="FixFirst AI - Analytics", page_icon="📈", layout="wide")

st.sidebar.title("🛠️ FixFirst AI")
st.sidebar.caption("Automated Feature Prioritization")

st.title("📈 Analytics")
st.subheader("Historical feature prioritization and overall trends from the database")

st.markdown("---")

healthy = api_client.check_api_health()
if not healthy:
    st.error(f"Cannot connect to the API at {api_client.settings.dashboard_api_base_url}")
    st.stop()

# 1. Top Features Overview
st.header("Overall Feature Statistics")
high_priority = api_client.get_criticality_scores(priority="high", limit=10)
low_priority = api_client.get_criticality_scores(priority="low", limit=10)

def format_criticality(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["score"] = df["score"].round(3)
    df["negative_ratio"] = (df["negative_ratio"] * 100).round(1).astype(str) + "%"
    df["trend_delta"] = df["trend_delta"].apply(lambda delta: f"{delta:+.3f}" if pd.notna(delta) else "—")
    
    return df.rename(
        columns={
            "display_name": "Feature",
            "score": "Criticality Score",
            "mention_count": "Mentions",
            "negative_ratio": "% Negative",
            "trend_delta": "Trend Δ",
        }
    )[["Feature", "Criticality Score", "Mentions", "% Negative", "Trend Δ"]]

c1, c2 = st.columns(2)
with c1:
    st.subheader("🔴 Top 10 Most Negative Features (Critical)")
    high_df = format_criticality(high_priority)
    if not high_df.empty:
        st.dataframe(high_df, use_container_width=True, hide_index=True)
    else:
        st.info("No critical features found.")
        
with c2:
    st.subheader("🟢 Top 10 Stable Features (Low Priority)")
    low_df = format_criticality(low_priority)
    if not low_df.empty:
        st.dataframe(low_df, use_container_width=True, hide_index=True)
    else:
        st.info("No stable features found.")

st.markdown("---")

# 2. Trends
st.header("Sentiment Trend Analysis")
features = api_client.get_features()

if not features:
    st.info("No features found — check that features_master is seeded.")
else:
    options = {feature["display_name"]: feature["feature_key"] for feature in features}
    selected_display = st.selectbox("Select a feature to view its trend:", list(options.keys()))
    feature_key = options[selected_display]

    trend = api_client.get_feature_trend(feature_key)
    if trend is None or not trend["points"]:
        st.info(f"No scoring history yet for {selected_display}.")
    else:
        points_df = pd.DataFrame(trend["points"])
        points_df["window_start"] = pd.to_datetime(points_df["window_start"])

        # Dual axis plot
        fig = go.Figure()
        
        # Add bars for mentions
        fig.add_trace(
            go.Bar(
                x=points_df["window_start"],
                y=points_df["mention_count"],
                name="Mentions",
                marker_color="rgba(31, 119, 180, 0.4)",
                yaxis="y2"
            )
        )
        
        # Add line for criticality score
        fig.add_trace(
            go.Scatter(
                x=points_df["window_start"],
                y=points_df["score"],
                mode="lines+markers",
                name="Criticality Score",
                line=dict(color="#d62728", width=3)
            )
        )
        
        fig.update_layout(
            title=f"Criticality Score and Mention Volume — {selected_display}",
            xaxis_title="Window",
            yaxis=dict(
                title="Criticality Score",
                titlefont=dict(color="#d62728"),
                tickfont=dict(color="#d62728")
            ),
            yaxis2=dict(
                title="Mentions",
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
