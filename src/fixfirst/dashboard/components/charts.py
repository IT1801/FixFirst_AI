"""Reusable chart components using Plotly."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

def render_feature_distribution(priority_df: pd.DataFrame) -> str:
    """Horizontal bar chart for feature mentions."""
    if priority_df.empty:
        return ""
        
    df = priority_df.sort_values(by="Mentions", ascending=True)
    fig = px.bar(
        df, 
        x="Mentions", 
        y="Feature", 
        orientation='h',
        title="Feature Distribution (Mentions)",
        color="Mentions",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=350)
    return fig.to_json()

def render_sentiment_distribution(aspects_df: pd.DataFrame) -> str:
    """Donut chart for sentiment distribution."""
    if aspects_df.empty:
        return ""
        
    counts = aspects_df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    
    color_map = {"positive": "#28a745", "neutral": "#6c757d", "negative": "#dc3545"}
    
    fig = px.pie(
        counts, 
        values="Count", 
        names="Sentiment", 
        hole=0.4,
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=color_map
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350)
    return fig.to_json()

def render_priority_chart(priority_df: pd.DataFrame) -> str:
    """Bar chart for top negative features."""
    if priority_df.empty or priority_df["Negative"].sum() == 0:
        return ""
        
    df = priority_df[priority_df["Negative"] > 0].sort_values(by="Negative", ascending=False).head(10)
    
    fig = px.bar(
        df,
        x="Feature",
        y="Negative",
        title="Top Features Needing Attention",
        color="Negative",
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=350)
    return fig.to_json()
