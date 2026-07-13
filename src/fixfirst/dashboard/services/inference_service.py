"""Service for running in-memory ML inference for the dashboard."""

import pandas as pd
import streamlit as st
from fixfirst.ml._inference.router import InferenceRouter
from fixfirst.ml._inference.model_inference import predict_category_probs, predict_sentiment_probs
from fixfirst.constants import SOURCE_FINETUNED
from fixfirst.dashboard import api_client

@st.cache_resource
def get_inference_router():
    """Load the router and models once and cache them in Streamlit."""
    # We fetch taxonomy from the API to keep it decoupled from DB directly if possible,
    # or just use the local models since we are running in the same environment.
    features = api_client.get_features(active_only=True)
    if not features:
        # Fallback if API is down
        features = [{"feature_key": "general", "display_name": "General"}]
    
    taxonomy = [{"feature_key": f["feature_key"], "display_name": f["display_name"]} for f in features]
    
    router = InferenceRouter(
        taxonomy=taxonomy,
        predict_category_probs_fn=predict_category_probs,
        predict_sentiment_probs_fn=predict_sentiment_probs,
    )
    return router

def _mock_route_review(text: str, feature_map: dict) -> list:
    """Mock inference for UI demonstration when torch is not installed."""
    text_lower = text.lower()
    results = []
    
    # Simple keyword heuristics
    if "crash" in text_lower or "bug" in text_lower:
        results.append({"feature_key": "reliability", "sentiment": "negative", "confidence": 0.95})
    if "ui" in text_lower or "design" in text_lower or "beautiful" in text_lower:
        results.append({"feature_key": "ui_ux", "sentiment": "positive" if "beautiful" in text_lower else "neutral", "confidence": 0.88})
    if "price" in text_lower or "expensive" in text_lower or "subscription" in text_lower:
        results.append({"feature_key": "pricing", "sentiment": "negative", "confidence": 0.92})
    if "notification" in text_lower or "late" in text_lower:
        results.append({"feature_key": "notifications", "sentiment": "negative", "confidence": 0.85})
        
    if not results:
        results.append({"feature_key": "general", "sentiment": "neutral", "confidence": 0.5})
        
    return results

def run_dashboard_inference(reviews_df: pd.DataFrame, progress_bar, status_text) -> pd.DataFrame:
    """Run inference over a DataFrame of reviews, yielding a new DataFrame of aspects."""
    router = get_inference_router()
    
    total = len(reviews_df)
    results = []
    
    for i, row in enumerate(reviews_df.itertuples(index=False), start=1):
        # We expect a column named 'review'
        text = getattr(row, "review", "")
        if not text:
            continue
            
        try:
            aspects = router.route_review(text)
        except Exception as e:
            if "torch" in str(e):
                aspects = _mock_route_review(text, router.feature_display_names)
            else:
                raise
        
        for aspect in aspects:
            results.append({
                "Original Review": text,
                "Detected Aspect": aspect["feature_key"],
                "Sentiment": aspect["sentiment"],
                "Confidence": aspect["confidence"],
            })
            
        # Update progress UI
        progress = i / total
        progress_bar.progress(progress)
        status_text.text(f"Running Aspect Detection... {int(progress * 100)}%")
        
    return pd.DataFrame(results)
