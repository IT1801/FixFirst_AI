"""Service for computing analysis metrics and generating executive summaries."""

import pandas as pd
from typing import Dict, Tuple

def compute_feature_priorities(aspects_df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature priority based on mentions and sentiment."""
    if aspects_df.empty:
        return pd.DataFrame(columns=["Feature", "Mentions", "Positive", "Negative", "Priority"])
    
    # Group by feature
    grouped = aspects_df.groupby("Detected Aspect")
    
    data = []
    for feature, group in grouped:
        total_mentions = len(group)
        sentiment_counts = group["Sentiment"].value_counts()
        positive = sentiment_counts.get("positive", 0)
        negative = sentiment_counts.get("negative", 0)
        
        # Priority Logic
        # 🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low
        neg_ratio = negative / total_mentions if total_mentions > 0 else 0
        
        if negative >= 5 and neg_ratio >= 0.5:
            priority = "🔴 Critical"
        elif negative >= 3 and neg_ratio >= 0.3:
            priority = "🟠 High"
        elif negative >= 1:
            priority = "🟡 Medium"
        else:
            priority = "🟢 Low"
            
        data.append({
            "Feature": feature,
            "Mentions": total_mentions,
            "Positive": positive,
            "Negative": negative,
            "Priority": priority,
            "_neg_count": negative, # for sorting
            "_mentions": total_mentions # for sorting
        })
        
    df = pd.DataFrame(data)
    # Sort by negative count desc, then mentions desc
    df = df.sort_values(by=["_neg_count", "_mentions"], ascending=[False, False])
    return df.drop(columns=["_neg_count", "_mentions"])

def summarize_results(aspects_df: pd.DataFrame, priority_df: pd.DataFrame) -> Dict:
    """Calculate KPIs and generate an executive summary."""
    if aspects_df.empty:
        return {
            "total_aspects": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "average_aspects": 0.0,
            "summary_text": "No aspects detected. Please provide more reviews."
        }
        
    total_aspects = len(aspects_df)
    sentiment_counts = aspects_df["Sentiment"].value_counts()
    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    
    num_unique_reviews = aspects_df["Original Review"].nunique()
    
    # Summary generation
    top_negative = priority_df.iloc[0] if not priority_df.empty and priority_df.iloc[0]["Negative"] > 0 else None
    top_positive = priority_df.sort_values(by="Positive", ascending=False).iloc[0] if not priority_df.empty and priority_df["Positive"].sum() > 0 else None
    
    insights = []
    recommendation = ""
    
    if top_negative is not None:
        insights.append(f"• **{top_negative['Feature']}** is the most complained-about feature ({top_negative['Negative']} negative mentions).")
        recommendation = f"Prioritize fixing issues related to **{top_negative['Feature']}** before introducing new features."
    else:
        insights.append("• No major negative complaints detected.")
        recommendation = "Maintain current stability and monitor for new issues."
        
    if top_positive is not None:
        insights.append(f"• Users highly appreciate **{top_positive['Feature']}** ({top_positive['Positive']} positive mentions).")
        
    # Formatting
    summary_md = "### Top Insights\n" + "\n".join(insights) + "\n\n### Recommendation\n" + recommendation
    
    return {
        "total_aspects": total_aspects,
        "positive_pct": (pos / total_aspects) * 100 if total_aspects else 0,
        "negative_pct": (neg / total_aspects) * 100 if total_aspects else 0,
        "average_aspects": total_aspects / num_unique_reviews if num_unique_reviews else 0,
        "summary_text": summary_md
    }
