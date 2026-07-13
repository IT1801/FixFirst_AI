import sys
from pathlib import Path
import io

import pandas as pd
import streamlit as st

# Ensure we can import from src
src_path = str(Path(__file__).resolve().parents[3])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fixfirst.dashboard.services import inference_service, analysis_service
from fixfirst.dashboard.components import charts

st.set_page_config(page_title="FixFirst AI - Home", page_icon="🚀", layout="wide")

st.sidebar.title("🛠️ FixFirst AI")
st.sidebar.caption("Automated Feature Prioritization")

st.title("🚀 FixFirst AI")
st.subheader("Automated Feature Prioritization from Mobile App Reviews using Aspect-Based Sentiment Analysis")

st.markdown("---")

# Session state initialization
if "aspects_df" not in st.session_state:
    st.session_state.aspects_df = pd.DataFrame()
if "priority_df" not in st.session_state:
    st.session_state.priority_df = pd.DataFrame()
if "summary_kpis" not in st.session_state:
    st.session_state.summary_kpis = {}

st.header("1. Submit Reviews")
input_method = st.radio("Choose Input Method:", ["Text Area", "CSV Upload"], horizontal=True)

reviews_df = pd.DataFrame()

if input_method == "Text Area":
    text_input = st.text_area("Paste one review per line:", placeholder="The app crashes after every update.\nThe UI looks beautiful.\nPremium subscription is too expensive.\nNotifications arrive very late.", height=150)
    if text_input:
        reviews = [r.strip() for r in text_input.split('\n') if r.strip()]
        reviews_df = pd.DataFrame({"review": reviews})
else:
    uploaded_file = st.file_uploader("Upload CSV (must contain a 'review' column)", type=["csv"])
    if uploaded_file is not None:
        try:
            # We can use pd.read_csv to read the uploaded bytes
            reviews_df = pd.read_csv(uploaded_file)
            if "review" not in reviews_df.columns:
                st.error("CSV must contain a column named 'review'")
                reviews_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if st.button("Analyze Reviews", type="primary"):
    if reviews_df.empty:
        st.warning("Please provide some reviews to analyze.")
    else:
        with st.container():
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("Initializing models..."):
                # Run inference
                aspects_df = inference_service.run_dashboard_inference(reviews_df, progress_bar, status_text)
                
                # Compute priorities and summary
                priority_df = analysis_service.compute_feature_priorities(aspects_df)
                summary = analysis_service.summarize_results(aspects_df, priority_df)
                
                st.session_state.aspects_df = aspects_df
                st.session_state.priority_df = priority_df
                st.session_state.summary_kpis = summary
                
            progress_bar.empty()
            status_text.empty()
            st.success(f"Successfully processed {len(reviews_df)} reviews.")

st.markdown("---")

if not st.session_state.aspects_df.empty:
    st.header("2. Results Summary")
    
    kpi = st.session_state.summary_kpis
    
    cols = st.columns(5)
    cols[0].metric("Reviews Processed", len(st.session_state.aspects_df["Original Review"].unique()))
    cols[1].metric("Total Aspects Detected", kpi.get("total_aspects", 0))
    cols[2].metric("Positive %", f"{kpi.get('positive_pct', 0):.1f}%")
    cols[3].metric("Negative %", f"{kpi.get('negative_pct', 0):.1f}%")
    cols[4].metric("Avg Aspects / Review", f"{kpi.get('average_aspects', 0):.1f}")
    
    st.markdown("---")
    
    st.header("3. Product Manager Summary")
    st.info(kpi.get("summary_text", ""))
    
    st.markdown("---")
    
    st.header("4. Feature Prioritization Table")
    st.dataframe(
        st.session_state.priority_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    st.header("5. Visual Analytics")
    c1, c2, c3 = st.columns(3)
    with c1:
        charts.render_feature_distribution(st.session_state.priority_df)
    with c2:
        charts.render_sentiment_distribution(st.session_state.aspects_df)
    with c3:
        charts.render_priority_chart(st.session_state.priority_df)
        
    st.markdown("---")
    
    st.header("6. Review Explorer")
    st.caption("Select a row in the table below to see more details.")
    
    event = st.dataframe(
        st.session_state.aspects_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    selected_rows = event.selection.rows
    if selected_rows:
        row_idx = selected_rows[0]
        row_data = st.session_state.aspects_df.iloc[row_idx]
        
        st.subheader("Review Details")
        with st.container(border=True):
            st.markdown(f"**Original Review:**\n> {row_data['Original Review']}")
            st.markdown(f"**Detected Aspect:** {row_data['Detected Aspect']}")
            
            color = {"negative": "red", "neutral": "gray", "positive": "green"}.get(row_data["Sentiment"], "gray")
            st.markdown(f"**Predicted Sentiment:** :{color}[{row_data['Sentiment']}]")
            st.markdown(f"**Prediction Confidence:** {row_data['Confidence']:.3f}")
