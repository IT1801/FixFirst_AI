import streamlit as st

st.set_page_config(page_title="FixFirst AI - Model Performance", page_icon="🤖", layout="wide")

st.sidebar.title("🛠️ FixFirst AI")
st.sidebar.caption("Automated Feature Prioritization")

st.title("🤖 Model Performance")
st.subheader("Latest evaluation metrics and architecture details")

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.header("Evaluation Metrics")
    
    st.subheader("Aspect Detection")
    st.metric("Micro F1 Score", "0.378")
    
    st.subheader("Aspect Sentiment")
    st.metric("Accuracy", "77.1%")
    
    st.subheader("Batch Inference Performance")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reviews", "2,500")
    col_b.metric("Aspects", "5,269")
    col_c.metric("LLM Fallback", "0%")
    
    st.info("The models successfully handle 100% of the extraction workload, requiring 0% LLM fallback, leading to highly efficient batch inference.")

with c2:
    st.header("Model Architecture")
    
    with st.container(border=True):
        st.markdown("""
        ### Base Model
        **Microsoft DeBERTa-v3-base**  
        State-of-the-art encoder language model known for its disentangled attention mechanism, making it highly effective for natural language understanding tasks.
        
        ### Fine-Tuning Strategy
        **LoRA (Low-Rank Adaptation / PEFT)**  
        Instead of full fine-tuning, we use LoRA to inject trainable rank decomposition matrices into the transformer architecture. This drastically reduces the number of trainable parameters while maintaining performance.
        
        ### Task 1: Aspect Detection
        **Multi-label Classification**  
        The base model with an Aspect Category LoRA adapter is used to detect presence of multiple aspects simultaneously using a dynamically tuned threshold per class.
        
        ### Task 2: Sentiment Analysis
        **Aspect-level Classification**  
        The base model with a Sentiment LoRA adapter is used to classify the sentiment (Positive, Negative, Neutral) specifically towards the detected aspect, not just the overall sentence sentiment.
        """)

st.markdown("---")
st.caption("Metrics derived from gold dataset evaluation. Inference stats based on the latest batch run on testing datasets.")
