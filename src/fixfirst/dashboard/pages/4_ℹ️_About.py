import streamlit as st

st.set_page_config(page_title="FixFirst AI - About", page_icon="ℹ️", layout="wide")

st.sidebar.title("🛠️ FixFirst AI")
st.sidebar.caption("Automated Feature Prioritization")

st.title("ℹ️ About FixFirst AI")
st.subheader("System Architecture and Pipeline")

st.markdown("---")

c1, c2 = st.columns([1, 1])

with c1:
    st.header("Pipeline Architecture")
    
    st.markdown("""
    ```mermaid
    graph TD
        A[Mobile App Reviews] --> B[Preprocessing & Formatting]
        B --> C[Aspect Detection Model]
        C --> D[Aspect Sentiment Model]
        
        subgraph Machine Learning Pipeline
        C
        D
        end
        
        D --> E[(PostgreSQL Database)]
        E --> F[API Layer FastAPI]
        F --> G[Analytics Dashboard]
        
        classDef primary fill:#1f77b4,stroke:#fff,stroke-width:2px,color:#fff;
        classDef db fill:#2ca02c,stroke:#fff,stroke-width:2px,color:#fff;
        classDef ui fill:#ff7f0e,stroke:#fff,stroke-width:2px,color:#fff;
        
        class A,B,C,D primary;
        class E db;
        class F,G ui;
    ```
    """)

with c2:
    st.header("How It Works")
    
    st.markdown("""
    **FixFirst AI** is an end-to-end Aspect-Based Sentiment Analysis (ABSA) system designed specifically for Product Managers and Engineering leads. 
    
    Instead of manually reading through thousands of mobile app reviews to figure out what's broken, FixFirst AI automatically extracts the specific features users are talking about and classifies their sentiment towards those features.
    
    ### 1. Ingestion & Preprocessing
    Raw reviews are ingested (e.g., via CSV upload or API) and formatted for the machine learning pipeline. 
    
    ### 2. Aspect Detection
    The first step of the ML pipeline identifies *what* the user is talking about. A fine-tuned DeBERTa-v3 model acts as a multi-label classifier, detecting the presence of predefined features (like 'Login', 'UI', 'Pricing', or 'Notifications').
    
    ### 3. Aspect Sentiment
    Once features are detected, the second model evaluates *how* the user feels about each specific feature. This allows the system to distinguish between cases where a user loves the UI but hates the pricing in the same sentence.
    
    ### 4. Prioritization
    Finally, the dashboard computes a priority score based on the frequency of mentions and the ratio of negative sentiment, highlighting exactly what the engineering team should fix first.
    """)
