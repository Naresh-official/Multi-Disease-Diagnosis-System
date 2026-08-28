"""
Explainable Multi-Disease Diagnosis System
==========================================

Streamlit entry point.  Uses Streamlit's native multi-page support:
each file under ``pages/`` automatically appears in the sidebar nav.

To run::

    streamlit run app.py
"""

import streamlit as st

from core.config import APP_CONFIG

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Disease Diagnosis",
    page_icon="🧬",
    layout="centered",
)

# ── Landing page content ────────────────────────────────────────────────────
st.title("🧬 Explainable Multi-Disease Diagnosis System")
st.write(
    "Gene Expression–Based Disease Prediction using "
    "Hybrid Swarm Intelligence (PSO + ACO) and Deep Learning"
)

st.markdown("---")

st.subheader("📋 Project Overview")

st.markdown("""
This application predicts **four diseases** from high-dimensional
gene-expression microarrays using a hybrid PSO + ACO feature
selection pipeline and a deep neural network classifier.

**Disease Classes:**
| Class | Disease | GEO Dataset |
|:-----:|:--------|:------------|
| 0 | Coronary Artery Disease | GSE66360 |
| 1 | Autoimmune Disease | GSE50772 |
| 2 | Alzheimer's Disease | GSE48350 |
| 3 | Parkinson's Disease | GSE6613 |

**Pipeline:**  
Raw CSV → StandardScaler → Variance Threshold → PSO → ACO → DNN (80 features) → 4-class softmax

**Navigate** the sidebar to explore each section of the system.
""")

st.markdown("---")

st.subheader("🚀 Quick Start")
st.markdown("""
1. Click **Diagnosis** in the sidebar.
2. Upload a single-patient gene-expression CSV.
3. View the predicted disease, confidence, probability table,
   and SHAP / LIME explanations.
""")

st.markdown("---")
st.caption(
    "Final Year Project | Hybrid PSO–ACO + Deep Learning | Gene Expression Analysis"
)
