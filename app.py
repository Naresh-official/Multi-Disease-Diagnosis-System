"""
Explainable Multi-Disease Diagnosis System
==========================================

Streamlit entry point for disease-specific gene-expression prediction models.
"""

from __future__ import annotations

import streamlit as st

from core.disease_models import available_disease_models
from utils.theme import apply_theme

st.set_page_config(
    page_title="Multi-Disease Diagnosis | Overview",
    page_icon="🧬",
    layout="wide",
)

apply_theme()

st.title("🧬 Multi-Disease Diagnosis System")
st.markdown(
    "**Gene Expression Decision Support & Biomarker Discovery Platform**  \n"
    "Automated microarray profiling, machine learning classification, and biomarker "
    "signature analysis across curated NCBI Gene Expression Omnibus (GEO) cohorts."
)

st.markdown("---")

models = available_disease_models()

CLINICAL_PROFILES: dict[str, dict[str, str]] = {
    "cad": {
        "icon": "🫀",
        "target": "Acute myocardial infarction & coronary stenosis",
        "tissue": "Peripheral Blood Mononuclear Cells (PBMC)",
        "summary": "Transcriptional profiling for circulating ischemic and vascular inflammatory markers.",
    },
    "alzheimers": {
        "icon": "🧠",
        "target": "Neurodegenerative dementia & amyloid burden",
        "tissue": "Post-Mortem Brain Cortex & Hippocampus",
        "summary": "Transcriptional signatures tracking synaptic loss and neuroinflammation in cognitive decline.",
    },
    "asthma": {
        "icon": "🫁",
        "target": "Chronic airway inflammatory phenotyping",
        "tissue": "Induced Sputum & Epithelial Cells",
        "summary": "Differential airway mucosal gene expression distinguishing asthmatic from healthy cohorts.",
    },
    "heart_failure": {
        "icon": "💓",
        "target": "Dilated and ischemic cardiomyopathy",
        "tissue": "Left Ventricular Myocardium",
        "summary": "Myocardial transcriptional profiling distinguishing end-stage heart failure from donor controls.",
    },
    "lung_adenocarcinoma": {
        "icon": "🔬",
        "target": "Non-small cell lung tumor tissue vs. non-malignant tissue",
        "tissue": "Pulmonary Resection Tissue",
        "summary": "Molecular tumor discrimination comparing primary adenocarcinomas with paired adjacent parenchyma.",
    },
}

if not models:
    st.error("Disease models are not available yet. Run `python3 train_disease_models.py` to create them.")
else:
    st.subheader("🩺 Supported Clinical Disease Modules")
    st.markdown(
        '<div style="margin-bottom: 12px;"><span class="status-dot"></span><span style="color:#14b8a6; font-weight:600; font-size:0.875rem;">5 Clinical Modules Online</span>'
        '<span style="color:#94a3b8; font-size:0.875rem;"> · Calibrated on disease-specific NCBI GEO cohorts with group-aware patient isolation</span></div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(min(3, len(models)))
    for index, item in enumerate(models):
        slug = item.get("slug", "")
        meta = CLINICAL_PROFILES.get(
            slug,
            {
                "icon": "🧬",
                "target": "Disease discrimination",
                "tissue": "Clinical Biospecimen",
                "summary": "Gene expression signature analysis.",
            },
        )
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                    f'<span style="font-size:1.05rem; font-weight:600; color:#f8fafc;">{meta["icon"]} {item["disease"]}</span>'
                    f'<span style="font-size:0.72rem; color:#14b8a6; background:rgba(20,184,166,0.12); padding:2px 8px; border-radius:12px; border:1px solid rgba(20,184,166,0.25);"><span class="status-dot"></span>Active</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"**Target:** {meta['target']}")
                st.markdown(
                    f"- **Cohort Accession:** `{item['study']}`  \n"
                    f"- **Specimen Type:** {meta['tissue']}  \n"
                    f"- **Biomarker Probes:** {item['selected_biomarker_count']} selected features  \n"
                    f"- **Classifier:** {item['selected_classifier']}"
                )
                st.markdown(f"*{meta['summary']}*")

    st.caption("💡 Full quantitative benchmark metrics (Accuracy, F1, Confusion Matrices, and ROC curves) are detailed on the **Model Performance** page.")

st.markdown("---")

st.subheader("🔄 Diagnostic & Analysis Workflow")
wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)
with wf_col1:
    with st.container(border=True):
        st.markdown("#### 1. Data Ingestion")
        st.caption("Upload patient CSV expression profiles or NCBI GEO Series Matrix archives (`.txt`, `.gz`).")
with wf_col2:
    with st.container(border=True):
        st.markdown("#### 2. Probe Alignment")
        st.caption("Automated schema validation against platform probe sets with group-aware scaling.")
with wf_col3:
    with st.container(border=True):
        st.markdown("#### 3. Biomarker Selection")
        st.caption("High-variance feature filtering isolates validated disease-discriminative probes.")
with wf_col4:
    with st.container(border=True):
        st.markdown("#### 4. Diagnostic Inference")
        st.caption("Calibrated probability scoring, binary condition classification, and risk reporting.")

st.markdown("---")

left_col, right_col = st.columns(2)
with left_col:
    with st.container(border=True):
        st.subheader("🚀 Patient Expression Diagnosis")
        st.markdown("""
        Assess individual patient profiles or multi-sample donor cohorts:
        - Upload expression data or test with curated sample profiles.
        - Automatic disease model matching by cohort metadata.
        - Binary prediction, class probability breakdown, and biomarker visualization.
        - Exportable clinical diagnostic summary CSV.
        """)
        st.page_link(
            "pages/1_Diagnosis.py",
            label="Launch Patient Diagnosis",
            icon="🩺",
        )

with right_col:
    with st.container(border=True):
        st.subheader("📊 Validation & Benchmark Metrics")
        st.markdown("""
        Explore held-out evaluation benchmarks across all 5 disease models:
        - Strict 70% train / 30% test group-isolated split metrics.
        - Comparative accuracy and macro F1 cross-disease evaluations.
        - Confusion matrices and per-class precision / recall / support.
        - Prioritized probe-level biomarker rankings.
        """)
        st.page_link(
            "pages/2_Model_Performance.py",
            label="Inspect Model Performance & Metrics",
            icon="📊",
        )

st.markdown("---")
st.warning(
    "⚠️ **Research Prototype Notice**: This system is designed for clinical decision-support research. "
    "Models are evaluated on held-out study-internal splits with group-aware patient isolation. "
    "Outputs do not constitute standalone clinical diagnoses."
)
