"""
Page 1 — Diagnosis
==================

Upload patient gene expression data and receive disease prediction,
confidence scores, probability distribution, and biomarker insights.
Supports single-patient profiles and multi-sample donor profiles (e.g.,
regional brain tissue or paired tumor/normal samples from the predictions folder).
"""

from __future__ import annotations

import io
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st

from core.disease_models import (
    available_disease_models,
    load_disease_model,
    parse_expression_input,
    predict_samples,
)
from utils.theme import apply_theme

st.set_page_config(
    page_title="Multi-Disease Diagnosis | Patient Analysis",
    page_icon="🩺",
    layout="wide",
)

apply_theme()

st.title("🩺 Patient Gene Expression Diagnosis")
st.write(
    "Upload a patient's microarray gene expression profile to assess disease status "
    "using trained binary models calibrated for clinical relevance."
)
st.markdown("---")

# ── Load available models ───────────────────────────────────────────────────
models = available_disease_models()
if not models:
    st.error("No disease models found in `artifacts/disease_models/`. Please train them first.")
    st.stop()

model_options = {m["slug"]: m for m in models}
slug_list = list(model_options.keys())

# Auto-detect disease from uploaded file name if already in session_state
default_slug_idx = 0

# ── File Upload ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload Patient Expression Data")
uploaded_file = st.file_uploader(
    "Upload patient CSV (from `predictions/` or clinical export) or GEO Series Matrix (`.txt`, `.gz`)",
    type=["csv", "txt", "gz"],
    help="Accepts single-patient expression CSVs or multi-sample donor profiles (e.g., regional tissue profiles).",
)

if uploaded_file is not None:
    fname_lower = uploaded_file.name.lower()
    for idx, slug in enumerate(slug_list):
        if slug in fname_lower:
            default_slug_idx = idx
            break

col_sel, col_info = st.columns([2, 3])
with col_sel:
    selected_slug = st.selectbox(
        "Target Disease Model",
        options=slug_list,
        index=default_slug_idx,
        format_func=lambda s: model_options[s]["disease"],
        help="Select the disease classifier corresponding to this patient or test profile.",
    )

selected_model_info = model_options[selected_slug]
with col_info:
    st.info(
        f"**Selected Model:** {selected_model_info['disease']}  \n"
        f"**Cohort:** {selected_model_info['study']} | **Held-out Accuracy:** {selected_model_info['test_accuracy']*100:.1f}% | "
        f"**Biomarkers:** {selected_model_info['selected_biomarker_count']} probes | **Classifier:** {selected_model_info['selected_classifier']}"
    )

if uploaded_file is None:
    st.info("👆 Please upload a patient CSV or GEO Series Matrix to run diagnosis.")
    st.stop()

# ── Parse and Validate Input ────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Input Parsing & Validation")

with st.status("🧬 Analyzing Patient Expression Profile...", expanded=True) as status_box:
    st.write("📥 Ingesting and parsing expression matrix...")
    time.sleep(0.2)
    try:
        file_bytes = uploaded_file.getvalue()
        df_input, sample_labels = parse_expression_input(file_bytes, uploaded_file.name)
        model, expected_features, info = load_disease_model(selected_slug)
    except Exception as exc:
        status_box.update(label="❌ File Parsing Failed", state="error", expanded=True)
        st.error(f"❌ Failed to parse uploaded file: {exc}")
        st.stop()

    n_samples = df_input.shape[0]
    n_cols = df_input.shape[1]
    expected_count = len(expected_features)

    # Check overlap of features
    input_features = set(str(c).strip('"\'') for c in df_input.columns)
    expected_set = set(expected_features)
    shared_features = input_features.intersection(expected_set)
    overlap_pct = (len(shared_features) / expected_count) * 100

    st.write(f"🔬 Validating probe schema ({len(shared_features):,} of {expected_count:,} features matched)...")
    time.sleep(0.2)

    if overlap_pct < 50.0:
        status_box.update(label="❌ Feature Schema Mismatch", state="error", expanded=True)
        st.error(
            f"❌ Feature Schema Mismatch: The uploaded file only matches {overlap_pct:.1f}% of the required probes "
            f"for {info.get('disease_name', selected_slug)}. Please ensure you selected the correct disease model."
        )
        st.stop()

    st.write(f"⚙️ Running calibrated inference ({info.get('selected_classifier', 'Ensemble ML')})...")
    time.sleep(0.2)

    try:
        preds, probs, selected_probes = predict_samples(model, expected_features, df_input)
    except Exception as exc:
        status_box.update(label="❌ Inference Error", state="error", expanded=True)
        st.error(f"❌ Prediction error: {exc}")
        st.stop()

    status_box.update(
        label=f"✅ Diagnostic Pipeline Complete ({n_samples} sample{'s' if n_samples > 1 else ''} evaluated)",
        state="complete",
        expanded=False,
    )

v_col1, v_col2, v_col3 = st.columns(3)
with v_col1:
    st.metric("Patient Samples Detected", n_samples)
with v_col2:
    st.metric("Total Probes in Input", f"{n_cols:,}")
with v_col3:
    st.metric("Feature Schema Alignment", f"{overlap_pct:.1f}%", help=f"{len(shared_features):,} of {expected_count:,} model probes present")

st.markdown("---")
st.subheader("📋 Diagnostic Results")

disease_name = info.get("disease_name", info.get("disease", selected_slug))

if n_samples == 1:
    # Single sample view
    pred = preds[0]
    prob_disease = probs[0, 1]
    prob_control = probs[0, 0]
    confidence = max(prob_disease, prob_control)

    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        if pred == 1:
            st.error(f"### ⚠️ Diagnosis: **{disease_name} (Positive)**")
            st.write(f"The model classified this profile as positive for **{disease_name}** with **{prob_disease*100:.1f}%** probability.")
        else:
            st.success(f"### ✅ Diagnosis: **Control / Non-Diseased (Negative)**")
            st.write(f"The model classified this profile as **Control / Normal** with **{prob_control*100:.1f}%** probability.")

    with res_col2:
        st.metric("Model Confidence", f"{confidence*100:.1f}%")
        st.progress(float(prob_disease), text=f"Disease Risk Score: {prob_disease*100:.1f}%")

    # Probability Distribution Table
    prob_df = pd.DataFrame({
        "Condition": ["Study Control / Healthy", disease_name],
        "Probability": [f"{prob_control*100:.2f}%", f"{prob_disease*100:.2f}%"],
        "Score": [prob_control, prob_disease]
    })
    st.dataframe(prob_df[["Condition", "Probability"]], use_container_width=True, hide_index=True)

else:
    # Multi-sample donor/patient view (e.g. 4 regional brain samples or paired normal/tumor)
    n_positive = int(np.sum(preds == 1))
    st.info(f"📁 **Multi-Sample Profile Analysis**: {n_samples} samples evaluated ({n_positive} positive for {disease_name}, {n_samples - n_positive} control).")

    summary_rows = []
    for idx, sid in enumerate(sample_labels):
        p_lbl = "Disease (Positive)" if preds[idx] == 1 else "Control (Normal)"
        summary_rows.append({
            "Sample ID": sid,
            "Predicted Condition": p_lbl,
            "Disease Probability": f"{probs[idx, 1]*100:.1f}%",
            "Control Probability": f"{probs[idx, 0]*100:.1f}%",
            "Risk Score": probs[idx, 1],
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.drop(columns=["Risk Score"]), use_container_width=True, hide_index=True)

    # Detailed inspector for specific sample
    selected_sample_idx = st.selectbox(
        "Inspect Individual Sample Details",
        range(n_samples),
        format_func=lambda i: f"{sample_labels[i]} — {summary_rows[i]['Predicted Condition']} ({summary_rows[i]['Disease Probability']})",
    )
    st.write(f"Detailed view for **{sample_labels[selected_sample_idx]}**:")
    s_prob_dis = probs[selected_sample_idx, 1]
    st.progress(float(s_prob_dis), text=f"Disease Probability: {s_prob_dis*100:.1f}%")

# ── Biomarker Expression Table ──────────────────────────────────────────────
if selected_probes:
    with st.expander(f"🧬 Selected Informative Biomarkers ({len(selected_probes)} probes)"):
        st.write(
            f"The following probes were selected by the model's feature selection pipeline "
            f"as key differentiators for **{disease_name}**:"
        )
        bio_df = df_input.reindex(columns=selected_probes[:20]).T
        bio_df.columns = sample_labels[:len(bio_df.columns)]
        st.dataframe(bio_df, use_container_width=True)

# ── Download Report ─────────────────────────────────────────────────────────
report_df = pd.DataFrame({
    "Sample_ID": sample_labels,
    "Target_Disease": [disease_name] * n_samples,
    "Prediction": ["Disease (1)" if p == 1 else "Control (0)" for p in preds],
    "Disease_Probability": probs[:, 1],
    "Control_Probability": probs[:, 0],
})
csv_buffer = io.StringIO()
report_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="📥 Download Diagnostic Report (CSV)",
    data=csv_buffer.getvalue(),
    file_name=f"diagnostic_report_{selected_slug}.csv",
    mime="text/csv",
)
