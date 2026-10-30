"""
Page 7 — Batch Analysis
=======================

Upload multiple gene-expression CSV files, validate each one
individually, run the prediction pipeline on valid files, and
display an aggregated results table with summary statistics and
CSV export.

Invalid files are rejected gracefully without crashing the batch.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from core.batch_predictor import (
    BatchResult,
    process_batch,
    results_to_dataframe,
)
from core.config import APP_CONFIG

# ── Page header ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Batch Analysis", layout="wide")
st.title("📋 Batch Analysis")
st.write(
    "Upload multiple gene-expression CSV files for batch prediction. "
    "Each file is validated individually — one invalid file will not "
    "crash the entire batch."
)
st.markdown("---")

# ── File upload ─────────────────────────────────────────────────────────────

st.subheader("📤 Upload Patient CSV Files")

uploaded_files = st.file_uploader(
    "Select one or more CSV files (one patient per file)",
    type=["csv"],
    accept_multiple_files=True,
    key="batch_upload",
)

# ── Process batch ───────────────────────────────────────────────────────────

if uploaded_files:
    # Collect raw bytes keyed by filename (no persistence)
    files_dict: dict[str, bytes] = {}
    for uf in uploaded_files:
        files_dict[uf.name] = uf.getvalue()

    total = len(files_dict)
    st.info(f"**{total}** file(s) uploaded. Processing…")

    # Progress bar
    progress_bar = st.progress(0, text="Starting…")
    status_text = st.empty()

    def _update_progress(processed: int, total_files: int) -> None:
        pct = processed / total_files
        progress_bar.progress(pct, text=f"Processed {processed}/{total_files}")

    result: BatchResult = process_batch(
        files_dict, progress_callback=_update_progress,
    )

    progress_bar.progress(1.0, text="Done.")
    status_text.empty()

    st.markdown("---")

    # ── Validation summary ────────────────────────────────────────────
    st.subheader("🔍 Validation Summary")

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Total Files", result.total_files)
    vc2.metric("Valid Files", result.valid_files)
    vc3.metric("Invalid Files", result.invalid_files)
    vc4.metric("Successfully Analyzed", result.successfully_analyzed)

    # Per-file validation status
    with st.expander("Per-file validation details", expanded=False):
        for s in result.samples:
            if s.valid:
                st.markdown(f"✅ **{s.filename}**")
            else:
                st.markdown(f"❌ **{s.filename}**")
                st.caption(f"Reason: {s.error_message}")

    # ── Results table ─────────────────────────────────────────────────
    valid_samples = [s for s in result.samples if s.valid and s.probabilities is not None]

    if valid_samples:
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # Build results DataFrame
        rows = []
        for s in valid_samples:
            probs = s.probabilities
            rows.append({
                "Sample": s.filename,
                "Prediction": s.prediction,
                "Confidence": f"{s.confidence * 100:.1f}%",
                "CAD": f"{probs[0]:.4f}",
                "Autoimmune": f"{probs[1]:.4f}",
                "Alzheimer's": f"{probs[2]:.4f}",
                "Parkinson's": f"{probs[3]:.4f}",
            })

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # ── Result summary ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Result Summary")

        # Disease distribution
        predictions = [s.prediction for s in valid_samples]
        disease_counts = {name: predictions.count(name) for name in APP_CONFIG.class_names}

        st.markdown("**Disease Distribution:**")
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("CAD", disease_counts.get("Coronary Artery Disease", 0))
        dc2.metric("Autoimmune", disease_counts.get("Autoimmune Disease", 0))
        dc3.metric("Alzheimer's", disease_counts.get("Alzheimer's Disease", 0))
        dc4.metric("Parkinson's", disease_counts.get("Parkinson's Disease", 0))

        # Low confidence count
        st.markdown("---")
        lc1, lc2 = st.columns(2)
        lc1.metric(
            "Low-Confidence Predictions",
            result.low_confidence_count,
            help="Samples where max probability < "
                 f"{APP_CONFIG.low_confidence_threshold:.0%}",
        )
        lc2.metric(
            "High-Confidence Predictions",
            result.successfully_analyzed - result.low_confidence_count,
        )

        if result.low_confidence_count > 0:
            st.warning(
                f"**{result.low_confidence_count}** sample(s) were flagged as "
                "low-confidence.  The model does not strongly distinguish "
                "between disease classes for these samples."
            )

        # ── CSV export ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📥 Export Results")

        export_df = results_to_dataframe(result)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
        )

    elif result.invalid_files == result.total_files:
        st.error(
            "All uploaded files failed validation.  "
            "Please check the file format and try again."
        )

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Final Year Project | Hybrid PSO–ACO + Deep Learning | "
    "Gene Expression Analysis"
)
