"""
Page 6 — Explainability
=======================

Deep-dive into SHAP vs LIME comparison for individual predictions.

Upload a patient sample, run both explainers, and compare their
feature attributions with Spearman rank correlation and overlap
analysis.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from core.config import APP_CONFIG
from core.pipeline import preprocess
from core.predictor import load_artifacts, predict
from core.validation import (
    ValidationResult,
    validate_dataframe,
    validate_file_content,
)
from explainability.shap_explainer import explain as shap_explain
from explainability.shap_explainer import extract_top_features as shap_top
from explainability.lime_explainer import explain as lime_explain
from explainability.lime_explainer import extract_top_features as lime_top
from explainability.comparison import compare_explanations
from utils.helpers import build_probability_dataframe, format_percentage

# ── Page header ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Explainability", layout="wide")
st.title("🔍 Explainability Deep-Dive")
st.write(
    "Compare SHAP and LIME explanations for individual patient predictions."
)
st.markdown("---")

# ── Load artefacts once ─────────────────────────────────────────────────────

artifacts = load_artifacts()

# ── File upload ─────────────────────────────────────────────────────────────

st.subheader("📤 Upload Patient Gene Expression Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing a single patient's gene expression values",
    type=["csv"],
    key="explain_upload",
)

# ── Helper: render validation status ────────────────────────────────────────


def _render_validation(vr: ValidationResult) -> None:
    """Display validation checklist."""
    st.markdown("---")
    st.subheader("🔍 Input Validation")

    if vr.valid:
        st.success("**✓ Dataset validated successfully**")
    else:
        st.error("**❌ Dataset validation failed**")

    checks = [
        ("✓", "CSV readable", True),
        (
            "✓" if vr.feature_count == APP_CONFIG.total_features else "❌",
            f"Expected feature structure detected ({vr.feature_count:,} features)"
            if vr.feature_count != APP_CONFIG.total_features
            else "Expected feature structure detected",
            vr.feature_count == APP_CONFIG.total_features,
        ),
        (
            "✓" if vr.invalid_value_count == 0 else "❌",
            "Numeric values verified"
            if vr.invalid_value_count == 0
            else f"Non-numeric values detected ({vr.invalid_value_count})",
            vr.invalid_value_count == 0,
        ),
        (
            "✓" if vr.missing_value_count == 0 else "❌",
            "No missing values"
            if vr.missing_value_count == 0
            else f"Missing values detected ({vr.missing_value_count:,} cells)",
            vr.missing_value_count == 0,
        ),
        (
            "✓" if vr.sample_count == 1 else "❌",
            "Single sample compatible"
            if vr.sample_count == 1
            else f"Expected 1 sample, found {vr.sample_count}",
            vr.sample_count == 1,
        ),
    ]

    for icon, label, is_ok in checks:
        st.markdown(f"&nbsp;&nbsp;{'✓' if is_ok else '❌'} {label}")

    if vr.valid:
        st.markdown("&nbsp;&nbsp;✓ **Dataset ready for analysis**")

    if vr.errors:
        with st.expander("❌ Validation Errors", expanded=True):
            for err in vr.errors:
                st.error(err)

    if vr.warnings:
        with st.expander("⚠️ Warnings"):
            for warn in vr.warnings:
                st.warning(warn)


# ── Main processing ─────────────────────────────────────────────────────────

if uploaded_file is not None:
    # ── Step 1: File readability ─────────────────────────────────────────
    raw_bytes = uploaded_file.getvalue()
    file_ok, file_msg = validate_file_content(raw_bytes)

    if not file_ok:
        st.markdown("---")
        st.subheader("🔍 Input Validation")
        st.error("**❌ Dataset validation failed**")
        st.error(file_msg)
        st.stop()

    # ── Step 2: Parse CSV ────────────────────────────────────────────────
    try:
        uploaded_file.seek(0)
        input_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.markdown("---")
        st.subheader("🔍 Input Validation")
        st.error("**❌ Dataset validation failed**")
        st.error(f"Failed to parse the CSV file: {exc}")
        st.stop()

    # ── Step 3: Validation ───────────────────────────────────────────────
    vr = validate_dataframe(input_df, check_schema=True)
    _render_validation(vr)

    if not vr.valid:
        st.stop()

    # ── Step 4: Prediction ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Prediction Result")

    col_data, col_pred = st.columns([1, 1])

    with col_data:
        st.write("Uploaded Sample (first 5 columns):")
        st.dataframe(input_df.iloc[:, :5])

    x_final = preprocess(input_df)
    probs = predict(artifacts, x_final)
    pred_class = int(np.argmax(probs))
    confidence = probs[pred_class]
    disease_name = APP_CONFIG.disease_map[pred_class]

    with col_pred:
        st.success(f"🩺 **Predicted Disease: {disease_name}**")
        st.info(f"Confidence: **{format_percentage(confidence)}**")
        prob_df = build_probability_dataframe(probs, APP_CONFIG.class_names)
        st.dataframe(prob_df, hide_index=True)

    # ── Step 5: Run both explainers (cached) ────────────────────────────
    st.markdown("---")

    @st.cache_data(show_spinner="Computing SHAP explanation…", hash_funcs={np.ndarray: lambda x: x.tobytes()})
    def _run_shap(x_bytes: bytes, pred_cls: int):
        """Run SHAP explainer (cached by input hash)."""
        x = np.frombuffer(x_bytes, dtype=np.float64).reshape(1, -1)
        shap_vals, fig = shap_explain(x, pred_cls, artifacts)
        return shap_vals, fig

    @st.cache_data(show_spinner="Computing LIME explanation…", hash_funcs={np.ndarray: lambda x: x.tobytes()})
    def _run_lime(x_bytes: bytes, pred_cls: int):
        """Run LIME explainer (cached by input hash)."""
        x = np.frombuffer(x_bytes, dtype=np.float64).reshape(1, -1)
        lime_exp, fig = lime_explain(x, pred_cls, artifacts)
        # Store as pickle-serializable data
        lime_pairs = {}
        for key in lime_exp.local_exp:
            lime_pairs[key] = lime_exp.as_list(key)
        return lime_pairs, fig, lime_exp.top_labels

    x_bytes = x_final.tobytes()

    shap_values, fig_shap = _run_shap(x_bytes, pred_class)
    lime_pairs, fig_lime, lime_top_labels = _run_lime(x_bytes, pred_class)

    # ── SHAP Explanation Section ─────────────────────────────────────────
    st.subheader("🧠 SHAP Explanation")
    st.markdown(
        "Feature contributions based on the **Shapley-value framework** "
        "from cooperative game theory.  SHAP provides globally consistent "
        "attributions — the sum of feature contributions equals the "
        "prediction difference from the expected output."
    )

    col_shap_chart, col_shap_table = st.columns([1, 1])

    with col_shap_chart:
        st.pyplot(fig_shap)

    with col_shap_table:
        shap_features = shap_top(shap_values, pred_class, top_n=10)
        shap_df = pd.DataFrame([
            {
                "Rank": f["rank"],
                "Feature": f["feature_name"],
                "Importance": f"{f['importance']:.6f}",
                "SHAP Value": f"{f['value']:+.6f}",
                "Direction": f["direction"].capitalize(),
            }
            for f in shap_features
        ])
        st.dataframe(shap_df, hide_index=True, use_container_width=True)

    # ── LIME Explanation Section ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 LIME Explanation")
    st.markdown(
        "Local surrogate model — a simple linear model fitted around "
        "this specific prediction.  LIME perturbs the input and "
        "observes output changes, then fits a weighted regression "
        "to approximate the model's local behavior."
    )

    col_lime_chart, col_lime_table = st.columns([1, 1])

    with col_lime_chart:
        st.pyplot(fig_lime)

    with col_lime_table:
        # Build a LIME-like explanation object for extract_top_features
        # Reconstruct from the cached pairs
        class FakeLimeExp:
            pass

        fake_exp = FakeLimeExp()
        fake_exp.local_exp = {k: None for k in lime_pairs.keys()}
        fake_exp.as_list = lambda k: lime_pairs.get(k, [])

        lime_features = lime_top(fake_exp, pred_class, top_n=10)
        lime_df = pd.DataFrame([
            {
                "Rank": f["rank"],
                "Feature": f["feature_name"],
                "Importance": f"{f['importance']:.6f}",
                "LIME Weight": f"{f['value']:+.6f}",
                "Direction": f["direction"].capitalize(),
            }
            for f in lime_features
        ])
        st.dataframe(lime_df, hide_index=True, use_container_width=True)

    # ── SHAP vs LIME Comparison Section ──────────────────────────────────
    st.markdown("---")
    st.subheader("⚖️ SHAP vs LIME Comparison")
    st.markdown(
        "Both methods explain the same prediction using different "
        "mathematical approaches.  This section compares their "
        "feature importance rankings."
    )

    top_n_compare = st.slider(
        "Number of top features to compare",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="compare_n",
    )

    # Run comparison
    comparison = compare_explanations(shap_features, lime_features, top_n=top_n_compare)

    # ── Metrics cards ────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Common Features",
            f"{comparison.overlap_count} / {top_n_compare}",
        )

    with m2:
        st.metric(
            "Overlap",
            f"{comparison.overlap_pct:.0f}%",
        )

    with m3:
        if comparison.spearman_rho is not None:
            st.metric(
                "Rank Correlation (ρ)",
                f"{comparison.spearman_rho:.3f}",
                help="Spearman rank correlation between SHAP and LIME rankings",
            )
        else:
            st.metric("Rank Correlation (ρ)", "N/A")

    # ── Comparison table ─────────────────────────────────────────────────
    st.markdown("#### Feature-by-Feature Comparison")
    st.dataframe(comparison.comparison_df, hide_index=True, use_container_width=True)

    # ── Interpretation ───────────────────────────────────────────────────
    st.markdown("#### Interpretation")
    st.markdown(comparison.interpretation)

    # ── Common features detail ───────────────────────────────────────────
    if comparison.common_features:
        with st.expander(f"Common Features ({comparison.overlap_count})"):
            common_df = pd.DataFrame([
                {
                    "Feature": f["feature_name"],
                    "SHAP Rank": f["shap_rank"],
                    "LIME Rank": f["lime_rank"],
                    "Rank Δ": f["rank_diff"],
                    "SHAP Dir": f["shap_direction"],
                    "LIME Dir": f["lime_direction"],
                    "Directions Match": "✓" if f["shap_direction"] == f["lime_direction"] else "✗",
                }
                for f in comparison.common_features
            ])
            st.dataframe(common_df, hide_index=True, use_container_width=True)

    # ── SHAP-only features ──────────────────────────────────────────────
    if comparison.shap_only:
        with st.expander(f"SHAP-only Features ({len(comparison.shap_only)})"):
            st.markdown(
                "These features appear in SHAP's top rankings but not LIME's.  "
                "This may indicate that SHAP captures global patterns that "
                "LIME's local surrogate model does not weight highly."
            )
            so_df = pd.DataFrame([
                {
                    "Feature": f["feature_name"],
                    "SHAP Rank": f["shap_rank"],
                    "SHAP Importance": f"{f['shap_importance']:.6f}",
                    "Direction": f["shap_direction"],
                }
                for f in comparison.shap_only
            ])
            st.dataframe(so_df, hide_index=True, use_container_width=True)

    # ── LIME-only features ──────────────────────────────────────────────
    if comparison.lime_only:
        with st.expander(f"LIME-only Features ({len(comparison.lime_only)})"):
            st.markdown(
                "These features appear in LIME's top rankings but not SHAP's.  "
                "This may indicate that LIME's local perturbation approach "
                "highlights features important in the immediate neighborhood "
                "of this prediction."
            )
            lo_df = pd.DataFrame([
                {
                    "Feature": f["feature_name"],
                    "LIME Rank": f["lime_rank"],
                    "LIME Importance": f"{f['lime_importance']:.6f}",
                    "Direction": f["lime_direction"],
                }
                for f in comparison.lime_only
            ])
            st.dataframe(lo_df, hide_index=True, use_container_width=True)

    # ── Method descriptions ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📖 About the Methods")

    col_shap_desc, col_lime_desc = st.columns(2)

    with col_shap_desc:
        st.markdown(comparison.shap_description)

    with col_lime_desc:
        st.markdown(comparison.lime_description)

    # ── Disclaimer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "**⚠️ Important:** Agreement (or disagreement) between SHAP and LIME "
        "does **not** prove that either explanation is correct.  Each method "
        "uses different mathematical assumptions: SHAP is based on game theory "
        "(globally consistent), while LIME fits a local surrogate model "
        "(locally faithful).  They may legitimately differ in their feature "
        "attributions."
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Final Year Project | Hybrid PSO–ACO + Deep Learning | "
    "Gene Expression Analysis | SHAP + LIME Explainability"
)
