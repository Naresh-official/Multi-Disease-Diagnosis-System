"""
Page 1 — Diagnosis
==================

Upload a patient gene-expression CSV and receive a predicted disease,
confidence score, probability distribution, SHAP explanation, and
LIME explanation.

This page reproduces the **exact** behaviour of the original single-file
``app.py``, but delegates all heavy lifting to the ``core`` and
``explainability`` packages.
"""

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
from explainability.lime_explainer import explain as lime_explain
from utils.helpers import build_probability_dataframe, format_percentage
from utils.uncertainty import (
    build_uncertainty_result,
    calculate_calibration_metrics,
    ConfidenceLevel,
)
from reports.pdf_report import (
    SampleInfo,
    PredictionResult,
    generate_diagnosis_report,
)

# ── Page header ─────────────────────────────────────────────────────────────
st.title("🧬 Multi-Disease Diagnosis System")
st.write(
    "Gene Expression–Based Disease Prediction using "
    "Hybrid Swarm Intelligence (PSO + ACO) and Deep Learning"
)
st.markdown("---")

# ── Load artefacts once ─────────────────────────────────────────────────────
artifacts = load_artifacts()

# ── File upload ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload Patient Gene Expression Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing a single patient's gene expression values",
    type=["csv"],
)


# ── Helper: render validation status ────────────────────────────────────────
def _render_validation_result(vr: ValidationResult) -> None:
    """Display the validation checklist in the Streamlit UI."""

    st.markdown("---")
    st.subheader("🔍 Input Validation")

    if vr.valid:
        st.success("**✓ Dataset validated successfully**")
    else:
        st.error("**❌ Dataset validation failed**")

    # Checklist items
    checks = [
        ("✓", "CSV readable", True),
        (
            "✓" if vr.feature_count == APP_CONFIG.total_features else "❌",
            f"Expected feature structure detected ({vr.feature_count:,} of {APP_CONFIG.total_features:,} features)"
            if vr.feature_count != APP_CONFIG.total_features
            else "Expected feature structure detected",
            vr.feature_count == APP_CONFIG.total_features,
        ),
        (
            "✓" if vr.invalid_value_count == 0 else "❌",
            "Numeric values verified"
            if vr.invalid_value_count == 0
            else f"Non-numeric / invalid values detected ({vr.invalid_value_count})",
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
            "✓" if vr.invalid_value_count == 0 else "❌",
            "No infinite values"
            if vr.invalid_value_count == 0
            else f"Infinite values detected ({vr.invalid_value_count})",
            vr.invalid_value_count == 0,
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
        if is_ok:
            st.markdown(f"&nbsp;&nbsp;{icon} {label}")
        else:
            st.markdown(f"&nbsp;&nbsp;{icon} {label}")

    if vr.valid:
        st.markdown("&nbsp;&nbsp;✓ **Dataset ready for analysis**")

    # Show errors
    if vr.errors:
        with st.expander("❌ Validation Errors", expanded=True):
            for err in vr.errors:
                st.error(err)

    # Show warnings
    if vr.warnings:
        with st.expander("⚠️ Warnings"):
            for warn in vr.warnings:
                st.warning(warn)


# ── Prediction & explanation ────────────────────────────────────────────────
if uploaded_file is not None:
    # ── Step 1: File readability check ──────────────────────────────────
    raw_bytes = uploaded_file.getvalue()
    file_ok, file_msg = validate_file_content(raw_bytes)

    if not file_ok:
        st.markdown("---")
        st.subheader("🔍 Input Validation")
        st.error("**❌ Dataset validation failed**")
        st.error(file_msg)
        st.stop()

    # ── Step 2: Parse CSV ───────────────────────────────────────────────
    try:
        uploaded_file.seek(0)
        input_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.markdown("---")
        st.subheader("🔍 Input Validation")
        st.error("**❌ Dataset validation failed**")
        st.error(f"Failed to parse the CSV file: {exc}")
        st.stop()

    # ── Step 3: Full validation ─────────────────────────────────────────
    vr = validate_dataframe(input_df, check_schema=True)
    _render_validation_result(vr)

    if not vr.valid:
        st.stop()

    # ── Step 4: Prediction pipeline ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Prediction Result")

    st.write("Uploaded Sample:")
    st.dataframe(input_df)

    # ── Preprocessing ───────────────────────────────────────────────
    x_final = preprocess(input_df)

    # ── Model prediction ────────────────────────────────────────────
    probs = predict(artifacts, x_final)
    pred_class = int(np.argmax(probs))
    confidence = probs[pred_class]

    disease_name = APP_CONFIG.disease_map[pred_class]

    # ── Uncertainty analysis ────────────────────────────────────────
    uncertainty = build_uncertainty_result(probs)

    st.success(
        f"🩺 **The model predicts {disease_name} as the "
        f"highest-probability class.**"
    )

    # Confidence metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model Probability",
        format_percentage(uncertainty.max_probability),
    )
    c2.metric(
        "Confidence Level",
        uncertainty.confidence_level.value,
    )
    c3.metric(
        "Prediction Entropy",
        f"{uncertainty.entropy:.3f} bits",
        help="Shannon entropy of the probability distribution. "
             "Lower = more certain.",
    )
    c4.metric(
        "Normalised Entropy",
        f"{uncertainty.normalised_entropy:.3f}",
        help="Entropy scaled to [0, 1]. 0 = certain, 1 = uniform.",
    )

    # Low-confidence warning
    if uncertainty.is_low_confidence:
        st.warning(uncertainty.warning_message)
    elif not uncertainty.probability_valid:
        st.warning(uncertainty.warning_message)

    # ── Probability table ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Disease Probability Distribution")

    prob_df = build_probability_dataframe(probs, APP_CONFIG.class_names)
    st.dataframe(prob_df)

    # ── SHAP explanation ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 SHAP Explanation (Why this disease?)")

    shap_values, fig_shap = shap_explain(x_final, pred_class, artifacts)
    st.pyplot(fig_shap)

    # ── LIME explanation ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 LIME Explanation (Local Interpretability)")

    _, fig_lime = lime_explain(x_final, pred_class, artifacts)
    st.pyplot(fig_lime)

    # ── PDF Report Generation ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Generate PDF Report")
    st.write(
        "Download a comprehensive PDF report containing the prediction, "
        "probability distribution, SHAP/LIME explanations, top biomarkers, "
        "and uncertainty analysis."
    )

    if st.button("📥 Generate PDF Report", key="generate_pdf"):
        with st.spinner("Generating PDF report…"):
            sample_info = SampleInfo(
                sample_id=uploaded_file.name.replace(".csv", ""),
                n_input_features=APP_CONFIG.total_features,
                n_selected_features=APP_CONFIG.selected_features,
            )
            pred_result = PredictionResult(
                predicted_disease=disease_name,
                predicted_class=pred_class,
                probabilities=probs,
                class_names=APP_CONFIG.class_names,
            )

            pdf_bytes = generate_diagnosis_report(
                sample_info=sample_info,
                prediction=pred_result,
                probabilities=probs,
                shap_image=fig_shap,
                lime_image=fig_lime,
                shap_values=shap_values,
                uncertainty=uncertainty,
            )

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"diagnosis_report_{sample_info.sample_id}.pdf",
                mime="application/pdf",
            )
            st.success("PDF report generated successfully.")

    # ── Calibration analysis (if available) ──────────────────────────
    cal = calculate_calibration_metrics()
    if cal is not None:
        st.markdown("---")
        st.subheader("📏 Model Calibration (Evaluation Set)")
        st.caption(
            "These metrics are computed on the held-out evaluation dataset "
            "(550 samples) and describe how well the model's confidence "
            "aligns with actual accuracy."
        )

        mc1, mc2 = st.columns(2)
        mc1.metric(
            "Brier Score",
            f"{cal.brier_score:.4f}",
            help="Mean squared error between predicted probabilities and "
                 "actual outcomes. 0 = perfect, lower is better.",
        )
        mc2.metric(
            "Expected Calibration Error (ECE)",
            f"{cal.ece:.4f}",
            help="Weighted average of |accuracy − confidence| per bin. "
                 "0 = perfectly calibrated.",
        )

        # Reliability diagram data
        with st.expander("Reliability Diagram Data"):
            cal_df = pd.DataFrame(cal.bin_data)
            cal_df.columns = [
                "Bin Lower", "Bin Upper",
                "Mean Confidence", "Mean Accuracy", "Count",
            ]
            st.dataframe(cal_df, use_container_width=True)

            st.caption(
                "In a perfectly calibrated model, Mean Confidence ≈ Mean "
                "Accuracy in every bin."
            )

    # ── Research disclaimer ──────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "**⚕️ Research Disclaimer:** This system is a research prototype "
        "for gene-expression-based disease classification.  The confidence "
        "scores shown are model-internal probabilities and have NOT been "
        "clinically validated.  They should NOT be used for medical "
        "diagnosis, treatment decisions, or as a substitute for professional "
        "medical advice."
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Final Year Project | Hybrid PSO–ACO + Deep Learning | Gene Expression Analysis"
)
