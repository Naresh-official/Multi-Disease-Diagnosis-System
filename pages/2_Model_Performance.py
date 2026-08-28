"""
Page 2 — Model Performance
==========================

Evaluate the trained DNN classifier on data parsed from the GEO
Series Matrix files.  Displays aggregate metrics, per-class
breakdown, confusion matrix, ROC curves, PR curves, and a
performance bar chart.

⚠️  Important limitation
    The evaluation dataset is constructed from the same GEO files
    used to train the model.  No separate held-out test set was
    provided.  The metrics reported here therefore reflect in-sample
    performance and should not be interpreted as true generalisation.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from core.config import APP_CONFIG
from evaluation.evaluator import (
    load_saved_results,
    run_evaluation,
    save_results,
)
from evaluation.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_per_class_performance,
    plot_roc_curves,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
st.write("Comprehensive evaluation of the Hybrid PSO + ACO + DNN classifier.")

# ── Limitation banner ───────────────────────────────────────────────────────
st.warning(
    "⚠️ **Data limitation:** The evaluation data is constructed from the same "
    "GEO Series Matrix files used to train the model.  No independent held-out "
    "test set was provided.  These metrics reflect **in-sample performance** and "
    "may overestimate true generalisation."
)

# ── Load or compute results ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Running model evaluation…", hash_funcs={pd.DataFrame: lambda df: hash(df.shape)})
def _get_results() -> dict:
    results = run_evaluation()
    save_results(results)
    return results

try:
    results = _get_results()
except Exception as e:
    st.error(f"Failed to run evaluation: {e}")
    st.stop()

metrics = results["metrics"]
per_class = results["per_class"]
cm = results["cm"]
y_true = results["y_true"]
y_prob = results["y_prob"]
class_names = results["class_names"]

# ── Dataset summary ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Dataset Summary")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total Samples", f"{results['n_samples']:,}")
col_b.metric("Raw Features", f"{results['n_features_raw']:,}")
col_c.metric("Selected Features", results["n_features_final"])
col_d.metric("Disease Classes", len(class_names))

# Per-class sample counts
with st.expander("Per-class sample counts"):
    class_counts_df = pd.DataFrame([
        {"Disease": class_names[i], "Samples": results["n_per_class"].get(i, 0)}
        for i in range(len(class_names))
    ])
    st.dataframe(class_counts_df, use_container_width=True, hide_index=True)

# ── Top-level metric cards ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Aggregate Metrics")

m1, m2, m3, m4 = st.columns(4)

accuracy = metrics["accuracy"]
f1_macro = metrics["f1_macro"]
f1_weighted = metrics["f1_weighted"]
roc_auc = metrics.get("roc_auc_ovr")

m1.metric("Accuracy", f"{accuracy:.4f}", help="Overall fraction of correct predictions")
m2.metric("Macro F1", f"{f1_macro:.4f}", help="Unweighted mean of per-class F1 scores — treats all classes equally")
m3.metric("Weighted F1", f"{f1_weighted:.4f}", help="F1 weighted by class support — reflects class imbalance")
m4.metric(
    "ROC-AUC (OvR)",
    f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
    help="Macro-average One-vs-Rest ROC-AUC",
)

# Additional detail
with st.expander("Detailed aggregate metrics"):
    detail_df = pd.DataFrame([
        {"Metric": "Accuracy", "Value": f"{metrics['accuracy']:.4f}"},
        {"Metric": "Precision (macro)", "Value": f"{metrics['precision_macro']:.4f}"},
        {"Metric": "Recall (macro)", "Value": f"{metrics['recall_macro']:.4f}"},
        {"Metric": "F1 (macro)", "Value": f"{metrics['f1_macro']:.4f}"},
        {"Metric": "F1 (weighted)", "Value": f"{metrics['f1_weighted']:.4f}"},
        {"Metric": "ROC-AUC (OvR, macro)", "Value": f"{metrics['roc_auc_ovr']:.4f}" if metrics.get("roc_auc_ovr") else "N/A"},
    ])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

# ── Classification performance table ────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Classification Performance")

perf_df = pd.DataFrame([
    {
        "Disease": m["class_name"],
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1-Score": f"{m['f1']:.4f}",
        "AUC": f"{m['auc']:.4f}" if m["auc"] is not None else "N/A",
        "Support": m["support"],
    }
    for m in per_class
])

st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.caption(
    "Macro averages — Precision: "
    f"{metrics['precision_macro']:.4f},  "
    "Recall: "
    f"{metrics['recall_macro']:.4f},  "
    "F1: "
    f"{metrics['f1_macro']:.4f}"
)

# ── Confusion matrix ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔢 Confusion Matrix")

fig_cm = plot_confusion_matrix(cm, class_names)
st.plotly_chart(fig_cm, use_container_width=True)

# Raw counts
with st.expander("Raw confusion matrix counts"):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    st.dataframe(cm_df, use_container_width=True)

# ── ROC curves ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 ROC Curves")

fig_roc = plot_roc_curves(y_true, y_prob, class_names)
st.plotly_chart(fig_roc, use_container_width=True)

# ── Precision-Recall curves ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Precision-Recall Curves")

fig_pr = plot_precision_recall_curves(y_true, y_prob, class_names)
st.plotly_chart(fig_pr, use_container_width=True)

# ── Per-class performance bar chart ─────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Class-wise Performance")

fig_bar = plot_per_class_performance(per_class)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Prediction confidence distribution ──────────────────────────────────────
st.markdown("---")
st.subheader("📊 Prediction Confidence Distribution")

pred_conf = np.max(y_prob, axis=1)
correct_mask = y_true == np.argmax(y_prob, axis=1)

import plotly.graph_objects as go

fig_conf = go.Figure()
fig_conf.add_trace(go.Histogram(
    x=pred_conf[correct_mask],
    name="Correct predictions",
    opacity=0.7,
    marker_color="#2ca02c",
    nbinsx=30,
))
fig_conf.add_trace(go.Histogram(
    x=pred_conf[~correct_mask],
    name="Incorrect predictions",
    opacity=0.7,
    marker_color="#d62728",
    nbinsx=30,
))
fig_conf.update_layout(
    barmode="overlay",
    xaxis_title="Prediction Confidence",
    yaxis_title="Count",
    height=400,
)
st.plotly_chart(fig_conf, use_container_width=True)

st.caption(
    f"Median confidence: {np.median(pred_conf):.4f} | "
    f"Mean confidence: {np.mean(pred_conf):.4f}"
)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Model: Hybrid PSO–ACO + DNN | "
    f"Evaluated on {results['n_samples']} samples from GEO datasets | "
    f"In-sample metrics — see warning above"
)
