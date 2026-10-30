"""
Page 2 — Model Performance
==========================

Performance metrics and evaluation breakdown for the 5 disease-specific
models trained on the 70-30 train-test split.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.disease_models import available_disease_models
from utils.theme import apply_theme, is_dark_theme

st.set_page_config(
    page_title="Multi-Disease Diagnosis | Model Performance",
    page_icon="📈",
    layout="wide",
)

apply_theme()

st.title("📈 Model Performance & Evaluation")
st.write(
    "Evaluation metrics on the **held-out 30% test split** for each disease-specific model. "
    "All splits use group-aware stratification to eliminate patient/donor data leakage."
)
st.markdown("---")

models = available_disease_models()
if not models:
    st.error("No trained disease models found in `artifacts/disease_models/`.")
    st.stop()

# ── Summary Table ───────────────────────────────────────────────────────────
st.subheader("📊 Comparative Performance Summary (30% Held-Out Test Set)")

table_rows = []
for m in models:
    table_rows.append({
        "Disease": m["disease"],
        "GEO Study": m["study"],
        "Train Samples (70%)": m["train_samples"],
        "Test Samples (30%)": m["test_samples"],
        "Total Probes": f"{m['total_probe_features']:,}",
        "Biomarkers Selected": m["selected_biomarker_count"],
        "Classifier": m["selected_classifier"],
        "Held-Out Accuracy": m["test_accuracy"],
        "Macro F1": m["test_macro_f1"],
        "Balanced Accuracy": m.get("test_balanced_accuracy", m["test_accuracy"]),
    })

summary_df = pd.DataFrame(table_rows)

st.dataframe(
    summary_df.style.format({
        "Held-Out Accuracy": "{:.2%}",
        "Macro F1": "{:.4f}",
        "Balanced Accuracy": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Performance Chart ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Performance Across Disease Models")

chart_df = summary_df.copy()
chart_df["Accuracy (%)"] = chart_df["Held-Out Accuracy"] * 100
chart_df["Macro F1 (%)"] = chart_df["Macro F1"] * 100

dark = is_dark_theme()
font_color = "#f8fafc" if dark else "#0f172a"
grid_color = "#334155" if dark else "#e2e8f0"

fig = px.bar(
    chart_df,
    x="Disease",
    y=["Accuracy (%)", "Macro F1 (%)"],
    barmode="group",
    title="Held-Out Accuracy and Macro F1 by Target Disease",
    color_discrete_sequence=["#0f766e" if not dark else "#14b8a6", "#2563eb" if not dark else "#3b82f6"],
    height=420,
)
fig.update_layout(
    yaxis_range=[50, 100],
    yaxis_title="Score (%)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=font_color),
    xaxis=dict(gridcolor=grid_color),
    yaxis=dict(gridcolor=grid_color),
)
st.plotly_chart(fig, use_container_width=True)

# ── Disease-Specific Drilldown ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Detailed Disease Breakdown")

selected_slug = st.selectbox(
    "Select Disease Model for In-Depth Evaluation",
    options=[m["slug"] for m in models],
    format_func=lambda s: next(m["disease"] for m in models if m["slug"] == s),
)

cur_model = next(m for m in models if m["slug"] == selected_slug)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Held-Out Accuracy", f"{cur_model['test_accuracy']*100:.2f}%")
col_m2.metric("Macro F1", f"{cur_model['test_macro_f1']:.4f}")
col_m3.metric("Balanced Accuracy", f"{cur_model.get('test_balanced_accuracy', 0)*100:.2f}%")
col_m4.metric("Test Split Size", f"{cur_model['test_samples']} samples")

d_col1, d_col2 = st.columns([1, 1])

with d_col1:
    st.write("#### Confusion Matrix")
    cm = np.array(cur_model.get("confusion_matrix", [[0, 0], [0, 0]]))
    labels = ["Control", "Disease"]

    # Build annotations manually (compatible with Plotly 7+)
    annotations = [
        dict(
            x=labels[j],
            y=labels[i],
            text=str(cm[i, j]),
            font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black", size=18),
            showarrow=False,
        )
        for i in range(len(labels))
        for j in range(len(labels))
    ]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
        )
    )
    fig_cm.update_layout(
        annotations=annotations,
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="True Label", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with d_col2:
    st.write("#### Classification Report")
    cr = cur_model.get("classification_report", {})
    if cr:
        cr_df = pd.DataFrame(cr).T
        cr_display = cr_df.loc[["Control", "Disease"]][["precision", "recall", "f1-score", "support"]]
        st.dataframe(
            cr_display.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }),
            use_container_width=True,
        )

    st.write(f"**Top Ranked Biomarker Probes ({len(cur_model.get('top_biomarkers', []))})**")
    top_probes = cur_model.get("top_biomarkers", [])
    if top_probes:
        st.caption(", ".join(top_probes[:15]) + ("..." if len(top_probes) > 15 else ""))
