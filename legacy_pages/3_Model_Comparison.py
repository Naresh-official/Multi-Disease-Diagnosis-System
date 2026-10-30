"""
Page 3 — Model Comparison
==========================

Compares five feature-selection strategies using identical DNN
architectures trained on different feature subsets.

This page demonstrates the scientific contribution of the hybrid
PSO + ACO feature-selection approach by showing how classification
performance varies with different feature-reduction strategies.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.model_comparison import (
    load_comparison,
    run_all_experiments,
    save_comparison,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")
st.write(
    "Comparing feature-selection strategies: "
    "how does reducing 10,571 gene-expression features "
    "affect classification performance?"
)

# ── Load or run ─────────────────────────────────────────────────────────────
saved = load_comparison()

if saved is None:
    st.info("No comparison results found. Running experiments (this may take a few minutes)…")
    with st.spinner("Training DNNs on different feature subsets…"):
        results = run_all_experiments(test_size=0.2, random_state=42, epochs=50, batch_size=32)
        save_comparison(results)
        # Reload to get the serialised form
        saved = load_comparison()

total_features = saved["total_features"]
class_names = saved["class_names"]
results_list = saved["results"]

# Build a tidy DataFrame for easy plotting
df = pd.DataFrame(results_list)

# ── Limitation banner ───────────────────────────────────────────────────────
st.warning(
    "⚠️ **Important:** The production model was pre-trained on 356 samples "
    "(X_train_fs).  Baseline models 1–4 are freshly trained on a stratified "
    "80/20 split of all 550 GEO samples.  The production model (5) was "
    "**not retrained** — it was evaluated on the same held-out test set.  "
    "Differences partly reflect different training data, not just "
    "feature-selection quality."
)

# ── Section 1: Comparison table ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Comparison Table")

table_df = pd.DataFrame([
    {
        "Approach": r["model_name"],
        "Features": f"{r['feature_count']:,}",
        "Reduction": f"{r['feature_reduction_pct']:.1f}%",
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision (macro)": f"{r['precision_macro']:.4f}",
        "Recall (macro)": f"{r['recall_macro']:.4f}",
        "F1 (macro)": f"{r['f1_macro']:.4f}",
        "F1 (weighted)": f"{r['f1_weighted']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A",
        "Train Time (s)": f"{r['training_time_sec']:.1f}",
    }
    for r in results_list
])

st.dataframe(table_df, use_container_width=True, hide_index=True)

# ── Section 2: Accuracy comparison ──────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Accuracy Comparison")

fig_acc = go.Figure()
fig_acc.add_trace(go.Bar(
    x=df["model_name"],
    y=df["accuracy"],
    marker_color=["#636EFA", "#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
    text=[f"{v:.3f}" for v in df["accuracy"]],
    textposition="outside",
))
fig_acc.update_layout(
    yaxis_title="Accuracy",
    yaxis_range=[0, 1.05],
    height=400,
    showlegend=False,
)
st.plotly_chart(fig_acc, use_container_width=True)

# ── Section 3: F1 comparison ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 F1-Score Comparison")

fig_f1 = go.Figure()
fig_f1.add_trace(go.Bar(
    name="Macro F1",
    x=df["model_name"],
    y=df["f1_macro"],
    marker_color="#636EFA",
    text=[f"{v:.3f}" for v in df["f1_macro"]],
    textposition="outside",
))
fig_f1.add_trace(go.Bar(
    name="Weighted F1",
    x=df["model_name"],
    y=df["f1_weighted"],
    marker_color="#EF553B",
    text=[f"{v:.3f}" for v in df["f1_weighted"]],
    textposition="outside",
))
fig_f1.update_layout(
    barmode="group",
    yaxis_title="F1-Score",
    yaxis_range=[0, 1.1],
    height=400,
)
st.plotly_chart(fig_f1, use_container_width=True)

# ── Section 4: Feature count comparison ─────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Feature Count Comparison")

fig_feat = go.Figure()
fig_feat.add_trace(go.Bar(
    x=df["model_name"],
    y=df["feature_count"],
    marker_color="#AB63FA",
    text=[f"{v:,}" for v in df["feature_count"]],
    textposition="outside",
))
fig_feat.update_layout(
    yaxis_title="Number of Features",
    yaxis_type="log",
    height=400,
    showlegend=False,
)
st.plotly_chart(fig_feat, use_container_width=True)

# ── Section 5: Feature reduction visualization ──────────────────────────────
st.markdown("---")
st.subheader("📉 Feature Reduction")

fig_red = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Features Used", "Reduction Percentage"),
    horizontal_spacing=0.15,
)

fig_red.add_trace(go.Bar(
    x=df["model_name"],
    y=df["feature_count"],
    marker_color="#00CC96",
    text=[f"{v:,}" for v in df["feature_count"]],
    textposition="outside",
    showlegend=False,
), row=1, col=1)

fig_red.add_trace(go.Bar(
    x=df["model_name"],
    y=df["feature_reduction_pct"],
    marker_color="#FFA15A",
    text=[f"{v:.1f}%" for v in df["feature_reduction_pct"]],
    textposition="outside",
    showlegend=False,
), row=1, col=2)

fig_red.update_yaxes(title_text="Feature Count (log)", type="log", row=1, col=1)
fig_red.update_yaxes(title_text="Reduction %", row=1, col=2)
fig_red.update_layout(height=450)
st.plotly_chart(fig_red, use_container_width=True)

# ── Section 6: ROC-AUC comparison ───────────────────────────────────────────
st.markdown("---")
st.subheader("📉 ROC-AUC Comparison")

roc_data = df[df["roc_auc"].notna()]
if not roc_data.empty:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Bar(
        x=roc_data["model_name"],
        y=roc_data["roc_auc"],
        marker_color="#19D3F3",
        text=[f"{v:.4f}" for v in roc_data["roc_auc"]],
        textposition="outside",
    ))
    fig_roc.update_layout(
        yaxis_title="ROC-AUC (OvR, macro)",
        yaxis_range=[0, 1.05],
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_roc, use_container_width=True)
else:
    st.info("ROC-AUC not available for all approaches.")

# ── Section 7: Per-class comparison (for top approaches) ────────────────────
st.markdown("---")
st.subheader("📊 Per-Class F1 by Approach")

# Build per-class data
per_class_rows = []
for r in results_list:
    for pc in r.get("per_class", []):
        per_class_rows.append({
            "Approach": r["model_name"],
            "Disease": pc["class_name"],
            "F1": pc["f1"],
            "Precision": pc["precision"],
            "Recall": pc["recall"],
        })
pc_df = pd.DataFrame(per_class_rows)

if not pc_df.empty:
    fig_pc = go.Figure()
    approaches = pc_df["Approach"].unique()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for i, approach in enumerate(approaches):
        subset = pc_df[pc_df["Approach"] == approach]
        fig_pc.add_trace(go.Bar(
            name=approach,
            x=subset["Disease"],
            y=subset["F1"],
            marker_color=colors[i % len(colors)],
        ))
    fig_pc.update_layout(
        barmode="group",
        yaxis_title="F1-Score",
        yaxis_range=[0, 1.1],
        height=500,
        legend=dict(x=1.02, y=1),
    )
    st.plotly_chart(fig_pc, use_container_width=True)

# ── Section 8: Training cost ────────────────────────────────────────────────
st.markdown("---")
st.subheader("⏱️ Training & Inference Cost")

train_data = df[df["training_time_sec"] > 0]
if not train_data.empty:
    fig_time = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Time", "Inference Time (test set)"),
        horizontal_spacing=0.15,
    )
    fig_time.add_trace(go.Bar(
        x=train_data["model_name"],
        y=train_data["training_time_sec"],
        marker_color="#FF6692",
        text=[f"{v:.1f}s" for v in train_data["training_time_sec"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig_time.add_trace(go.Bar(
        x=df["model_name"],
        y=df["inference_time_sec"],
        marker_color="#B6E880",
        text=[f"{v:.4f}s" for v in df["inference_time_sec"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig_time.update_yaxes(title_text="Seconds", row=1, col=1)
    fig_time.update_yaxes(title_text="Seconds", row=1, col=2)
    fig_time.update_layout(height=400)
    st.plotly_chart(fig_time, use_container_width=True)

# ── Section 9: Key findings ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📝 Key Findings")

# Compute key stats
best_acc_idx = df["accuracy"].idxmax()
best_f1_idx = df["f1_macro"].idxmax()
prod_idx = df[df["model_name"].str.contains("Production")].index[0]

best_acc_row = df.loc[best_acc_idx]
best_f1_row = df.loc[best_f1_idx]
prod_row = df.loc[prod_idx]

st.markdown(f"""
- **Best accuracy:** {best_acc_row['model_name']} — **{best_acc_row['accuracy']:.4f}**
  ({best_acc_row['feature_count']:,} features, {best_acc_row['feature_reduction_pct']:.1f}% reduction)

- **Best macro F1:** {best_f1_row['model_name']} — **{best_f1_row['f1_macro']:.4f}**

- **Production model:** {prod_row['model_name']} —
  accuracy {prod_row['accuracy']:.4f}, F1 {prod_row['f1_macro']:.4f}
  ({prod_row['feature_count']:,} features)

- The hybrid PSO + ACO approach reduces the feature space from
  **{total_features:,} to {prod_row['feature_count']:,}** features
  ({prod_row['feature_reduction_pct']:.1f}% reduction) while maintaining
  {"competitive" if prod_row['f1_macro'] > 0.5 else "moderate"} classification
  performance.

- **Note:** Results are sensitive to train/test split and random seed.
  The production model was trained on a different data subset (356 samples)
  than the baselines (440 training samples).  Direct accuracy comparisons
  should be interpreted with this caveat.
""")

# ── Section 10: Methodology ─────────────────────────────────────────────────
with st.expander("📋 Methodology & Reproducibility"):
    st.markdown("""
**Data:** Samples from GEO Series Matrix files (GSE66360, GSE50772,
GSE48350, GSE6613) after alignment to the model's 10,571 feature space.

**Split:** Stratified 80/20 train/test split (random_state=42).

**Baseline DNN architecture** (same for all approaches):
```
Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(4, softmax)
```
Optimizer: Adam | Loss: sparse_categorical_crossentropy | Epochs: 50

**Feature selection methods:**
1. **No selection:** All 10,571 features
2. **Variance filtering:** VarianceThreshold(threshold=0.01) — all features pass
3. **PSO:** 5,302 features selected by Particle Swarm Optimization indices
4. **ACO:** 80 features selected by Ant Colony Optimization indices
5. **Hybrid PSO+ACO:** Pre-trained production model (not retrained)

**Reproducibility:** Results saved to `data/evaluation/model_comparison.json`.
To re-run: `python -c "from evaluation.model_comparison import run_all_experiments, save_comparison; save_comparison(run_all_experiments())"`
""")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Compared {len(results_list)} approaches | "
    f"{total_features:,} → {prod_row['feature_count']:,} features | "
    f"Train split: 440 samples | Test split: 110 samples"
)
