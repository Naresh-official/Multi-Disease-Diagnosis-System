"""
Page 4 — Biomarker Analysis
===========================

Rank and visualise the 80 selected biomarkers using SHAP values.
Provides global importance, disease-specific rankings, contribution
direction, expression distributions, and the feature selection journey.

Biological annotation is NOT fabricated — the system uses raw
Affymetrix probe identifiers and clearly states when gene-level
annotation is unavailable.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import APP_CONFIG

st.set_page_config(page_title="Biomarker Analysis", layout="wide")
st.title("🧬 Biomarker Analysis")

# ── Load / compute SHAP values ──────────────────────────────────────────────


@st.cache_data(show_spinner="Loading evaluation data…")
def _load_eval_data():
    """Load preprocessed evaluation features and labels."""
    X_80 = np.load("data/visualizations/X_80.npy")
    y = np.load("data/visualizations/y.npy")
    return X_80, y


@st.cache_data(show_spinner="Computing SHAP values for biomarker analysis…")
def _compute_shap_cached(_X_hash: str, X_80: np.ndarray):
    """Compute SHAP values across all evaluation samples (cached)."""
    from biomarkers.analyzer import compute_shap_values
    from core.predictor import load_artifacts

    artifacts = load_artifacts()
    shap_values = compute_shap_values(X_80, artifacts)
    return shap_values


@st.cache_data(show_spinner="Computing rankings…")
def _compute_rankings(_shap_hash: str, shap_values: np.ndarray, y: np.ndarray):
    """Compute all rankings from SHAP values (cached)."""
    from biomarkers.analyzer import compute_all_rankings
    probe_ids = get_all_probe_ids()
    return compute_all_rankings(shap_values, y, probe_ids)


# ── Load data ───────────────────────────────────────────────────────────────

try:
    X_80, y = _load_eval_data()
except Exception as e:
    st.error(f"Failed to load evaluation data: {e}")
    st.stop()

# ── Compute SHAP values and rankings ────────────────────────────────────────

shap_key = f"{X_80.shape}_{X_80.sum():.6f}"
shap_values = _compute_shap_cached(shap_key, X_80)
rankings = _compute_rankings(shap_key, shap_values, y)

from biomarkers.ranking import ranking_to_dataframe, get_top_n
from biomarkers.metadata import (
    get_identifier_type,
    get_annotation_availability,
    get_identifier_format_description,
    get_all_probe_ids,
)

probe_ids = get_all_probe_ids()

# ── Page sections ───────────────────────────────────────────────────────────

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Feature Selection Journey
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🔬 Feature Selection Journey")

from biomarkers.analyzer import get_feature_selection_journey
journey = get_feature_selection_journey()

st.markdown(
    "The hybrid PSO + ACO feature selection pipeline reduces the "
    "gene-expression feature space from thousands of probes to a "
    "compact set of 80 biomarkers."
)

# Visual journey
cols = st.columns(5)
steps = [
    ("Original", journey["total_features"], "📋"),
    ("Variance Filter", journey["after_variance"], "📊"),
    ("PSO Selected", journey["after_pso"], "🐝"),
    ("ACO Selected", journey["after_aco"], "🐜"),
    ("Final Biomarkers", journey["after_aco"], "🧬"),
]

for col, (name, count, icon) in zip(cols, steps):
    with col:
        st.metric(label=f"{icon} {name}", value=f"{count:,}")

reduction = journey["reduction_pct"]
st.info(
    f"**Feature reduction: {reduction}%** "
    f"({journey['total_features']:,} → {journey['after_aco']} features)"
)

# Show the journey as a flow
st.markdown("```")
st.markdown(
    f"  {journey['total_features']:,} raw probes "
    f"→ Variance Threshold ({journey['after_variance']:,}) "
    f"→ PSO ({journey['after_pso']:,}) "
    f"→ ACO ({journey['after_aco']})"
)
st.markdown("```")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Top Biomarkers
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🏆 Top Biomarkers")

col_n, col_view = st.columns([1, 3])
with col_n:
    top_n = st.selectbox(
        "Show top",
        [10, 20, 30, 50, 80],
        index=0,
        key="top_n_select",
    )
with col_view:
    view_mode = st.radio(
        "View",
        ["Table", "Chart"],
        horizontal=True,
        key="view_mode",
    )

global_ranking = rankings["Global"]
top_bio = get_top_n(global_ranking, top_n)
df_ranking = ranking_to_dataframe(global_ranking).head(top_n)

if view_mode == "Table":
    st.dataframe(
        df_ranking,
        use_container_width=True,
        hide_index=True,
    )
else:
    # Horizontal bar chart of importance
    df_chart = df_ranking.copy()
    df_chart["Label"] = df_chart.apply(
        lambda r: f"#{r['Rank']} {r['Probe ID']}", axis=1
    )

    colors = ["#2ecc71" if d == "Positive" else "#e74c3c" if d == "Negative" else "#95a5a6"
              for d in df_chart["Direction"]]

    fig = go.Figure(go.Bar(
        x=df_chart["Importance (|SHAP|)"],
        y=df_chart["Label"],
        orientation="h",
        marker_color=colors,
        hovertext=df_chart.apply(
            lambda r: (
                f"Probe: {r['Probe ID']}<br>"
                f"Importance: {r['Importance (|SHAP|)']:.6f}<br>"
                f"Mean SHAP: {r['Mean SHAP']:.6f}<br>"
                f"Direction: {r['Direction']}"
            ),
            axis=1,
        ),
        hoverinfo="text",
    ))

    fig.update_layout(
        title=f"Top {top_n} Biomarkers by SHAP Importance",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 28 + 100),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Disease-Specific Biomarkers
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🎯 Disease-Specific Biomarkers")

st.markdown(
    "Select a disease to see which biomarkers are most important "
    "for predicting that specific class.  Importance is computed as "
    "the mean absolute SHAP value for the selected class, averaged "
    "over samples belonging to that disease."
)

disease_options = list(APP_CONFIG.disease_map.values())
selected_disease = st.selectbox(
    "Disease",
    disease_options,
    key="disease_select",
)

if selected_disease in rankings:
    disease_ranking = rankings[selected_disease]
    disease_top = get_top_n(disease_ranking, 10)
    df_disease = ranking_to_dataframe(disease_ranking).head(10)

    st.markdown(
        f"**{selected_disease}** — based on {disease_ranking.total_samples} samples"
    )

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.dataframe(
            df_disease[["Rank", "Probe ID", "Importance (|SHAP|)", "Mean SHAP", "Direction"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_chart:
        df_dchart = df_disease.copy()
        df_dchart["Label"] = df_dchart.apply(
            lambda r: f"#{r['Rank']} {r['Probe ID'][:15]}", axis=1
        )

        colors = ["#2ecc71" if d == "Positive" else "#e74c3c" if d == "Negative" else "#95a5a6"
                  for d in df_dchart["Direction"]]

        fig = go.Figure(go.Bar(
            x=df_dchart["Importance (|SHAP|)"],
            y=df_dchart["Label"],
            orientation="h",
            marker_color=colors,
            hovertext=df_dchart.apply(
                lambda r: (
                    f"Probe: {r['Probe ID']}<br>"
                    f"Importance: {r['Importance (|SHAP|)']:.6f}<br>"
                    f"Mean SHAP: {r['Mean SHAP']:.6f}"
                ),
                axis=1,
            ),
            hoverinfo="text",
        ))

        fig.update_layout(
            title=f"Top 10 Biomarkers for {selected_disease}",
            xaxis_title="Mean |SHAP Value|",
            yaxis=dict(autorange="reversed"),
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Biomarker Contribution
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Biomarker Contribution (Positive vs Negative)")

st.markdown(
    "SHAP values can be positive or negative. A **positive** SHAP value "
    "means the feature pushes the prediction **toward** the class.  A "
    "**negative** value pushes **away** from the class."
)

contrib_top_n = st.slider(
    "Number of biomarkers to show",
    min_value=10,
    max_value=80,
    value=20,
    step=5,
    key="contrib_n",
)

df_contrib = ranking_to_dataframe(global_ranking).head(contrib_top_n)
df_contrib["abs_mean"] = df_contrib["Mean SHAP"].abs()

fig = go.Figure()

# Positive contributions
pos = df_contrib[df_contrib["Mean SHAP"] > 0].sort_values("Mean SHAP", ascending=True)
if len(pos) > 0:
    fig.add_trace(go.Bar(
        x=pos["Mean SHAP"],
        y=pos["Probe ID"],
        orientation="h",
        name="Positive (→ class)",
        marker_color="#2ecc71",
    ))

# Negative contributions
neg = df_contrib[df_contrib["Mean SHAP"] < 0].sort_values("Mean SHAP", ascending=False)
if len(neg) > 0:
    fig.add_trace(go.Bar(
        x=neg["Mean SHAP"],
        y=neg["Probe ID"],
        orientation="h",
        name="Negative (← class)",
        marker_color="#e74c3c",
    ))

fig.update_layout(
    title=f"SHAP Contribution Direction (Top {contrib_top_n} Biomarkers)",
    xaxis_title="Mean SHAP Value (signed)",
    barmode="relative",
    height=max(400, contrib_top_n * 28 + 100),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# Summary statistics
n_pos = (df_contrib["Mean SHAP"] > 0).sum()
n_neg = (df_contrib["Mean SHAP"] < 0).sum()
n_neutral = (df_contrib["Mean SHAP"] == 0).sum()
st.markdown(
    f"**Summary:** {n_pos} positive, {n_neg} negative, "
    f"{n_neutral} neutral contributions (out of {contrib_top_n})"
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Expression Distribution
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Biomarker Expression Distribution")

st.markdown(
    "Explore how a biomarker's expression varies across disease classes. "
    "This shows the preprocessed (scaled) expression values."
)

# Select a biomarker
selected_probe = st.selectbox(
    "Select biomarker",
    probe_ids,
    key="expr_probe",
    format_func=lambda x: f"{x} (idx {probe_ids.index(x)})",
)

feat_idx = probe_ids.index(selected_probe)

# Get expression values for this feature across all samples
expr_values = X_80[:, feat_idx]

# Build dataframe with disease labels    short_names = ["CAD", "Autoimmune", "Alzheimer's", "Parkinson's"]
df_expr = pd.DataFrame({
    "Expression": expr_values,
    "Disease": [short_names[int(label)] for label in y],
})

col_box, col_hist = st.columns([1, 1])

with col_box:
    fig = px.box(
        df_expr,
        x="Disease",
        y="Expression",
        color="Disease",
        title=f"Expression of {selected_probe}",
        points="all",
        opacity=0.6,
        width=500,
        height=400,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_hist:
    fig = px.histogram(
        df_expr,
        x="Expression",
        color="Disease",
        nbins=30,
        opacity=0.6,
        title=f"Expression Distribution — {selected_probe}",
        width=500,
        height=400,
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Show biomarker detail card
st.markdown("#### Biomarker Detail")

rank_info = global_ranking.biomarkers[feat_idx] if feat_idx < len(global_ranking.biomarkers) else None

detail_cols = st.columns(4)
detail_cols[0].metric("Probe ID", selected_probe)
detail_cols[1].metric("Global Rank", f"#{rank_info.rank}" if rank_info else "N/A")
detail_cols[2].metric("Importance", f"{rank_info.importance:.6f}" if rank_info else "N/A")
detail_cols[3].metric(
    "Direction",
    "Positive →" if rank_info and rank_info.mean_shap > 0
    else "Negative ←" if rank_info and rank_info.mean_shap < 0
    else "Neutral"
)

if rank_info:
    st.markdown(
        f"Mean SHAP: **{rank_info.mean_shap:+.6f}** | "
        f"Std: {rank_info.std_shap:.6f} | "
        f"Importance (|SHAP|): {rank_info.importance:.6f}"
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Biomarker Detail Explorer
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Biomarker Detail Explorer")

st.markdown(
    "Select any biomarker to view its full ranking information, "
    "expression pattern, and contribution details."
)

explorer_probe = st.selectbox(
    "Select biomarker to explore",
    probe_ids,
    key="explorer_probe",
    format_func=lambda x: f"{x}",
)

explorer_idx = probe_ids.index(explorer_probe)

# Find ranking info for this probe across all rankings
st.markdown(f"**{explorer_probe}**")

explorer_cols = st.columns(2)

with explorer_cols[0]:
    st.markdown("**Global Ranking:**")
    global_bio = global_ranking.biomarkers[explorer_idx]
    st.json({
        "rank": global_bio.rank,
        "probe_id": global_bio.probe_id,
        "importance": round(global_bio.importance, 6),
        "mean_shap": round(global_bio.mean_shap, 6),
        "std_shap": round(global_bio.std_shap, 6),
        "direction": "Positive" if global_bio.mean_shap > 0 else "Negative",
    })

with explorer_cols[1]:
    st.markdown("**Disease-Specific Rankings:**")
    disease_data = []
    for disease_name in APP_CONFIG.disease_map.values():
        if disease_name in rankings:
            dr = rankings[disease_name]
            if explorer_idx < len(dr.biomarkers):
                b = dr.biomarkers[explorer_idx]
                disease_data.append({
                    "disease": disease_name,
                    "rank": b.rank,
                    "importance": round(b.importance, 6),
                    "mean_shap": round(b.mean_shap, 6),
                    "direction": "Positive" if b.mean_shap > 0 else "Negative",
                })
    st.dataframe(pd.DataFrame(disease_data), hide_index=True, use_container_width=True)

# ── Annotation notice ───────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📝 Biological Annotation Status")

st.warning(get_annotation_availability())

with st.expander("About the feature identifiers"):
    st.markdown(get_identifier_format_description())

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "SHAP values computed using KernelExplainer on the DNN model. "
    "Importance = mean |SHAP| across relevant samples. "
    "Direction = sign of mean SHAP (positive → toward class). "
    "All biomarkers use raw Affymetrix probe IDs — no gene annotations are fabricated."
)
