"""
Page 5 — Dataset Visualization
==============================

Dimensionality-reduction plots (PCA, t-SNE, UMAP) and gene-expression
heatmaps for the evaluation datasets.

This page visualizes the 550 samples from the 4 GEO Series Matrix
files after running them through the same preprocessing pipeline
used for inference (StandardScaler → VarianceThreshold → PSO → ACO).

Visualization preprocessing is distinct from model inference
preprocessing — this page does NOT modify any model artifacts.
"""

from __future__ import annotations

import streamlit as st
import numpy as np

st.set_page_config(page_title="Dataset Visualization", layout="wide")
st.title("📊 Dataset Visualization")

# ── Load data (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading gene-expression data…")
def _load_data():
    """Load and preprocess data for visualization (cached)."""
    from visualization.data_loader import load_or_compute_viz_data
    return load_or_compute_viz_data()


@st.cache_data(show_spinner="Running PCA…")
def _run_pca(X_scaled: np.ndarray, n_components: int):
    """Run PCA (cached)."""
    from visualization.pca import run_pca
    return run_pca(X_scaled, n_components=n_components)


@st.cache_data(show_spinner="Running t-SNE…")
def _run_tsne(X: np.ndarray, perplexity: float, max_iter: int):
    """Run t-SNE (cached — this is expensive)."""
    from visualization.tsne import run_tsne
    return run_tsne(X, perplexity=perplexity, max_iter=max_iter)


@st.cache_data(show_spinner="Running UMAP…")
def _run_umap(X: np.ndarray, n_neighbors: int, min_dist: float):
    """Run UMAP (cached — this can be expensive)."""
    from visualization.umap import run_umap
    return run_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)


# ── Main content ────────────────────────────────────────────────────────────

try:
    data = _load_data()
except Exception as e:
    st.error(f"Failed to load visualization data: {e}")
    st.stop()

X_80 = data["X_80"]
X_scaled = data["X_scaled"]
y = data["y"]
class_names = data["class_names"]
short_names = ["CAD", "Autoimmune", "Alzheimer's", "Parkinson's"]
feature_names_80 = data["feature_names_80"]
dataset_labels = data["dataset_labels"]
n_samples = data["n_samples"]

# ── Dataset overview ────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", n_samples)
col2.metric("Raw Features", f"{X_scaled.shape[1]:,}")
col3.metric("Selected Features", X_80.shape[1])
col4.metric("Disease Classes", len(class_names))

# Class distribution
st.markdown("**Samples per disease class:**")
class_counts = {}
for label in y:
    name = short_names[int(label)]
    class_counts[name] = class_counts.get(name, 0) + 1

col_a, col_b, col_c, col_d = st.columns(4)
for i, (name, count) in enumerate(class_counts.items()):
    [col_a, col_b, col_c, col_d][i].metric(name, count)

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────

tab_pca, tab_tsne, tab_umap, tab_heatmap = st.tabs([
    "🔍 PCA",
    "🗺️ t-SNE",
    "🌐 UMAP",
    "🔥 Biomarker Heatmap",
])

# ── PCA Tab ─────────────────────────────────────────────────────────────────

with tab_pca:
    st.markdown("""
    **Principal Component Analysis (PCA)** projects high-dimensional
    gene-expression data into a lower-dimensional space while
    preserving maximum variance.  This helps visualize whether
    disease classes separate in the principal component space.
    """)

    st.info(
        "PCA is run on the full scaled feature space (10,571 features) "
        "to capture maximum variance, not on the 80 selected biomarkers."
    )

    n_components = st.slider(
        "Number of components",
        min_value=2,
        max_value=min(30, X_scaled.shape[1]),
        value=2,
        step=1,
        key="pca_n",
    )

    pca_model, X_pca = _run_pca(X_scaled, n_components)

    from visualization.pca import plot_pca_scatter, plot_explained_variance

    color_by = st.radio(
        "Color points by",
        ["disease", "dataset"],
        horizontal=True,
        key="pca_color",
    )

    fig_scatter = plot_pca_scatter(
        X_pca, y, class_names, short_names, pca_model,
        dataset_labels=dataset_labels, color_by=color_by,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if n_components >= 2:
        fig_var = plot_explained_variance(pca_model, max_components=n_components)
        st.plotly_chart(fig_var, use_container_width=True)

        cum_var = np.cumsum(pca_model.explained_variance_ratio_)
        st.markdown(
            f"**First 2 components explain {cum_var[1]:.1%} of total variance.** "
            f"Top {n_components} components explain {cum_var[-1]:.1%}."
        )

# ── t-SNE Tab ───────────────────────────────────────────────────────────────

with tab_tsne:
    st.markdown("""
    **t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a
    non-linear technique that visualizes high-dimensional data by
    preserving local neighborhoods.  It is particularly good at
    revealing clusters in the data.
    """)

    st.warning(
        "⚠️ t-SNE is stochastic and computationally expensive. "
        "Results are cached for the current parameter combination."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        perplexity = st.slider(
            "Perplexity",
            min_value=5.0,
            max_value=50.0,
            value=30.0,
            step=5.0,
            key="tsne_perp",
            help="Controls the balance between local and global structure. "
                 "Typical values: 5-50.",
        )
    with col_p2:
        n_iter = st.slider(
            "Iterations",
            min_value=250,
            max_value=2000,
            value=1000,
            step=250,
            key="tsne_iter",
        )

    X_tsne = _run_tsne(X_80, perplexity, n_iter)

    # Validate: no NaN coordinates
    if np.any(np.isnan(X_tsne)):
        st.error("t-SNE produced NaN coordinates. Try different parameters.")
    else:
        from visualization.tsne import plot_tsne_scatter

        color_by = st.radio(
            "Color points by",
            ["disease", "dataset"],
            horizontal=True,
            key="tsne_color",
        )

        fig = plot_tsne_scatter(
            X_tsne, y, class_names, short_names,
            perplexity, n_iter,
            dataset_labels=dataset_labels, color_by=color_by,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── UMAP Tab ────────────────────────────────────────────────────────────────

with tab_umap:
    st.markdown("""
    **Uniform Manifold Approximation and Projection (UMAP)** is a
    non-linear dimensionality reduction technique that preserves both
    local and global data structure.  It often produces more
    interpretable global structure than t-SNE.
    """)

    st.warning(
        "⚠️ UMAP can be slow on large datasets. "
        "Results are cached for the current parameter combination."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            key="umap_nn",
            help="Controls local vs global structure. "
                 "Smaller = more local focus.",
        )
    with col_u2:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=0.99,
            value=0.1,
            step=0.05,
            key="umap_dist",
            help="Controls how tightly points are packed. "
                 "Smaller = tighter clusters.",
        )

    X_umap = _run_umap(X_80, n_neighbors, min_dist)

    if np.any(np.isnan(X_umap)):
        st.error("UMAP produced NaN coordinates. Try different parameters.")
    else:
        from visualization.umap import plot_umap_scatter

        color_by = st.radio(
            "Color points by",
            ["disease", "dataset"],
            horizontal=True,
            key="umap_color",
        )

        fig = plot_umap_scatter(
            X_umap, y, class_names, short_names,
            n_neighbors, min_dist,
            dataset_labels=dataset_labels, color_by=color_by,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Heatmap Tab ─────────────────────────────────────────────────────────────

with tab_heatmap:
    st.markdown("""
    **Biomarker Expression Heatmap** displays the expression levels of
    the selected biomarker features (post-PSO + ACO) across all
    samples, sorted by disease class.  Values are Z-score normalized
    per feature for visual clarity.
    """)

    heatmap_mode = st.radio(
        "View mode",
        ["All samples (80 biomarkers)", "Disease summary (top N features)"],
        horizontal=True,
        key="heatmap_mode",
    )

    if heatmap_mode == "All samples (80 biomarkers)":
        top_n_option = st.slider(
            "Show top N features (by variance)",
            min_value=10,
            max_value=80,
            value=20,
            step=5,
            key="heatmap_topn",
        )

        from visualization.heatmap import create_biomarker_heatmap

        fig = create_biomarker_heatmap(
            X_80, y, feature_names_80, class_names, short_names,
            top_n=top_n_option,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"Showing **{top_n_option}** of 80 biomarkers, "
            f"sorted by variance across {n_samples} samples."
        )

    else:
        top_n_summary = st.slider(
            "Number of top features",
            min_value=5,
            max_value=40,
            value=20,
            step=5,
            key="heatmap_summary_topn",
        )

        from visualization.heatmap import create_disease_summary_heatmap

        fig = create_disease_summary_heatmap(
            X_80, y, feature_names_80, class_names, short_names,
            top_n=top_n_summary,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "Shows mean expression per disease class for the "
            f"**{top_n_summary}** most variable biomarkers."
        )

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Visualization preprocessing is independent of model inference. "
    "Dimensionality reduction (PCA, t-SNE, UMAP) is performed on the "
    "preprocessed data for exploratory analysis only and does not affect "
    "the production prediction pipeline."
)
