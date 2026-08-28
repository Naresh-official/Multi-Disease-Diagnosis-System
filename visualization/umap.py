"""
UMAP visualization for gene-expression data.

Uses the umap-learn library for uniform manifold approximation
and projection.  UMAP preserves both local and global structure
better than t-SNE in many cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP on the feature matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_neighbors : int
        Number of neighbors for local connectivity.
    min_dist : float
        Minimum distance between embedded points.
    n_components : int
        Number of output dimensions (default 2).
    metric : str
        Distance metric to use.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components)
        UMAP coordinates.
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap


def plot_umap_scatter(
    X_umap: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    short_names: list[str],
    n_neighbors: int,
    min_dist: float,
    dataset_labels: list[str] | None = None,
    color_by: str = "disease",
) -> go.Figure:
    """Create an interactive UMAP scatter plot.

    Parameters
    ----------
    X_umap : np.ndarray of shape (n_samples, 2)
        UMAP coordinates.
    y : np.ndarray
        Integer class labels.
    class_names : list of str
        Full disease names.
    short_names : list of str
        Short names for legend.
    n_neighbors : int
        n_neighbors used (shown in title).
    min_dist : float
        min_dist used (shown in title).
    dataset_labels : list of str, optional
        GEO dataset source.
    color_by : str
        'disease' or 'dataset'.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pd.DataFrame({
        "UMAP 1": X_umap[:, 0],
        "UMAP 2": X_umap[:, 1],
        "Disease": [short_names[int(label)] for label in y],
        "Full Name": [class_names[int(label)] for label in y],
        "Sample Index": range(len(y)),
    })

    if dataset_labels is not None:
        df["Dataset"] = dataset_labels

    color_col = "Dataset" if color_by == "dataset" and "Dataset" in df.columns else "Disease"

    fig = px.scatter(
        df,
        x="UMAP 1",
        y="UMAP 2",
        color=color_col,
        hover_data=["Full Name", "Sample Index"],
        title=f"UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})",
        width=800,
        height=600,
        opacity=0.7,
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_layout(
        legend_title_text=color_col,
        template="plotly_white",
    )

    return fig
