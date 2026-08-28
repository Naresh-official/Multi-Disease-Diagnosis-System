"""
t-SNE visualization for gene-expression data.

Uses sklearn.manifold.TSNE for non-linear dimensionality reduction.
t-SNE is computationally expensive — results should be cached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def run_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE on the feature matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_components : int
        Number of output dimensions (default 2).
    perplexity : float
        Perplexity parameter (typical range 5-50).
    learning_rate : float or 'auto'
        Learning rate for t-SNE optimization.
    max_iter : int
        Number of optimization iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components)
        t-SNE coordinates.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        init="pca",
        method="exact" if X.shape[1] > 100 else "barnes_hut",
    )
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def plot_tsne_scatter(
    X_tsne: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    short_names: list[str],
    perplexity: float,
    max_iter: int,
    dataset_labels: list[str] | None = None,
    color_by: str = "disease",
) -> go.Figure:
    """Create an interactive t-SNE scatter plot.

    Parameters
    ----------
    X_tsne : np.ndarray of shape (n_samples, 2)
        t-SNE coordinates.
    y : np.ndarray
        Integer class labels.
    class_names : list of str
        Full disease names.
    short_names : list of str
        Short names for legend.
    perplexity : float
        Perplexity used (shown in title).
    max_iter : int
        Iterations used (shown in title).
    dataset_labels : list of str, optional
        GEO dataset source.
    color_by : str
        'disease' or 'dataset'.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pd.DataFrame({
        "t-SNE 1": X_tsne[:, 0],
        "t-SNE 2": X_tsne[:, 1],
        "Disease": [short_names[int(label)] for label in y],
        "Full Name": [class_names[int(label)] for label in y],
        "Sample Index": range(len(y)),
    })

    if dataset_labels is not None:
        df["Dataset"] = dataset_labels

    color_col = "Dataset" if color_by == "dataset" and "Dataset" in df.columns else "Disease"

    fig = px.scatter(
        df,
        x="t-SNE 1",
        y="t-SNE 2",
        color=color_col,
        hover_data=["Full Name", "Sample Index"],
        title=f"t-SNE Projection (perplexity={perplexity}, iter={max_iter})",
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
