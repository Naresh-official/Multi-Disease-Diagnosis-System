"""
PCA visualization for gene-expression data.

Projects high-dimensional gene-expression data into 2D while
preserving maximum variance.  Uses sklearn.decomposition.PCA
and renders interactive Plotly charts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


def run_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Run PCA on the feature matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_components : int
        Number of principal components (default 2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pca : fitted PCA object
    X_pca : np.ndarray of shape (n_samples, n_components)
        Transformed coordinates.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def plot_pca_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    short_names: list[str],
    pca_model: PCA,
    dataset_labels: list[str] | None = None,
    color_by: str = "disease",
) -> go.Figure:
    """Create an interactive 2D PCA scatter plot.

    Parameters
    ----------
    X_pca : np.ndarray of shape (n_samples, 2)
        PCA coordinates.
    y : np.ndarray
        Integer class labels.
    class_names : list of str
        Full disease names.
    short_names : list of str
        Short disease names for legend.
    pca_model : PCA
        Fitted PCA model (for explained variance info).
    dataset_labels : list of str, optional
        GEO dataset source for each sample.
    color_by : str
        'disease' or 'dataset' — what to color points by.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    var_ratio = pca_model.explained_variance_ratio_

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Disease": [short_names[int(label)] for label in y],
        "Full Name": [class_names[int(label)] for label in y],
        "Sample Index": range(len(y)),
    })

    if dataset_labels is not None:
        df["Dataset"] = dataset_labels

    color_col = "Dataset" if color_by == "dataset" and "Dataset" in df.columns else "Disease"

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=color_col,
        hover_data=["Full Name", "Sample Index"],
        title=f"PCA Projection (Explained Variance: {var_ratio.sum():.1%})",
        labels={
            "PC1": f"PC1 ({var_ratio[0]:.1%})",
            "PC2": f"PC2 ({var_ratio[1]:.1%})",
        },
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


def plot_explained_variance(
    pca_model: PCA,
    max_components: int = 30,
) -> go.Figure:
    """Create a bar chart of explained variance per component.

    Parameters
    ----------
    pca_model : PCA
        Fitted PCA model.
    max_components : int
        Maximum number of components to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_show = min(len(pca_model.explained_variance_ratio_), max_components)
    var = pca_model.explained_variance_ratio_[:n_show]
    cumulative = np.cumsum(var)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=list(range(1, n_show + 1)),
            y=var,
            name="Individual",
            marker_color="steelblue",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, n_show + 1)),
            y=cumulative,
            name="Cumulative",
            mode="lines+markers",
            marker_color="coral",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Principal Component")
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance", secondary_y=True)
    fig.update_layout(
        title="PCA Explained Variance",
        template="plotly_white",
        width=800,
        height=400,
    )

    return fig
