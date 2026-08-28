"""
Biomarker heatmap visualization for gene-expression data.

Renders interactive heatmaps of the selected biomarker features
(80 features after PSO + ACO) rather than attempting to display
all 10,571 original probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_biomarker_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    short_names: list[str],
    top_n: int | None = None,
    sort_by_variance: bool = True,
) -> go.Figure:
    """Create an interactive gene-expression heatmap of biomarkers.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Processed feature matrix (80 features).
    y : np.ndarray
        Integer class labels.
    feature_names : list of str
        Names of the biomarker features (e.g. probe IDs).
    class_names : list of str
        Full disease names.
    short_names : list of str
        Short disease names for y-axis labels.
    top_n : int, optional
        If set, show only the top N most variable features.
        If None, show all 80.
    sort_by_variance : bool
        Whether to sort features by variance (most variable first).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["Disease"] = [short_names[int(label)] for label in y]

    # Sort samples by disease class
    df = df.sort_values("Disease").reset_index(drop=True)
    sorted_y = y[df.index.values]

    # Select features
    feature_cols = list(feature_names)
    if sort_by_variance:
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = list(variances.index)

    if top_n is not None and top_n < len(feature_cols):
        feature_cols = feature_cols[:top_n]

    # Extract heatmap data
    heatmap_data = df[feature_cols].values

    # Z-score normalize per feature for better visual contrast
    means = heatmap_data.mean(axis=0)
    stds = heatmap_data.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    heatmap_z = (heatmap_data - means) / stds

    # Build y-axis labels: sample index + disease
    sample_labels = [
        f"{short_names[int(label)]} #{i}"
        for i, label in enumerate(sorted_y)
    ]

    # Build color-bar tick labels for diseases
    disease_counts = {}
    for label in sorted_y:
        name = short_names[int(label)]
        disease_counts[name] = disease_counts.get(name, 0) + 1

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_z.T,
            x=sample_labels,
            y=[name[:20] for name in feature_cols],
            colorscale="RdBu_r",
            reversescale=True,
            colorbar=dict(title="Z-score"),
            hovertemplate=(
                "Feature: %{y}<br>"
                "Sample: %{x}<br>"
                "Z-score: %{z:.2f}<extra></extra>"
            ),
        )
    )

    title = "Biomarker Expression Heatmap"
    if top_n is not None:
        title += f" (Top {top_n} of {len(feature_names)} features)"
    else:
        title += f" ({len(feature_names)} features)"

    fig.update_layout(
        title=title,
        xaxis_title="Samples (sorted by disease)",
        yaxis_title="Biomarker Features",
        width=1000,
        height=max(400, len(feature_cols) * 8 + 100),
        template="plotly_white",
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
    )

    return fig


def create_disease_summary_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    short_names: list[str],
    top_n: int = 20,
) -> go.Figure:
    """Create a summary heatmap showing mean expression per disease class.

    This aggregates samples by disease class and shows the mean
    expression for the top N most variable features.

    Parameters
    ----------
    X : np.ndarray
        Processed feature matrix.
    y : np.ndarray
        Integer class labels.
    feature_names : list of str
        Biomarker feature names.
    class_names : list of str
        Full disease names.
    short_names : list of str
        Short disease names.
    top_n : int
        Number of top variable features to show.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["Label"] = y

    # Compute mean per class
    class_means = df.groupby("Label")[feature_names].mean()

    # Sort by overall variance
    variances = class_means.var().sort_values(ascending=False)
    top_features = list(variances.index[:top_n])

    data = class_means[top_features].values

    # Z-score normalize
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds == 0] = 1.0
    z_data = (data - means) / stds

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data.T,
            x=[short_names[i] for i in class_means.index],
            y=[name[:20] for name in top_features],
            colorscale="RdBu_r",
            reversescale=True,
            colorbar=dict(title="Z-score"),
            hovertemplate=(
                "Feature: %{y}<br>"
                "Disease: %{x}<br>"
                "Z-score: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Mean Expression by Disease Class (Top {top_n} Variable Features)",
        xaxis_title="Disease Class",
        yaxis_title="Biomarker Features",
        width=800,
        height=max(400, top_n * 10 + 100),
        template="plotly_white",
        yaxis=dict(tickfont=dict(size=9)),
    )

    return fig
