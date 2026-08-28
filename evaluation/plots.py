"""
Evaluation visualisation — Plotly interactive charts.

Generates:
- Confusion matrix heatmap
- ROC curves (one-vs-rest, per class + macro average)
- Precision-Recall curves
- Per-class performance bar chart
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
) -> go.Figure:
    """Plot an interactive confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)``.
    class_names : list[str]
        Disease names for axis labels.

    Returns
    -------
    go.Figure
    """
    # Normalised row-wise (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    text_combined = np.array([
        [f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(len(class_names))]
        for i in range(len(class_names))
    ])

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=text_combined,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale="Blues",
        colorbar=dict(title="Recall"),
        hovertemplate=(
            "True: %{y}<br>Predicted: %{x}<br>"
            "Count: %{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Confusion Matrix (row-normalised)",
        xaxis_title="Predicted Disease",
        yaxis_title="True Disease",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=500,
        width=600,
    )

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> go.Figure:
    """Plot per-class ROC curves with macro-average AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n, n_classes)``.
    class_names : list[str]

    Returns
    -------
    go.Figure
    """
    n_classes = len(class_names)
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig = go.Figure()

    # Per-class ROC (one-vs-rest)
    all_fpr = []
    all_tpr = []
    all_auc = []

    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        if y_true_bin.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc_val)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{class_names[i]} (AUC = {roc_auc_val:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    # Macro-average ROC
    if all_fpr:
        # Interpolate to common FPR grid
        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(all_fpr, all_tpr):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(all_fpr)
        macro_auc = auc(mean_fpr, mean_tpr)

        fig.add_trace(go.Scatter(
            x=mean_fpr, y=mean_tpr,
            mode="lines",
            name=f"Macro-average (AUC = {macro_auc:.3f})",
            line=dict(color="#000000", width=3, dash="dash"),
        ))

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random classifier",
        line=dict(color="gray", width=1, dash="dot"),
        showlegend=False,
    ))

    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=550,
        legend=dict(x=0.55, y=0.05),
    )

    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> go.Figure:
    """Plot per-class Precision-Recall curves.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n, n_classes)``.
    class_names : list[str]

    Returns
    -------
    go.Figure
    """
    n_classes = len(class_names)
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig = go.Figure()

    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        if y_true_bin.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, i])
        pr_auc = auc(recall, precision)

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{class_names[i]} (AUC = {pr_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    # Baseline (prevalence)
    n = len(y_true)
    prevalence = {i: (y_true == i).sum() / n for i in range(n_classes)}
    avg_prevalence = np.mean(list(prevalence.values()))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[avg_prevalence, avg_prevalence],
        mode="lines",
        name=f"Baseline (prevalence ≈ {avg_prevalence:.2f})",
        line=dict(color="gray", width=1, dash="dot"),
        showlegend=True,
    ))

    fig.update_layout(
        title="Precision-Recall Curves (One-vs-Rest)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1.02]),
        yaxis=dict(range=[0, 1.05]),
        height=550,
        legend=dict(x=0.55, y=0.05),
    )

    return fig


def plot_per_class_performance(
    per_class_metrics: list[dict[str, Any]],
) -> go.Figure:
    """Plot a grouped bar chart of per-class precision, recall, F1.

    Parameters
    ----------
    per_class_metrics : list[dict]
        Output of :func:`evaluation.metrics.compute_per_class_metrics`.

    Returns
    -------
    go.Figure
    """
    class_names = [m["class_name"] for m in per_class_metrics]
    precisions = [m["precision"] for m in per_class_metrics]
    recalls = [m["recall"] for m in per_class_metrics]
    f1s = [m["f1"] for m in per_class_metrics]
    supports = [m["support"] for m in per_class_metrics]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Precision / Recall / F1", "Support (sample count)"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=class_names, y=precisions,
        name="Precision", marker_color="#1f77b4",
        text=[f"{v:.2f}" for v in precisions],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=class_names, y=recalls,
        name="Recall", marker_color="#ff7f0e",
        text=[f"{v:.2f}" for v in recalls],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=class_names, y=f1s,
        name="F1-Score", marker_color="#2ca02c",
        text=[f"{v:.2f}" for v in f1s],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=class_names, y=supports,
        name="Support", marker_color="#9467bd",
        text=[str(s) for s in supports],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        barmode="group",
        height=450,
        yaxis=dict(title="Score", range=[0, 1.15]),
        yaxis2=dict(title="Count"),
        legend=dict(x=0.35, y=1.12, orientation="h"),
    )

    return fig
