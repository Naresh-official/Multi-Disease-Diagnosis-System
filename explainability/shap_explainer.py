"""
SHAP (SHapley Additive exPlanations) explainer.

Uses a ``KernelExplainer`` initialised on a background subset of the
training data.  The public :func:`explain` function returns both the
raw shap values and a ready-to-plot matplotlib figure.

The :func:`extract_top_features` function returns a structured
dictionary suitable for comparison with LIME explanations.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap

from core.config import APP_CONFIG
from core.predictor import load_artifacts
from utils.helpers import gene_feature_names


def explain(
    x_final: np.ndarray,
    pred_class: int,
    artifacts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, plt.Figure]:
    """Compute SHAP values for the predicted class and return a bar chart.

    Parameters
    ----------
    x_final : np.ndarray
        Preprocessed feature vector of shape ``(1, SELECTED_FEATURES)``.
    pred_class : int
        Index of the predicted disease class.
    artifacts : dict, optional
        Pre-loaded artefacts (avoids a redundant cache lookup when the
        caller already has them).

    Returns
    -------
    shap_values : np.ndarray
        Raw SHAP values array.
    fig : matplotlib.figure.Figure
        SHAP summary bar chart.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    shap_values = _compute_shap_values(x_final, artifacts)

    shap_sample = shap_values[0, :, pred_class]

    feature_names = gene_feature_names(x_final.shape[1])

    fig = plt.figure()
    shap.summary_plot(
        shap_sample.reshape(1, -1),
        x_final,
        feature_names=feature_names,
        plot_type="bar",
        max_display=APP_CONFIG.shap_max_display,
        show=False,
    )
    plt.tight_layout()

    return shap_values, fig


def extract_top_features(
    shap_values: np.ndarray,
    pred_class: int,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Extract top-N features from SHAP values as structured data.

    Parameters
    ----------
    shap_values : np.ndarray of shape ``(1, n_features, n_classes)``
        Raw SHAP values from :func:`explain`.
    pred_class : int
        Disease class index to extract features for.
    top_n : int
        Number of top features to return.

    Returns
    -------
    list of dict
        Each dict has keys: ``feature_index``, ``feature_name``,
        ``importance`` (absolute SHAP), ``value`` (signed SHAP),
        ``direction`` (``'positive'`` / ``'negative'``).
        Sorted by importance descending.
    """
    # SHAP values for this class: (n_features,)
    class_shap = shap_values[0, :, pred_class]

    abs_vals = np.abs(class_shap)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        val = float(class_shap[idx])
        results.append({
            "feature_index": int(idx),
            "feature_name": f"Gene_{idx}",
            "importance": float(abs_vals[idx]),
            "value": val,
            "direction": "positive" if val > 0 else "negative",
            "rank": rank,
        })

    return results


def _compute_shap_values(
    x_final: np.ndarray,
    artifacts: dict[str, Any],
) -> np.ndarray:
    """Internal: compute SHAP values using KernelExplainer.

    Returns np.ndarray of shape ``(1, n_features, n_classes)``.
    """
    background = artifacts["x_train_fs"][:APP_CONFIG.shap_background_samples]

    explainer = shap.KernelExplainer(
        artifacts["model"].predict,
        background,
    )
    return explainer.shap_values(x_final)
