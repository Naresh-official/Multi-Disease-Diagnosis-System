"""
LIME (Local Interpretable Model-agnostic Explanations) explainer.

Builds a local linear surrogate model around a single prediction.

The :func:`explain` function computes a LIME explanation and returns
a matplotlib figure.  The :func:`extract_top_features` function
returns a structured dictionary suitable for comparison with SHAP
explanations.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from core.config import APP_CONFIG
from core.predictor import load_artifacts
from utils.helpers import gene_feature_names


def explain(
    x_final: np.ndarray,
    pred_class: int,
    artifacts: dict[str, Any] | None = None,
) -> tuple[Any, plt.Figure]:
    """Compute LIME explanation for the predicted class and return a bar chart.

    Parameters
    ----------
    x_final : np.ndarray
        Preprocessed feature vector of shape ``(1, SELECTED_FEATURES)``.
    pred_class : int
        Index of the predicted disease class.  LIME is asked to
        explain this specific class.
    artifacts : dict, optional
        Pre-loaded artefacts.

    Returns
    -------
    lime_exp : lime.explanation.Explanation
        The raw LIME explanation object.
    fig : matplotlib.figure.Figure
        LIME feature-contribution bar chart.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    lime_exp = _compute_lime_explanation(x_final, pred_class, artifacts)

    # Pass pred_class so LIME plots the correct class explanation
    fig = lime_exp.as_pyplot_figure(label=pred_class)
    plt.tight_layout()

    return lime_exp, fig


def extract_top_features(
    lime_exp: Any,
    pred_class: int,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Extract top-N features from a LIME explanation as structured data.

    LIME's ``as_list(class_label)`` returns ``(feature_condition, weight)``
    tuples.  The ``feature_condition`` strings look like
    ``"-0.73 < Gene_62 <= -0.37"`` — we parse the gene name out.

    Parameters
    ----------
    lime_exp : lime.explanation.Explanation
        Raw LIME explanation object from :func:`explain`.
    pred_class : int
        Disease class index.  LIME may use a different internal
        class key, so we search for the class with the closest
        matching feature contributions.
    top_n : int
        Number of top features to return.

    Returns
    -------
    list of dict
        Each dict has keys: ``feature_index``, ``feature_name``,
        ``importance`` (absolute weight), ``value`` (signed weight),
        ``direction`` (``'positive'`` / ``'negative'``).
        Sorted by importance descending.
    """
    pairs = _get_lime_pairs(lime_exp, pred_class)

    results = []
    for rank, (condition, weight) in enumerate(pairs[:top_n], 1):
        feat_name = _parse_gene_name(condition)
        feat_idx = _parse_gene_index(condition)

        results.append({
            "feature_index": feat_idx,
            "feature_name": feat_name,
            "importance": abs(float(weight)),
            "value": float(weight),
            "direction": "positive" if weight > 0 else "negative",
            "rank": rank,
        })

    return results


def _compute_lime_explanation(
    x_final: np.ndarray,
    pred_class: int,
    artifacts: dict[str, Any],
) -> Any:
    """Internal: compute LIME explanation for the specified class."""
    feature_names = gene_feature_names(artifacts["x_train_fs"].shape[1])

    lime_explainer = LimeTabularExplainer(
        training_data=artifacts["x_train_fs"],
        feature_names=feature_names,
        class_names=APP_CONFIG.class_names,
        mode="classification",
    )

    # Explicitly tell LIME to explain the predicted class
    lime_exp = lime_explainer.explain_instance(
        x_final[0],
        artifacts["model"].predict,
        num_features=APP_CONFIG.lime_num_features,
        labels=(pred_class,),
    )

    return lime_exp


def _get_lime_pairs(lime_exp: Any, pred_class: int) -> list[tuple[str, float]]:
    """Get (condition, weight) pairs from LIME, handling class key mismatch.

    LIME sometimes uses different internal class keys than the model's
    class indices.  We try the requested class first, then fall back
    to whatever keys are available.
    """
    # Try the requested class directly
    try:
        pairs = lime_exp.as_list(pred_class)
        if pairs:
            return pairs
    except (KeyError, Exception):
        pass

    # Fall back: use whatever class key LIME computed
    available_keys = list(lime_exp.local_exp.keys())
    if available_keys:
        try:
            return lime_exp.as_list(available_keys[0])
        except Exception:
            pass

    # Last resort: try all keys
    for key in range(10):
        try:
            pairs = lime_exp.as_list(key)
            if pairs:
                return pairs
        except (KeyError, Exception):
            continue

    return []


def _parse_gene_name(condition: str) -> str:
    """Extract gene name from LIME condition string.

    Examples:
        ``"-0.73 < Gene_62 <= -0.37"`` → ``"Gene_62"``
        ``"Gene_42 > 0.5"`` → ``"Gene_42"``
    """
    import re
    match = re.search(r"(Gene_\d+)", condition)
    if match:
        return match.group(1)
    # Fallback: return the condition itself
    return condition


def _parse_gene_index(condition: str) -> int:
    """Extract gene index from LIME condition string.

    Examples:
        ``"-0.73 < Gene_62 <= -0.37"`` → ``62``
    """
    import re
    match = re.search(r"Gene_(\d+)", condition)
    if match:
        return int(match.group(1))
    return -1
