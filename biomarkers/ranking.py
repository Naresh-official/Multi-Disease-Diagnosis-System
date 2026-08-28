"""
Biomarker ranking utilities.

Computes global and disease-specific feature importance rankings
from SHAP values.  Importance is measured by mean |SHAP| across
samples; the original signed SHAP value is preserved so that
direction of contribution (positive/negative) is always available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class BiomarkerRanking:
    """A single biomarker's ranking information.

    Attributes
    ----------
    rank : int
        1-based importance rank (1 = most important).
    feature_index : int
        Index within the 80-feature vector.
    probe_id : str
        Affymetrix probe identifier (e.g. ``'219170_at'``).
    importance : float
        Mean absolute SHAP value across the relevant samples.
    mean_shap : float
        Mean signed SHAP value (positive → pushes toward the class).
    std_shap : float
        Standard deviation of signed SHAP values.
    """

    rank: int
    feature_index: int
    probe_id: str
    importance: float
    mean_shap: float
    std_shap: float


@dataclass
class RankingResult:
    """A complete ranking of biomarkers.

    Attributes
    ----------
    disease_name : str
        Which disease this ranking is for, or ``'Global'``.
    disease_id : int or None
        Class index, or ``None`` for global rankings.
    total_samples : int
        Number of samples used to compute this ranking.
    biomarkers : list of BiomarkerRanking
        Ranked list sorted by importance (descending).
    """

    disease_name: str
    disease_id: int | None
    total_samples: int
    biomarkers: list[BiomarkerRanking] = field(default_factory=list)


# ── Ranking computation ─────────────────────────────────────────────────────


def compute_ranking(
    shap_values_all: np.ndarray,
    y_true: np.ndarray,
    probe_ids: list[str],
    disease_id: int | None = None,
    class_names: list[str] | None = None,
) -> RankingResult:
    """Compute a biomarker importance ranking from SHAP values.

    Parameters
    ----------
    shap_values_all : np.ndarray of shape (n_samples, n_features, n_classes)
        SHAP values for every sample, feature, and class.
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth integer labels for each sample.
    probe_ids : list of str of length n_features
        Affymetrix probe IDs for the 80 selected features.
    disease_id : int or None
        If given, compute ranking for this specific disease class.
        If ``None``, compute the global ranking across all classes.
    class_names : list of str, optional
        Human-readable disease names.

    Returns
    -------
    RankingResult
        A fully populated ranking object.
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(shap_values_all.shape[2])]

    n_features = shap_values_all.shape[1]

    if disease_id is not None:
        # Disease-specific: average absolute SHAP for this class across
        # samples that actually belong to this disease
        mask = y_true == disease_id
        if mask.sum() == 0:
            # Fallback: use all samples if disease not present
            mask = np.ones(len(y_true), dtype=bool)
        subset = shap_values_all[mask, :, disease_id]  # (n_subset, 80)
        disease_name = class_names[disease_id]
    else:
        # Global: average max-absolute-class SHAP per feature across all samples
        # For each sample, take the SHAP value for the predicted (argmax) class
        max_abs = np.abs(shap_values_all).max(axis=2)  # (n_samples, 80)
        subset = max_abs
        disease_name = "Global"
        disease_id = None

    mean_abs = np.abs(subset).mean(axis=0)  # (80,)
    mean_signed = subset.mean(axis=0) if disease_id is not None else np.zeros(n_features)
    std_signed = subset.std(axis=0) if disease_id is not None else np.zeros(n_features)

    # Sort by importance (descending)
    order = np.argsort(mean_abs)[::-1]

    biomarkers = []
    for rank, idx in enumerate(order, 1):
        biomarkers.append(BiomarkerRanking(
            rank=rank,
            feature_index=int(idx),
            probe_id=probe_ids[idx],
            importance=float(mean_abs[idx]),
            mean_shap=float(mean_signed[idx]),
            std_shap=float(std_signed[idx]),
        ))

    return RankingResult(
        disease_name=disease_name,
        disease_id=disease_id,
        total_samples=int(mask.sum()) if disease_id is not None else int(len(y_true)),
        biomarkers=biomarkers,
    )


def ranking_to_dataframe(result: RankingResult) -> pd.DataFrame:
    """Convert a RankingResult to a tidy DataFrame.

    Returns
    -------
    pd.DataFrame with columns: rank, feature_index, probe_id,
    importance, mean_shap, std_shap, direction.
    """
    rows = []
    for b in result.biomarkers:
        direction = "Positive" if b.mean_shap > 0 else "Negative" if b.mean_shap < 0 else "Neutral"
        rows.append({
            "Rank": b.rank,
            "Feature Index": b.feature_index,
            "Probe ID": b.probe_id,
            "Importance (|SHAP|)": round(b.importance, 6),
            "Mean SHAP": round(b.mean_shap, 6),
            "Std SHAP": round(b.std_shap, 6),
            "Direction": direction,
        })
    return pd.DataFrame(rows)


def get_top_n(result: RankingResult, n: int) -> list[BiomarkerRanking]:
    """Return the top-N biomarkers from a ranking."""
    return result.biomarkers[:n]
