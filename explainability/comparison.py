"""
XAI comparison utilities.

Compares SHAP and LIME feature explanations for the same prediction.
Uses Spearman rank correlation and feature overlap metrics.

IMPORTANT: Agreement between SHAP and LIME does NOT prove that
either explanation is "correct."  It only indicates that two
different explanation methods assign similar importance rankings
to features for a given prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ComparisonResult:
    """Structured result of a SHAP vs LIME comparison.

    Attributes
    ----------
    common_features : list of dict
        Features present in both SHAP and LIME top-N lists.
    shap_only : list of dict
        Features only in SHAP's top-N.
    lime_only : list of dict
        Features only in LIME's top-N.
    overlap_count : int
        Number of features in both rankings.
    overlap_pct : float
        Percentage of top-N features shared (0–100).
    spearman_rho : float or None
        Spearman rank correlation coefficient, or ``None`` if
        insufficient common features.
    spearman_p : float or None
        p-value for the correlation, or ``None``.
    interpretation : str
        Human-readable interpretation of the agreement level.
    comparison_df : pd.DataFrame
        Side-by-side comparison table.
    shap_description : str
        Brief description of how SHAP works.
    lime_description : str
        Brief description of how LIME works.
    """

    common_features: list[dict] = field(default_factory=list)
    shap_only: list[dict] = field(default_factory=list)
    lime_only: list[dict] = field(default_factory=list)
    overlap_count: int = 0
    overlap_pct: float = 0.0
    spearman_rho: float | None = None
    spearman_p: float | None = None
    interpretation: str = ""
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_description: str = ""
    lime_description: str = ""


def compare_explanations(
    shap_features: list[dict],
    lime_features: list[dict],
    top_n: int = 10,
) -> ComparisonResult:
    """Compare SHAP and LIME feature explanations.

    Parameters
    ----------
    shap_features : list of dict
        Top features from SHAP (output of
        :func:`explainability.shap_explainer.extract_top_features`).
    lime_features : list of dict
        Top features from LIME (output of
        :func:`explainability.lime_explainer.extract_top_features`).
    top_n : int
        Number of top features used in the comparison.

    Returns
    -------
    ComparisonResult
        Full comparison with metrics, dataframes, and interpretation.
    """
    # Build lookup dicts keyed by feature_index
    shap_dict = {f["feature_index"]: f for f in shap_features[:top_n]}
    lime_dict = {f["feature_index"]: f for f in lime_features[:top_n]}

    shap_indices = set(shap_dict.keys())
    lime_indices = set(lime_dict.keys())

    common_indices = shap_indices & lime_indices
    shap_only_indices = shap_indices - lime_indices
    lime_only_indices = lime_indices - shap_indices

    # ── Common features ──────────────────────────────────────────────────
    common_features = []
    for idx in sorted(common_indices):
        s = shap_dict[idx]
        l = lime_dict[idx]
        common_features.append({
            "feature_index": idx,
            "feature_name": s["feature_name"],
            "shap_importance": s["importance"],
            "shap_value": s["value"],
            "shap_direction": s["direction"],
            "lime_importance": l["importance"],
            "lime_value": l["value"],
            "lime_direction": l["direction"],
            "shap_rank": s["rank"],
            "lime_rank": l["rank"],
            "rank_diff": abs(s["rank"] - l["rank"]),
        })

    # ── SHAP-only features ──────────────────────────────────────────────
    shap_only = []
    for idx in sorted(shap_only_indices):
        s = shap_dict[idx]
        shap_only.append({
            "feature_index": idx,
            "feature_name": s["feature_name"],
            "shap_importance": s["importance"],
            "shap_direction": s["direction"],
            "shap_rank": s["rank"],
        })

    # ── LIME-only features ──────────────────────────────────────────────
    lime_only = []
    for idx in sorted(lime_only_indices):
        l = lime_dict[idx]
        lime_only.append({
            "feature_index": idx,
            "feature_name": l["feature_name"],
            "lime_importance": l["importance"],
            "lime_direction": l["direction"],
            "lime_rank": l["rank"],
        })

    # ── Overlap metrics ─────────────────────────────────────────────────
    overlap_count = len(common_indices)
    overlap_pct = (overlap_count / top_n * 100) if top_n > 0 else 0.0

    # ── Spearman correlation ────────────────────────────────────────────
    spearman_rho: float | None = None
    spearman_p: float | None = None

    if len(common_indices) >= 3:
        # Build rank arrays for common features
        shap_ranks = [shap_dict[idx]["rank"] for idx in sorted(common_indices)]
        lime_ranks = [lime_dict[idx]["rank"] for idx in sorted(common_indices)]

        corr, p_val = spearmanr(shap_ranks, lime_ranks)
        spearman_rho = float(corr)
        spearman_p = float(p_val)

    # ── Interpretation ──────────────────────────────────────────────────
    interpretation = _interpret(spearman_rho, overlap_pct, overlap_count)

    # ── Comparison DataFrame ────────────────────────────────────────────
    comparison_df = _build_comparison_dataframe(
        shap_features[:top_n],
        lime_features[:top_n],
        common_indices,
    )

    return ComparisonResult(
        common_features=common_features,
        shap_only=shap_only,
        lime_only=lime_only,
        overlap_count=overlap_count,
        overlap_pct=overlap_pct,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        interpretation=interpretation,
        comparison_df=comparison_df,
        shap_description=_shap_description(),
        lime_description=_lime_description(),
    )


def _build_comparison_dataframe(
    shap_features: list[dict],
    lime_features: list[dict],
    common_indices: set[int],
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame."""
    shap_dict = {f["feature_index"]: f for f in shap_features}
    lime_dict = {f["feature_index"]: f for f in lime_features}

    all_indices = sorted(set(shap_dict.keys()) | set(lime_dict.keys()))

    rows = []
    for idx in all_indices:
        s = shap_dict.get(idx)
        l = lime_dict.get(idx)

        in_both = idx in common_indices

        rows.append({
            "Feature": f"Gene_{idx}" if idx >= 0 else "Unknown",
            "In Both": "✓" if in_both else "",
            "SHAP Rank": s["rank"] if s else "—",
            "SHAP Importance": f"{s['importance']:.6f}" if s else "—",
            "SHAP Direction": s["direction"] if s else "—",
            "LIME Rank": l["rank"] if l else "—",
            "LIME Importance": f"{l['importance']:.6f}" if l else "—",
            "LIME Direction": l["direction"] if l else "—",
        })

    return pd.DataFrame(rows)


def _interpret(
    rho: float | None,
    overlap_pct: float,
    overlap_count: int,
) -> str:
    """Generate a scientifically cautious interpretation string."""
    parts = []

    if rho is not None:
        abs_rho = abs(rho)
        if abs_rho >= 0.7:
            parts.append(
                f"**Strong agreement** in feature ranking "
                f"(Spearman ρ = {rho:.3f})."
            )
        elif abs_rho >= 0.4:
            parts.append(
                f"**Moderate agreement** in feature ranking "
                f"(Spearman ρ = {rho:.3f})."
            )
        else:
            parts.append(
                f"**Low agreement** in feature ranking "
                f"(Spearman ρ = {rho:.3f})."
            )
    else:
        parts.append(
            "Insufficient common features for reliable rank correlation "
            "(need ≥ 3 shared features)."
        )

    parts.append(
        f"{overlap_count} of the top features appear in both "
        f"SHAP and LIME rankings ({overlap_pct:.0f}% overlap)."
    )

    parts.append(
        "⚠️ Agreement between methods does **not** prove that either "
        "explanation is correct — it indicates concordance between two "
        "different mathematical approaches to feature attribution."
    )

    return " ".join(parts)


def _shap_description() -> str:
    return (
        "**SHAP** (SHapley Additive exPlanations) computes feature "
        "contributions based on the Shapley-value framework from "
        "cooperative game theory.  Each feature's contribution is "
        "determined by its marginal contribution across all possible "
        "feature coalitions.  SHAP provides **globally consistent** "
        "attributions — the sum of all feature contributions equals "
        "the prediction difference from the expected output."
    )


def _lime_description() -> str:
    return (
        "**LIME** (Local Interpretable Model-agnostic Explanations) "
        "builds a **local surrogate model** — a simple linear model "
        "fitted around the specific prediction being explained.  It "
        "perturbs the input and observes how the model's output "
        "changes, then fits a weighted linear regression to approximate "
        "the model's behavior locally.  LIME provides **locally faithful** "
        "but not globally consistent explanations."
    )
