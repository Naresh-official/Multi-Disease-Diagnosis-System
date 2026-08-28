"""
Biomarker analyzer.

Aggregates SHAP values across the evaluation dataset to produce
global and per-disease importance rankings.  Caches results to
``data/evaluation/biomarker_shap.json`` so expensive SHAP
computations are not repeated on every Streamlit rerun.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import shap

from core.config import APP_CONFIG
from core.predictor import load_artifacts
from biomarkers.ranking import compute_ranking, RankingResult

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
BIOMARKER_CACHE = EVAL_DIR / "biomarker_shap.json"


def _get_probe_ids() -> list[str]:
    """Get the 80 selected Affymetrix probe IDs."""
    from utils.helpers import get_expected_feature_names

    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    all_names = get_expected_feature_names()
    pso_names = [all_names[i] for i in pso_features]
    final_names = [pso_names[i] for i in aco_features]

    # Strip surrounding quotes
    return [name.strip('"') for name in final_names]


def compute_shap_values(
    X_eval: np.ndarray,
    artifacts: dict[str, Any] | None = None,
    max_samples: int | None = None,
) -> np.ndarray:
    """Compute SHAP values for evaluation samples.

    Parameters
    ----------
    X_eval : np.ndarray of shape (n_samples, 80)
        Preprocessed evaluation features.
    artifacts : dict, optional
        Pre-loaded model artefacts.
    max_samples : int, optional
        Limit number of samples (for speed).  None = all.

    Returns
    -------
    np.ndarray of shape (n_samples, 80, n_classes)
        SHAP values for each sample, feature, and class.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    if max_samples is not None:
        X_eval = X_eval[:max_samples]

    background = artifacts["x_train_fs"][:APP_CONFIG.shap_background_samples]

    explainer = shap.KernelExplainer(
        artifacts["model"].predict,
        background,
    )

    shap_values = explainer.shap_values(X_eval)
    return shap_values


def compute_all_rankings(
    shap_values: np.ndarray,
    y_true: np.ndarray,
    probe_ids: list[str],
    class_names: list[str] | None = None,
) -> dict[str, RankingResult]:
    """Compute global and per-disease rankings from SHAP values.

    Parameters
    ----------
    shap_values : np.ndarray of shape (n_samples, 80, n_classes)
    y_true : np.ndarray of shape (n_samples,)
    probe_ids : list of str
    class_names : list of str, optional

    Returns
    -------
    dict mapping disease name to RankingResult, plus a 'Global' key.
    """
    if class_names is None:
        class_names = APP_CONFIG.class_names

    results: dict[str, RankingResult] = {}

    # Global ranking
    results["Global"] = compute_ranking(
        shap_values, y_true, probe_ids,
        disease_id=None, class_names=class_names,
    )

    # Per-disease rankings
    for disease_id, disease_name in APP_CONFIG.disease_map.items():
        results[disease_name] = compute_ranking(
            shap_values, y_true, probe_ids,
            disease_id=disease_id, class_names=class_names,
        )

    return results


def save_biomarker_results(
    shap_values: np.ndarray,
    rankings: dict[str, RankingResult],
) -> None:
    """Save SHAP values and rankings to disk for caching.

    Saves to ``data/evaluation/biomarker_shap.json``.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw SHAP values as numpy
    np.save(str(EVAL_DIR / "biomarker_shap_values.npy"), shap_values)

    # Save rankings as JSON
    rankings_json = {}
    for name, result in rankings.items():
        rankings_json[name] = {
            "disease_name": result.disease_name,
            "disease_id": result.disease_id,
            "total_samples": result.total_samples,
            "biomarkers": [
                {
                    "rank": b.rank,
                    "feature_index": b.feature_index,
                    "probe_id": b.probe_id,
                    "importance": b.importance,
                    "mean_shap": b.mean_shap,
                    "std_shap": b.std_shap,
                }
                for b in result.biomarkers
            ],
        }

    with open(BIOMARKER_CACHE, "w") as f:
        json.dump(rankings_json, f, indent=2)


def load_biomarker_results() -> tuple[np.ndarray | None, dict[str, RankingResult] | None]:
    """Load cached SHAP values and rankings from disk.

    Returns
    -------
    (shap_values, rankings) or (None, None) if not cached.
    """
    shap_path = EVAL_DIR / "biomarker_shap_values.npy"
    if not shap_path.exists() or not BIOMARKER_CACHE.exists():
        return None, None

    shap_values = np.load(str(shap_path))

    with open(BIOMARKER_CACHE) as f:
        rankings_json = json.load(f)

    rankings = {}
    for name, data in rankings_json.items():
        biomarkers = [
            type("BiomarkerRanking", (), b)()
            for b in data["biomarkers"]
        ]
        # Restore proper type
        from biomarkers.ranking import BiomarkerRanking, RankingResult
        biomarkers = [
            BiomarkerRanking(
                rank=b["rank"],
                feature_index=b["feature_index"],
                probe_id=b["probe_id"],
                importance=b["importance"],
                mean_shap=b["mean_shap"],
                std_shap=b["std_shap"],
            )
            for b in data["biomarkers"]
        ]
        rankings[name] = RankingResult(
            disease_name=data["disease_name"],
            disease_id=data["disease_id"],
            total_samples=data["total_samples"],
            biomarkers=biomarkers,
        )

    return shap_values, rankings


def get_feature_selection_journey() -> dict[str, Any]:
    """Return the feature selection journey information.

    Returns the actual numbers from the trained artifacts:
    total features → variance filtering → PSO → ACO → final biomarkers.
    """
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    from utils.helpers import get_expected_feature_names
    n_total = len(get_expected_feature_names())
    n_variance = int((selector.variances_ > selector.threshold).sum())
    n_pso = len(pso_features)
    n_aco = len(aco_features)

    return {
        "total_features": n_total,
        "after_variance": n_variance,
        "after_pso": n_pso,
        "after_aco": n_aco,
        "reduction_pct": round((1 - n_aco / n_total) * 100, 2),
    }


def get_feature_metadata(probe_ids: list[str]) -> list[dict[str, Any]]:
    """Return metadata for each probe ID.

    Since the identifiers are Affymetrix probe IDs (not gene symbols),
    no biological annotation is fabricated.  This returns the probe ID
    and its position in the feature selection pipeline.

    Returns
    -------
    list of dict with keys: probe_id, feature_type, note.
    """
    return [
        {
            "probe_id": pid,
            "feature_type": "Affymetrix probe ID",
            "note": (
                "This is an Affymetrix microarray probe identifier. "
                "Biological annotation (gene symbol, pathway) is not "
                "available without an external mapping database."
            ),
        }
        for pid in probe_ids
    ]
