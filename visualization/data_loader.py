"""
Data loader for visualization modules.

Loads the evaluation dataset from GEO Series Matrix files and
applies the standard preprocessing pipeline to produce feature
matrices suitable for PCA, t-SNE, UMAP, and heatmap visualizations.

IMPORTANT: This uses the EXACT SAME preprocessing pipeline as
inference (StandardScaler → VarianceThreshold → PSO → ACO). It
does NOT modify or retrain any model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from core.config import APP_CONFIG

VIZ_DIR = Path(__file__).resolve().parent.parent / "data" / "visualizations"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"


def get_selected_feature_names() -> list[str]:
    """Return the names of the 80 selected biomarker features.

    Traces through the PSO and ACO feature selection to identify
    which original probe IDs were ultimately selected by the
    hybrid pipeline.

    Returns
    -------
    list of str
        The 80 probe IDs (e.g. '"219170_at"') in the order the
        model expects them.
    """
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    from utils.helpers import get_expected_feature_names
    all_names = get_expected_feature_names()  # 10571 probe IDs
    pso_names = [all_names[i] for i in pso_features]  # 500 names
    final_names = [pso_names[i] for i in aco_features]  # 80 names

    return final_names


def get_selected_feature_names_clean() -> list[str]:
    """Return clean (no quotes) names of the 80 selected biomarkers."""
    return [name.strip('"') for name in get_selected_feature_names()]


def load_raw_and_labels() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Load all evaluation samples with their labels.

    Reuses the evaluator's GSE parsing logic.

    Returns
    -------
    X_raw : pd.DataFrame
        Raw expression matrix (n_samples, 10571).
    y : np.ndarray
        Integer class labels (n_samples,).
    class_names : list of str
        Human-readable disease names.
    dataset_labels : list of str
        Which GEO dataset each sample came from.
    """
    from evaluation.evaluator import load_evaluation_data

    eval_data = load_evaluation_data()
    return (
        eval_data["X_raw"],
        eval_data["y"],
        eval_data["class_names"],
        eval_data["dataset_labels"],
    )


def preprocess_for_visualization(
    X_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full preprocessing pipeline for visualization.

    Steps: StandardScaler → VarianceThreshold → PSO → ACO

    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw expression matrix (n_samples, 10571).

    Returns
    -------
    X_processed : np.ndarray
        Processed feature matrix (n_samples, 80).
    X_scaled : np.ndarray
        After StandardScaler but before feature selection (n_samples, 10571).
        Useful for PCA on the full scaled feature space.
    """
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    X = X_raw.values.astype(np.float64)

    # 1) StandardScaler
    X_scaled = scaler.transform(X)

    # 2) Variance threshold
    X_var = selector.transform(X_scaled)

    # 3) PSO feature selection
    X_pso = X_var[:, pso_features]

    # 4) ACO feature selection
    X_final = X_pso[:, aco_features]  # (n, 80)

    return X_final, X_scaled


def load_or_compute_viz_data() -> dict[str, Any]:
    """Load cached visualization data or compute from scratch.

    Checks ``data/visualizations/`` for pre-computed CSVs.
    If absent, parses GSE files, preprocesses, and saves.

    Returns
    -------
    dict with keys:
        X_80 : np.ndarray — (n_samples, 80) post-pipeline features
        X_scaled : np.ndarray — (n_samples, 10571) scaled features
        y : np.ndarray — integer labels
        class_names : list[str]
        dataset_labels : list[str]
        feature_names_80 : list[str] — biomarker probe IDs
    """
    metadata_path = VIZ_DIR / "viz_metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        X_80 = np.load(str(VIZ_DIR / "X_80.npy"))
        X_scaled = np.load(str(VIZ_DIR / "X_scaled.npy"))
        y = np.load(str(VIZ_DIR / "y.npy"))
        return {
            "X_80": X_80,
            "X_scaled": X_scaled,
            "y": y,
            "class_names": meta["class_names"],
            "dataset_labels": meta["dataset_labels"],
            "feature_names_80": meta["feature_names_80"],
            "n_samples": meta["n_samples"],
        }

    # Compute from scratch
    X_raw, y, class_names, dataset_labels = load_raw_and_labels()
    X_80, X_scaled = preprocess_for_visualization(X_raw)
    feature_names_80 = get_selected_feature_names_clean()

    # Save for caching
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(VIZ_DIR / "X_80.npy"), X_80)
    np.save(str(VIZ_DIR / "X_scaled.npy"), X_scaled)
    np.save(str(VIZ_DIR / "y.npy"), y)

    metadata = {
        "class_names": class_names,
        "dataset_labels": dataset_labels,
        "feature_names_80": feature_names_80,
        "n_samples": int(len(y)),
    }
    with open(VIZ_DIR / "viz_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "X_80": X_80,
        "X_scaled": X_scaled,
        "y": y,
        "class_names": class_names,
        "dataset_labels": dataset_labels,
        "feature_names_80": feature_names_80,
        "n_samples": len(y),
    }


def get_short_class_names() -> list[str]:
    """Return short disease names for legend display."""
    return list(APP_CONFIG.class_short_names.values())
