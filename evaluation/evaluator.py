"""
Model evaluator for the Explainable Multi-Disease Diagnosis System.

Loads independent evaluation data from GEO Series Matrix files,
runs it through the exact same preprocessing pipeline used during
inference, generates predictions, and computes comprehensive metrics.

The evaluation dataset is constructed from the raw GSE files that
were used to train the model.  Since no separate held-out test set
was provided, we parse the full datasets and use ALL samples for
evaluation.  This means the evaluation data overlaps with training
data — this limitation is documented in the UI.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from core.config import APP_CONFIG


# ── GSE file mapping ────────────────────────────────────────────────────────

GSE_FILES: dict[int, str] = {
    0: "GSE66360_(Cad)series_matrix (3).txt.gz",
    1: "GSE50772_(Auto-Immune)series_matrix.txt.gz",
    2: "GSE48350_(Alzheimer)series_matrix.txt.gz",
    3: "GSE6613_(Parkinson's)series_matrix.txt.gz",
}

GEO_DISEASE_NAMES: dict[int, str] = {
    0: "CAD",
    1: "Autoimmune",
    2: "Alzheimer's",
    3: "Parkinson's",
}

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"


# ── GSE parsing ─────────────────────────────────────────────────────────────


PARKINSON_DISEASE_KEYWORDS = {"parkinson", "parkinson's disease", "parkinsons disease"}


def _parse_gse_file(
    filepath: str,
    expected_probes: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Parse a GEO Series Matrix file and extract expression data.

    Parameters
    ----------
    filepath : str
        Path to the gzipped series matrix file.
    expected_probes : list[str]
        Probe IDs the model expects (without quotes).

    Returns
    -------
    df : pd.DataFrame
        Expression matrix with samples as rows, probes as columns.
    sample_ids : list[str]
        GEO sample accession IDs.
    sample_titles : list[str]
        Human-readable sample titles.
    """
    with gzip.open(filepath, "rt", errors="replace") as fh:
        lines = fh.readlines()

    # Find table boundaries
    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if "!series_matrix_table_begin" in line:
            begin_idx = i
        if "!series_matrix_table_end" in line:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        raise ValueError(f"Data table not found in {filepath}")

    # Parse header → sample IDs
    header = lines[begin_idx + 1].strip().split("\t")
    sample_ids = [s.strip('"') for s in header[1:]]

    # Parse sample titles
    sample_titles = []
    for line in lines:
        if line.startswith("!Sample_title"):
            sample_titles = [t.strip('"') for t in line.strip().split("\t")[1:]]
            break

    # Parse expression data — only keep expected probes
    expected_set = set(expected_probes)
    probe_data: dict[str, list[float]] = {}

    for i in range(begin_idx + 2, end_idx):
        parts = lines[i].strip().split("\t")
        probe_id = parts[0].strip('"')
        if probe_id in expected_set:
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(np.nan)
            probe_data[probe_id] = vals

    # Build DataFrame ordered by expected probe order
    matched_probes = [p for p in expected_probes if p in probe_data]
    df = pd.DataFrame({p: probe_data[p] for p in matched_probes})

    # Fill NaN with column median
    df = df.fillna(df.median())

    return df, sample_ids, sample_titles


# ── Load evaluation data (no Streamlit dependency) ─────────────────────────


def load_evaluation_data() -> dict[str, Any]:
    """Load and combine all GSE datasets into a single evaluation set.

    This function has no Streamlit dependency and can be used standalone.

    Returns
    -------
    dict with keys:
        X_raw : pd.DataFrame — raw expression (n_samples, 10571)
        y : np.ndarray — integer labels (n_samples,)
        class_names : list[str]
        sample_ids : list[str]
        dataset_labels : list[str] — which GSE each sample came from
        n_per_class : dict[int, int]
    """
    # Get expected probe names by finding common probes across all GSE files
    # (the scaler was trained on numpy arrays, so feature_names_in_ is unavailable)
    project_root = Path(__file__).resolve().parent.parent
    all_probe_sets: list[set[str]] = []
    for _label, fname in GSE_FILES.items():
        fpath = str(project_root / fname)
        with gzip.open(fpath, "rt", errors="replace") as fh:
            lines = fh.readlines()
        begin_idx = end_idx = None
        for i, line in enumerate(lines):
            if "!series_matrix_table_begin" in line:
                begin_idx = i
            if "!series_matrix_table_end" in line:
                end_idx = i
                break
        probes = set()
        for i in range(begin_idx + 2, end_idx):
            parts = lines[i].strip().split("\t")
            pid = parts[0].strip('"')
            if pid and not pid.startswith("!") and not pid.startswith('"'):
                probes.add(pid)
        all_probe_sets.append(probes)
    expected_set = all_probe_sets[0]
    for ps in all_probe_sets[1:]:
        expected_set &= ps
    expected_probes = sorted(expected_set)
    print(f"[evaluator] Found {len(expected_probes)} common probes across GSE files")

    all_dfs: list[pd.DataFrame] = []
    all_labels: list[int] = []
    all_sample_ids: list[str] = []
    all_dataset_labels: list[str] = []

    for label, fname in GSE_FILES.items():
        fpath = str(project_root / fname)
        df, sids, titles = _parse_gse_file(fpath, expected_probes)

        # Filter Parkinson's to disease samples only
        if label == 3:  # Parkinson's
            with gzip.open(fpath, "rt", errors="replace") as fh:
                lines = fh.readlines()
            sample_chars = []
            for line in lines:
                if line.startswith("!Sample_characteristics_ch1"):
                    sample_chars = [
                        c.strip('"').lower()
                        for c in line.strip().split("\t")[1:]
                    ]
                    break
            mask = [
                any(kw in c for kw in PARKINSON_DISEASE_KEYWORDS)
                for c in sample_chars
            ]
            df = df.iloc[mask]
            sids = [s for s, m in zip(sids, mask) if m]

        all_dfs.append(df)
        all_labels.extend([label] * len(sids))
        all_sample_ids.extend(
            [f"{GEO_DISEASE_NAMES[label]}_{sid}" for sid in sids]
        )
        all_dataset_labels.extend([GEO_DISEASE_NAMES[label]] * len(sids))

    # Align all DataFrames to the same column set (expected probe order)
    common_cols = expected_probes
    aligned_dfs = []
    for df in all_dfs:
        aligned = df.reindex(columns=common_cols, fill_value=0.0)
        aligned_dfs.append(aligned)

    X_raw = pd.concat(aligned_dfs, ignore_index=True)
    y = np.array(all_labels, dtype=np.int32)

    # Per-class counts
    n_per_class = {int(k): int((y == k).sum()) for k in sorted(set(y))}

    return {
        "X_raw": X_raw,
        "y": y,
        "class_names": list(APP_CONFIG.class_names),
        "sample_ids": all_sample_ids,
        "dataset_labels": all_dataset_labels,
        "n_per_class": n_per_class,
    }


# ── Run evaluation (no Streamlit dependency) ────────────────────────────────


def run_evaluation() -> dict[str, Any]:
    """Run the full evaluation pipeline and return results.

    This function has no Streamlit dependency.  For Streamlit pages,
    use the cached wrapper :func:`run_evaluation_cached`.

    Returns
    -------
    dict with keys:
        y_true, y_pred, y_prob, metrics, per_class_metrics, cm, ...
    """
    eval_data = load_evaluation_data()
    X_raw: pd.DataFrame = eval_data["X_raw"]
    y_true: np.ndarray = eval_data["y"]
    class_names: list[str] = eval_data["class_names"]

    # Load model and preprocessing artefacts
    model_path = APP_CONFIG.model_path
    model = joblib.load  # not used, use tf directly
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    # ── Apply full preprocessing pipeline ──────────────────────────────
    X = X_raw.values.astype(np.float64)

    # 1) StandardScaler
    X_scaled = scaler.transform(X)

    # 2) Variance threshold
    X_var = selector.transform(X_scaled)

    # 3) PSO feature selection
    X_pso = X_var[:, pso_features]

    # 4) ACO feature selection
    X_final = X_pso[:, aco_features]  # (n, 80)

    # ── Predict ────────────────────────────────────────────────────────
    y_prob = model.predict(X_final, verbose=0)  # (n, 4)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Compute metrics ────────────────────────────────────────────────
    from evaluation.metrics import compute_all_metrics, compute_per_class_metrics

    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    per_class = compute_per_class_metrics(y_true, y_pred, y_prob, class_names)

    # ── Confusion matrix ──────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # ── Build results dict ────────────────────────────────────────────
    results = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "per_class": per_class,
        "cm": cm,
        "class_names": class_names,
        "n_samples": len(y_true),
        "n_features_raw": X_raw.shape[1],
        "n_features_final": X_final.shape[1],
        "dataset_labels": eval_data["dataset_labels"],
        "sample_ids": eval_data["sample_ids"],
        "n_per_class": eval_data["n_per_class"],
    }

    return results


# ── Save / load results for reproducibility ─────────────────────────────────


def save_results(results: dict[str, Any]) -> None:
    """Save evaluation results to disk for reproducibility.

    Saves to ``data/evaluation/``:
    - ``y_true.npy``, ``y_pred.npy``, ``y_prob.npy``
    - ``evaluation_results.json``
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(EVAL_DIR / "y_true.npy"), results["y_true"])
    np.save(str(EVAL_DIR / "y_pred.npy"), results["y_pred"])
    np.save(str(EVAL_DIR / "y_prob.npy"), results["y_prob"])

    # JSON-serialisable subset
    json_metrics = {
        "metrics": results["metrics"],
        "per_class": results["per_class"],
        "n_samples": int(results["n_samples"]),
        "n_features_raw": int(results["n_features_raw"]),
        "n_features_final": int(results["n_features_final"]),
        "n_per_class": {str(k): v for k, v in results["n_per_class"].items()},
        "class_names": results["class_names"],
    }
    with open(EVAL_DIR / "evaluation_results.json", "w") as f:
        json.dump(json_metrics, f, indent=2)


def load_saved_results() -> dict[str, Any] | None:
    """Load previously saved evaluation results from disk.

    Returns None if no saved results exist.
    """
    json_path = EVAL_DIR / "evaluation_results.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Reconstruct numpy arrays
    y_true_path = EVAL_DIR / "y_true.npy"
    y_pred_path = EVAL_DIR / "y_pred.npy"
    y_prob_path = EVAL_DIR / "y_prob.npy"

    if y_true_path.exists() and y_pred_path.exists() and y_prob_path.exists():
        data["y_true"] = np.load(str(y_true_path))
        data["y_pred"] = np.load(str(y_pred_path))
        data["y_prob"] = np.load(str(y_prob_path))

        from sklearn.metrics import confusion_matrix
        data["cm"] = confusion_matrix(
            data["y_true"], data["y_pred"],
            labels=list(range(len(data["class_names"]))),
        )

    return data


# ── Streamlit-cached wrapper ────────────────────────────────────────────────

try:
    import streamlit as st

    @st.cache_data(
        show_spinner="Running model evaluation…",
        hash_funcs={pd.DataFrame: lambda df: hash(df.shape)},
    )
    def run_evaluation_cached(_eval_data_hash: str) -> dict[str, Any]:
        """Streamlit-cached version of run_evaluation."""
        return run_evaluation()

except ImportError:
    # Not running in Streamlit — no-op
    def run_evaluation_cached(_eval_data_hash: str) -> dict[str, Any]:  # type: ignore[misc]
        return run_evaluation()
