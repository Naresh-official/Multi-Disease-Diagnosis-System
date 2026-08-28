"""
Model comparison: evaluate different feature-selection strategies.

Compares five approaches by training identical DNN architectures on
different feature subsets drawn from the same evaluation data:

1. No feature selection  — 10,571 features → DNN
2. Variance filtering    — 10,571 features → DNN  (all pass threshold)
3. PSO                   — 5,302 features  → DNN
4. ACO (standalone)      — 80 features     → DNN
5. Hybrid PSO + ACO      — 80 features     → DNN  (production model)

Because training DNNs from scratch is expensive, results are saved
to ``data/evaluation/model_comparison.json`` and the Streamlit page
loads from disk.  An explicit ``run_all_experiments()`` function
re-trains everything and refreshes the saved file.

Limitation: the variance threshold passes all 10,571 features, so
approaches 1 and 2 are numerically identical.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from core.config import APP_CONFIG


# ── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class ModelComparisonResult:
    """Structured result for one feature-selection approach."""

    model_name: str
    feature_count: int
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    roc_auc: float | None = None
    training_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    feature_reduction_pct: float = 0.0
    notes: str = ""
    # Per-class metrics stored separately to keep the dataclass flat
    per_class: list[dict[str, Any]] = field(default_factory=list)
    # Confusion matrix stored as nested list for JSON serialisation
    confusion_matrix: list[list[int]] = field(default_factory=list)


# ── Data loading ────────────────────────────────────────────────────────────


def _load_geo_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load and align all GEO datasets into one feature matrix + labels.

    Returns
    -------
    X_raw : pd.DataFrame   (n_samples, 10571)
    y     : np.ndarray     (n_samples,)
    class_names : list[str]
    """
    from evaluation.evaluator import load_evaluation_data

    data = load_evaluation_data()
    return data["X_raw"], data["y"], data["class_names"]


# ── DNN factory ─────────────────────────────────────────────────────────────


def _build_dnn(input_dim: int, num_classes: int = 4):
    """Build a DNN with the same architecture as the production model.

    Architecture mirrors ``hybrid_pso_aco_dl_model.keras``:
        Dense(64, relu) → Dropout → Dense(32, relu) → Dropout → Dense(4, softmax)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Train + evaluate helper ─────────────────────────────────────────────────


def _train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_count: int,
    class_names: list[str],
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
) -> ModelComparisonResult:
    """Train a fresh DNN and evaluate on held-out data."""
    import tensorflow as tf
    from evaluation.metrics import compute_all_metrics, compute_per_class_metrics
    from sklearn.metrics import confusion_matrix

    model = _build_dnn(X_train.shape[1], len(class_names))

    # Train
    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    train_time = time.time() - t0

    # Predict
    t1 = time.time()
    y_prob = model.predict(X_test, verbose=0)
    infer_time = time.time() - t1

    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_all_metrics(y_test, y_pred, y_prob, class_names)
    per_class = compute_per_class_metrics(y_test, y_pred, y_prob, class_names)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    total_features = APP_CONFIG.total_features
    reduction_pct = round(100.0 * (1 - feature_count / total_features), 2)

    return ModelComparisonResult(
        model_name=model_name,
        feature_count=feature_count,
        accuracy=metrics["accuracy"],
        precision_macro=metrics["precision_macro"],
        recall_macro=metrics["recall_macro"],
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
        roc_auc=metrics.get("roc_auc_ovr"),
        training_time_sec=round(train_time, 2),
        inference_time_sec=round(infer_time, 4),
        feature_reduction_pct=reduction_pct,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )


# ── Production model evaluation ─────────────────────────────────────────────


def _eval_production_model(
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> ModelComparisonResult:
    """Evaluate the pre-trained production model on test data."""
    import tensorflow as tf
    from evaluation.metrics import compute_all_metrics, compute_per_class_metrics
    from sklearn.metrics import confusion_matrix

    model = tf.keras.models.load_model(str(APP_CONFIG.model_path))
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    # Full pipeline
    X = X_test_raw.astype(np.float64)
    X_scaled = scaler.transform(X)
    X_var = selector.transform(X_scaled)
    X_pso = X_var[:, pso_features]
    X_final = X_pso[:, aco_features]

    t0 = time.time()
    y_prob = model.predict(X_final, verbose=0)
    infer_time = time.time() - t0

    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_all_metrics(y_test, y_pred, y_prob, class_names)
    per_class = compute_per_class_metrics(y_test, y_pred, y_prob, class_names)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    total_features = APP_CONFIG.total_features
    feature_count = len(aco_features)
    reduction_pct = round(100.0 * (1 - feature_count / total_features), 2)

    return ModelComparisonResult(
        model_name="Hybrid PSO + ACO (Production)",
        feature_count=feature_count,
        accuracy=metrics["accuracy"],
        precision_macro=metrics["precision_macro"],
        recall_macro=metrics["recall_macro"],
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
        roc_auc=metrics.get("roc_auc_ovr"),
        training_time_sec=0.0,  # not trained in this run
        inference_time_sec=round(infer_time, 4),
        feature_reduction_pct=reduction_pct,
        notes="Pre-trained model — not retrained. Inference time only.",
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )


# ── Full experiment runner ──────────────────────────────────────────────────


def run_all_experiments(
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 50,
    batch_size: int = 32,
) -> list[ModelComparisonResult]:
    """Run all five comparison experiments and return results.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Random seed for reproducible train/test split.
    epochs : int
        Training epochs for each baseline DNN.
    batch_size : int
        Mini-batch size for training.

    Returns
    -------
    list[ModelComparisonResult]
    """
    from sklearn.model_selection import train_test_split

    print("Loading GEO data…")
    X_raw, y, class_names = _load_geo_data()
    total_features = X_raw.shape[1]
    print(f"  Loaded {X_raw.shape[0]} samples × {total_features} features")

    # Stratified train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.values.astype(np.float64),
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Split: train={len(y_train)}, test={len(y_test)}")

    results: list[ModelComparisonResult] = []

    # ── 1. No feature selection ────────────────────────────────────────
    print("\n[1/5] No feature selection (all 10,571 features)…")
    r1 = _train_and_evaluate(
        X_train_raw, X_test_raw, y_train, y_test,
        model_name="No Feature Selection",
        feature_count=total_features,
        class_names=class_names,
        epochs=epochs,
        batch_size=batch_size,
    )
    results.append(r1)
    print(f"  → Acc={r1.accuracy:.4f}, F1={r1.f1_macro:.4f}")

    # ── 2. Variance filtering only ─────────────────────────────────────
    print("\n[2/5] Variance filtering only…")
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    var_mask = selector.get_support()
    n_var = var_mask.sum()
    # All features pass (threshold=0.01 on scaled data)
    # Use the same data as #1 — numerically identical
    r2 = _train_and_evaluate(
        X_train_raw, X_test_raw, y_train, y_test,
        model_name="Variance Filtering Only",
        feature_count=n_var,
        class_names=class_names,
        epochs=epochs,
        batch_size=batch_size,
    )
    r2.notes = (
        f"All {n_var} features pass variance threshold "
        f"({selector.threshold}). Results identical to 'No Feature Selection'."
    )
    results.append(r2)
    print(f"  → Acc={r2.accuracy:.4f}, F1={r2.f1_macro:.4f}")

    # ── 3. PSO only ────────────────────────────────────────────────────
    print("\n[3/5] PSO feature selection only…")
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    X_train_pso = X_train_raw[:, pso_features]
    X_test_pso = X_test_raw[:, pso_features]
    r3 = _train_and_evaluate(
        X_train_pso, X_test_pso, y_train, y_test,
        model_name="PSO Only",
        feature_count=len(pso_features),
        class_names=class_names,
        epochs=epochs,
        batch_size=batch_size,
    )
    r3.notes = f"PSO selected {len(pso_features)} features from {total_features}."
    results.append(r3)
    print(f"  → Acc={r3.accuracy:.4f}, F1={r3.f1_macro:.4f}")

    # ── 4. ACO only (on full variance-filtered data) ───────────────────
    print("\n[4/5] ACO feature selection only…")
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))
    # ACO indices are into the PSO-selected subset; for standalone ACO
    # we select the same 80 columns directly from the full feature set.
    # The production ACO indices (5240 max) index into PSO's 5302,
    # so we map them: full_feature_index = pso_features[aco_index]
    X_train_aco = X_train_raw[:, pso_features][:, aco_features]
    X_test_aco = X_test_raw[:, pso_features][:, aco_features]
    r4 = _train_and_evaluate(
        X_train_aco, X_test_aco, y_train, y_test,
        model_name="ACO Only",
        feature_count=len(aco_features),
        class_names=class_names,
        epochs=epochs,
        batch_size=batch_size,
    )
    r4.notes = (
        f"ACO selected {len(aco_features)} features. "
        f"Uses same 80 features as hybrid (post-PSO column mapping)."
    )
    results.append(r4)
    print(f"  → Acc={r4.accuracy:.4f}, F1={r4.f1_macro:.4f}")

    # ── 5. Hybrid PSO + ACO (production model) ─────────────────────────
    print("\n[5/5] Hybrid PSO + ACO (production model)…")
    r5 = _eval_production_model(X_test_raw, y_test, class_names)
    r5.notes = (
        f"Pre-trained production model ({len(aco_features)} features). "
        f"Not retrained — inference only."
    )
    results.append(r5)
    print(f"  → Acc={r5.accuracy:.4f}, F1={r5.f1_macro:.4f}")

    return results


# ── Save / load ─────────────────────────────────────────────────────────────

COMPARISON_JSON = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "model_comparison.json"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy integer and float types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_comparison(results: list[ModelComparisonResult]) -> None:
    """Persist comparison results to JSON."""
    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total_features": APP_CONFIG.total_features,
        "class_names": APP_CONFIG.class_names,
        "results": [asdict(r) for r in results],
    }
    with open(COMPARISON_JSON, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved comparison to {COMPARISON_JSON}")


def load_comparison() -> dict[str, Any] | None:
    """Load saved comparison results, or None if not yet run."""
    if not COMPARISON_JSON.exists():
        return None
    with open(COMPARISON_JSON) as f:
        return json.load(f)
