"""
Evaluation metrics for the multi-disease classifier.

Computes accuracy, precision, recall, F1, ROC-AUC, and per-class
breakdowns.  All metrics are computed using scikit-learn to ensure
correctness and reproducibility.

Statistical notes
-----------------
- **Macro F1**: Unweighted mean of per-class F1 scores.
  Treats all classes equally regardless of support.
- **Weighted F1**: Weighted mean of per-class F1 scores by support.
  Reflects overall performance accounting for class imbalance.
- **ROC-AUC (multiclass)**: Computed using the one-vs-rest (OvR)
  scheme.  Requires probability estimates for each class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute all aggregate metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n_samples, n_classes)``.
    class_names : list[str]
        Human-readable class names.

    Returns
    -------
    dict
        Keys: accuracy, precision_macro, recall_macro, f1_macro,
        f1_weighted, roc_auc_ovr, roc_auc_macro.
    """
    n_classes = len(class_names)

    accuracy = float(accuracy_score(y_true, y_pred))

    # Precision / Recall / F1 — macro (unweighted) and weighted
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ROC-AUC (One-vs-Rest)
    # All classes must be present in y_true for multiclass AUC.
    present_classes = sorted(set(y_true))
    roc_auc_ovr: float | None = None
    roc_auc_macro: float | None = None

    if len(present_classes) >= 2 and y_prob.shape[1] >= 2:
        try:
            roc_auc_ovr = float(roc_auc_score(
                y_true, y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            ))
            roc_auc_macro = roc_auc_ovr  # same as macro OvR
        except ValueError:
            # Can happen if a class has no samples
            roc_auc_ovr = None
            roc_auc_macro = None

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr": roc_auc_ovr,
        "roc_auc_macro": roc_auc_macro,
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Compute per-class precision, recall, F1, and support.

    Returns
    -------
    list of dict
        Each dict has keys: class_name, class_id, precision, recall,
        f1, support, auc (if computable).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    n_classes = len(class_names)
    present = sorted(set(y_true))

    results = []
    for cls_id in range(n_classes):
        name = class_names[cls_id]

        # Binary one-vs-rest for this class
        y_true_bin = (y_true == cls_id).astype(int)
        y_pred_bin = (y_pred == cls_id).astype(int)
        support = int(y_true_bin.sum())

        if support == 0:
            results.append({
                "class_name": name,
                "class_id": cls_id,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "auc": None,
            })
            continue

        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        # AUC for this class
        auc_val: float | None = None
        if cls_id in present and y_prob.shape[1] > cls_id:
            try:
                auc_val = float(roc_auc_score(y_true_bin, y_prob[:, cls_id]))
            except ValueError:
                auc_val = None

        results.append({
            "class_name": name,
            "class_id": cls_id,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "auc": auc_val,
        })

    return results
