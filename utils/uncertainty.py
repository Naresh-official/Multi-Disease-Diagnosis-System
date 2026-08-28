"""
Uncertainty analysis for model predictions.

Provides functions to assess prediction confidence, compute entropy,
classify uncertainty levels, and (optionally) evaluate model
calibration against held-out evaluation data.

All thresholds are documented as configurable engineering values,
not clinically validated cutoffs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────────

# These are engineering thresholds, NOT clinical decision boundaries.
# They should be adjusted based on empirical evaluation or domain input.
LOW_CONFIDENCE_THRESHOLD: float = 0.50
HIGH_CONFIDENCE_THRESHOLD: float = 0.90
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.70


class ConfidenceLevel(str, Enum):
    """Human-readable confidence categories."""

    HIGH = "High"
    MEDIUM = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class UncertaintyResult:
    """Structured result of uncertainty analysis for a single prediction.

    Attributes
    ----------
    max_probability : float
        The highest softmax probability among all classes.
    predicted_class : int
        Index of the class with the highest probability.
    entropy : float
        Shannon entropy of the probability distribution (bits).
    normalised_entropy : float
        Entropy normalised to [0, 1] where 0 = certain, 1 = uniform.
    confidence_level : ConfidenceLevel
        Categorical confidence assignment.
    is_low_confidence : bool
        True if max_probability < LOW_CONFIDENCE_THRESHOLD.
    probability_sum : float
        Sum of all class probabilities (should be ≈ 1.0).
    probability_valid : bool
        True if sum is within tolerance and no NaN/inf values.
    warning_message : str | None
        User-facing warning text, or None if confidence is adequate.
    """

    max_probability: float
    predicted_class: int
    entropy: float
    normalised_entropy: float
    confidence_level: ConfidenceLevel
    is_low_confidence: bool
    probability_sum: float
    probability_valid: bool
    warning_message: str | None = None


# ── Core functions ──────────────────────────────────────────────────────────


def get_max_probability(probabilities: np.ndarray) -> tuple[float, int]:
    """Return the maximum softmax probability and its class index.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_classes,)
        Softmax output from the model.

    Returns
    -------
    (max_prob, class_index) : tuple[float, int]

    Raises
    ------
    ValueError
        If the array is empty or contains non-finite values.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError("Probability array is empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probability array contains NaN or infinite values."
        )

    class_index = int(np.argmax(probabilities))
    max_prob = float(probabilities[class_index])
    return max_prob, class_index


def calculate_entropy(probabilities: np.ndarray) -> tuple[float, float]:
    """Compute Shannon entropy and its normalised value.

    Uses log base 2 so the result is in **bits**.  Normalised
    entropy is in [0, 1] where:
    - 0  → completely certain (one class has probability 1)
    - 1  → completely uncertain (uniform distribution)

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_classes,)
        Softmax probabilities.  Need not sum to 1 (will be
        normalised internally if necessary).

    Returns
    -------
    (entropy, normalised_entropy) : tuple[float, float]
        Entropy in bits and normalised entropy in [0, 1].

    Raises
    ------
    ValueError
        If the array is empty, contains negative values, or
        contains non-finite values.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError("Probability array is empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probability array contains NaN or infinite values."
        )

    if np.any(probabilities < 0):
        raise ValueError("Probability array contains negative values.")

    # Normalise to ensure sum = 1 (guards against floating-point drift)
    total = probabilities.sum()
    if total > 0:
        p = probabilities / total
    else:
        # Degenerate case: treat as uniform
        p = np.ones_like(probabilities) / probabilities.size

    n_classes = len(p)

    # Shannon entropy with safe log: replace log(0) → 0
    log_p = np.where(p > 0, np.log2(p), 0.0)
    entropy = -float(np.sum(p * log_p))

    # Normalise: max entropy for n_classes is log2(n_classes)
    max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0

    return entropy, float(normalised)


def classify_confidence(
    max_prob: float,
    low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    medium_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> ConfidenceLevel:
    """Assign a confidence category based on the maximum probability.

    Thresholds are configurable engineering values, not clinical
    decision boundaries.

    Parameters
    ----------
    max_prob : float
        Maximum softmax probability (0–1).
    low_threshold : float
        Below this → Very Low.
    medium_threshold : float
        Below this → Low, at or above → Moderate.
    high_threshold : float
        At or above → High.

    Returns
    -------
    ConfidenceLevel
    """
    if max_prob >= high_threshold:
        return ConfidenceLevel.HIGH
    elif max_prob >= medium_threshold:
        return ConfidenceLevel.MEDIUM
    elif max_prob >= low_threshold:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def validate_probabilities(probabilities: np.ndarray) -> tuple[bool, float]:
    """Check that a probability vector is well-formed.

    Validates:
    - No NaN or infinite values.
    - All values in [0, 1].
    - Sum is approximately 1.0 (tolerance ±0.05 to allow rounding).

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_classes,)

    Returns
    -------
    (is_valid, prob_sum) : tuple[bool, float]
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        return False, 0.0

    if not np.all(np.isfinite(probabilities)):
        return False, float("nan")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        return False, float(probabilities.sum())

    prob_sum = float(probabilities.sum())
    is_valid = abs(prob_sum - 1.0) <= 0.05
    return is_valid, prob_sum


def build_uncertainty_result(probabilities: np.ndarray) -> UncertaintyResult:
    """Compute a full UncertaintyResult from a softmax probability vector.

    This is the main entry point that combines all uncertainty
    analyses into a single structured result.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_classes,)
        Softmax output from the DNN.

    Returns
    -------
    UncertaintyResult
    """
    prob_valid, prob_sum = validate_probabilities(probabilities)
    max_prob, pred_class = get_max_probability(probabilities)
    entropy, norm_entropy = calculate_entropy(probabilities)
    conf_level = classify_confidence(max_prob)
    is_low = max_prob < LOW_CONFIDENCE_THRESHOLD

    warning = None
    if is_low:
        warning = (
            "⚠️ **Low-confidence prediction.** "
            "The model does not strongly distinguish between the "
            "available disease classes for this sample.  Results "
            "should be interpreted with caution and are not suitable "
            "for clinical decision-making without further validation."
        )
    elif not prob_valid:
        warning = (
            "⚠️ **Probability validation warning.** "
            "The model output probabilities do not sum to approximately "
            "1.0, which may indicate a numerical issue."
        )

    return UncertaintyResult(
        max_probability=max_prob,
        predicted_class=pred_class,
        entropy=entropy,
        normalised_entropy=norm_entropy,
        confidence_level=conf_level,
        is_low_confidence=is_low,
        probability_sum=prob_sum,
        probability_valid=prob_valid,
        warning_message=warning,
    )


# ── Calibration analysis (optional, requires evaluation data) ───────────────


def load_evaluation_probabilities() -> tuple[np.ndarray, np.ndarray] | None:
    """Load true labels and predicted probabilities from the evaluation set.

    Returns None if the evaluation files do not exist.

    Returns
    -------
    (y_true, y_prob) : tuple[np.ndarray, np.ndarray] or None
        y_true shape (n_samples,), y_prob shape (n_samples, n_classes).
    """
    eval_dir = Path(__file__).resolve().parent.parent / "data" / "evaluation"
    y_true_path = eval_dir / "y_true.npy"
    y_prob_path = eval_dir / "y_prob.npy"

    if not y_true_path.exists() or not y_prob_path.exists():
        return None

    y_true = np.load(str(y_true_path))
    y_prob = np.load(str(y_prob_path))
    return y_true, y_prob


@dataclass
class CalibrationMetrics:
    """Results of model calibration analysis.

    Attributes
    ----------
    brier_score : float
        Mean squared difference between predicted probabilities
        and actual outcomes.  Lower is better (0 = perfect).
    ece : float
        Expected Calibration Error — weighted average of
        per-bin |accuracy − confidence|.  Lower is better.
    n_bins : int
        Number of probability bins used.
    bin_data : list of dict
        Per-bin statistics: bin_lower, bin_upper, mean_confidence,
        mean_accuracy, count.
    """

    brier_score: float
    ece: float
    n_bins: int
    bin_data: list[dict] = field(default_factory=list)


def calculate_calibration_metrics(
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationMetrics | None:
    """Compute calibration metrics from evaluation data.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,), optional
        True class labels.  If None, loaded from evaluation files.
    y_prob : np.ndarray of shape (n_samples, n_classes), optional
        Predicted probabilities.  If None, loaded from evaluation files.
    n_bins : int
        Number of bins for the reliability diagram.

    Returns
    -------
    CalibrationMetrics or None
        None if evaluation data is not available.
    """
    if y_true is None or y_prob is None:
        data = load_evaluation_probabilities()
        if data is None:
            return None
        y_true, y_prob = data

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_samples = len(y_true)
    n_classes = y_prob.shape[1]

    # ── Brier score (multiclass) ──────────────────────────────────────
    # One-hot encode true labels
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, label in enumerate(y_true):
        y_onehot[i, int(label)] = 1.0

    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    # ── Expected Calibration Error ────────────────────────────────────
    # For each sample, take max predicted prob as "confidence"
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        # Include right edge in the last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(mask.sum())
        if count == 0:
            bin_data.append({
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_confidence": 0.0,
                "mean_accuracy": 0.0,
                "count": 0,
            })
            continue

        mean_conf = float(confidences[mask].mean())
        mean_acc = float(accuracies[mask].mean())

        bin_data.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "count": count,
        })

        ece += (count / n_samples) * abs(mean_acc - mean_conf)

    return CalibrationMetrics(
        brier_score=brier,
        ece=float(ece),
        n_bins=n_bins,
        bin_data=bin_data,
    )
