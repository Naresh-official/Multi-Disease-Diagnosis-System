"""
Batch prediction for multiple gene-expression samples.

Processes a collection of uploaded CSV files through the **same**
preprocessing and prediction pipeline used for single-sample
diagnosis.  Invalid files are rejected individually without
crashing the entire batch.

Architecture
------------
- :func:`process_batch` is the main entry point.  It accepts a
  dict mapping filename → raw bytes and returns a
  :class:`BatchResult`.
- Each file is validated, preprocessed, and predicted using the
  existing ``core/pipeline.py`` and ``core/predictor.py`` modules.
- Model artefacts are loaded once via ``load_artifacts()`` (which
  uses ``@st.cache_resource``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.config import APP_CONFIG
from core.pipeline import preprocess
from core.predictor import load_artifacts, predict
from core.validation import (
    validate_dataframe,
    validate_file_content,
)
from utils.uncertainty import build_uncertainty_result, UncertaintyResult


# ── Structured results ──────────────────────────────────────────────────────


@dataclass
class SampleResult:
    """Prediction result for a single sample within a batch.

    Attributes
    ----------
    filename : str
        Original uploaded filename (without path).
    valid : bool
        ``True`` if the file passed all validation checks.
    error_message : str | None
        First validation error, or ``None`` if valid.
    prediction : str | None
        Human-readable predicted disease name, or ``None``.
    predicted_class : int | None
        Numeric class index (0–3), or ``None``.
    confidence : float | None
        Maximum softmax probability, or ``None``.
    probabilities : np.ndarray | None
        Full 4-class probability vector, or ``None``.
    uncertainty : UncertaintyResult | None
        Full uncertainty analysis, or ``None``.
    """

    filename: str
    valid: bool
    error_message: str | None = None
    prediction: str | None = None
    predicted_class: int | None = None
    confidence: float | None = None
    probabilities: np.ndarray | None = None
    uncertainty: UncertaintyResult | None = None


@dataclass
class BatchResult:
    """Aggregated result of processing an entire batch.

    Attributes
    ----------
    total_files : int
        Number of files submitted.
    valid_files : int
        Number that passed validation.
    invalid_files : int
        Number that failed validation.
    successfully_analyzed : int
        Number that were successfully predicted.
    low_confidence_count : int
        Number flagged as low-confidence.
    samples : list[SampleResult]
        Per-sample results in submission order.
    """

    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    successfully_analyzed: int = 0
    low_confidence_count: int = 0
    samples: list[SampleResult] = field(default_factory=list)


# ── Core batch processor ────────────────────────────────────────────────────


def process_batch(
    files: dict[str, bytes],
    *,
    progress_callback: Any = None,
) -> BatchResult:
    """Validate and predict on every file in the batch.

    Parameters
    ----------
    files : dict[str, bytes]
        Mapping of ``filename → raw file bytes`` for each uploaded CSV.
    progress_callback : callable, optional
        Called as ``progress_callback(processed, total)`` after each
        file is processed.  Used by Streamlit to update a progress bar.

    Returns
    -------
    BatchResult
        Contains per-sample results and aggregate statistics.
    """
    artifacts = load_artifacts()
    result = BatchResult(total_files=len(files))

    for idx, (filename, raw_bytes) in enumerate(files.items(), 1):
        sample = _process_single_file(filename, raw_bytes, artifacts)
        result.samples.append(sample)

        if sample.valid:
            result.valid_files += 1
            if sample.prediction is not None:
                result.successfully_analyzed += 1
            if (
                sample.uncertainty is not None
                and sample.uncertainty.is_low_confidence
            ):
                result.low_confidence_count += 1
        else:
            result.invalid_files += 1

        if progress_callback is not None:
            progress_callback(idx, result.total_files)

    return result


def _process_single_file(
    filename: str,
    raw_bytes: bytes,
    artifacts: dict[str, Any],
) -> SampleResult:
    """Validate, preprocess, and predict for one file.

    Errors are caught and returned as part of the
    :class:`SampleResult` — they never propagate.
    """
    # ── File-level readability ────────────────────────────────────────
    file_ok, file_msg = validate_file_content(raw_bytes)
    if not file_ok:
        return SampleResult(filename=filename, valid=False, error_message=file_msg)

    # ── Parse CSV ─────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        return SampleResult(
            filename=filename,
            valid=False,
            error_message=f"Failed to parse CSV: {exc}",
        )

    # ── Full validation ───────────────────────────────────────────────
    vr = validate_dataframe(df, check_schema=True)
    if not vr.valid:
        # Return the first error as the summary message
        error_msg = vr.errors[0] if vr.errors else "Unknown validation error."
        return SampleResult(filename=filename, valid=False, error_message=error_msg)

    # ── Preprocessing ─────────────────────────────────────────────────
    try:
        x_final = preprocess(df)
    except Exception as exc:
        return SampleResult(
            filename=filename,
            valid=False,
            error_message=f"Preprocessing failed: {exc}",
        )

    # ── Model prediction ──────────────────────────────────────────────
    try:
        probs = predict(artifacts, x_final)
    except Exception as exc:
        return SampleResult(
            filename=filename,
            valid=False,
            error_message=f"Prediction failed: {exc}",
        )

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    disease_name = APP_CONFIG.disease_map[pred_class]

    # ── Uncertainty analysis ──────────────────────────────────────────
    uncertainty = build_uncertainty_result(probs)

    return SampleResult(
        filename=filename,
        valid=True,
        prediction=disease_name,
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=probs,
        uncertainty=uncertainty,
    )


# ── CSV export helper ───────────────────────────────────────────────────────


def results_to_dataframe(result: BatchResult) -> pd.DataFrame:
    """Convert a :class:`BatchResult` into a tidy DataFrame for export.

    Columns match the specification::

        sample, prediction, confidence, uncertainty,
        diabetes_probability, cad_probability,
        autoimmune_probability, alzheimers_probability
    """
    rows = []
    for s in result.samples:
        if not s.valid or s.probabilities is None:
            rows.append({
                "sample": s.filename,
                "prediction": "INVALID",
                "confidence": "",
                "uncertainty": s.error_message or "",
                "diabetes_probability": "",
                "cad_probability": "",
                "autoimmune_probability": "",
                "alzheimers_probability": "",
            })
            continue

        probs = s.probabilities
        uncertainty_label = (
            s.uncertainty.confidence_level.value if s.uncertainty else ""
        )
        rows.append({
            "sample": s.filename,
            "prediction": s.prediction or "",
            "confidence": round(s.confidence, 6) if s.confidence is not None else "",
            "uncertainty": uncertainty_label,
            "diabetes_probability": round(float(probs[0]), 6),
            "cad_probability": round(float(probs[1]), 6),
            "autoimmune_probability": round(float(probs[2]), 6),
            "alzheimers_probability": round(float(probs[3]), 6),
        })

    return pd.DataFrame(rows)
