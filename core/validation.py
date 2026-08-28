"""
Input validation for uploaded patient gene-expression data.

This module provides a **reusable**, Streamlit-independent validation
layer that checks an uploaded CSV before it enters the prediction
pipeline.  Every check is additive — the validator runs *all* checks
and returns a structured :class:`ValidationResult` so the caller
(or UI) can display every issue at once.

Checks performed
----------------
1. File readability (handled at caller level — CSV must already parse)
2. Dataset shape — row count, column count
3. Duplicate column names
4. Empty rows / empty columns
5. Feature schema — column-name match against the scaler's training schema
6. Numeric values — all gene-expression values must be numeric
7. Missing values — NaN / null detection with location
8. Infinite values — ±inf detection
9. Single-sample compatibility — exactly one row

The expected feature count is read from ``APP_CONFIG.total_features``
so that it is never hardcoded in multiple locations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.config import APP_CONFIG


# ── Structured result ───────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Structured outcome of all validation checks.

    Attributes
    ----------
    valid : bool
        ``True`` only if *no* errors were recorded.
    errors : list[str]
        Human-readable fatal error messages.
    warnings : list[str]
        Human-readable non-fatal warning messages.
    feature_count : int | None
        Number of feature columns detected in the input.
    sample_count : int | None
        Number of patient rows detected in the input.
    missing_value_count : int
        Total number of NaN/null cells.
    invalid_value_count : int
        Total number of non-finite (inf) or non-numeric cells.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feature_count: Optional[int] = None
    sample_count: Optional[int] = None
    missing_value_count: int = 0
    invalid_value_count: int = 0

    def add_error(self, msg: str) -> None:
        """Append an error and mark the result as invalid."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Append a non-fatal warning."""
        self.warnings.append(msg)


# ── Public API ──────────────────────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    *,
    check_schema: bool = True,
) -> ValidationResult:
    """Run the full validation suite on an already-parsed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The parsed CSV content (caller is responsible for the initial
        ``pd.read_csv`` and for catching parse errors).
    check_schema : bool
        If ``True``, attempt to validate column names against the
        training scaler's expected feature schema.

    Returns
    -------
    ValidationResult
        Contains all errors, warnings, and summary statistics.
    """
    result = ValidationResult()

    if df.empty:
        result.add_error("The uploaded file is empty (no rows or columns).")
        return result

    result.sample_count = df.shape[0]
    result.feature_count = df.shape[1]

    _check_shape(df, result)
    _check_empty_rows_columns(df, result)
    _check_duplicate_columns(df, result)

    if check_schema:
        _check_feature_schema(df, result)

    _check_numeric_values(df, result)
    _check_missing_values(df, result)
    _check_infinite_values(df, result)
    _check_single_sample(df, result)

    return result


# ── Individual checks ───────────────────────────────────────────────────────


def _check_shape(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate the number of feature columns matches the expected count."""
    expected = APP_CONFIG.total_features
    actual = df.shape[1]

    if actual != expected:
        result.add_error(
            f"Expected {expected:,} gene features, but the uploaded file "
            f"has {actual:,} columns."
        )


def _check_empty_rows_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Detect obviously empty rows or columns."""
    # Completely empty columns (all NaN)
    # Use .iloc-based iteration to avoid issues with duplicate column names,
    # where df[col] would return a DataFrame instead of a Series.
    empty_cols: list[str] = []
    for i in range(df.shape[1]):
        if df.iloc[:, i].isna().all():
            empty_cols.append(str(df.columns[i]))
    if empty_cols:
        names = (
            ", ".join(empty_cols[:5])
            + (" ..." if len(empty_cols) > 5 else "")
        )
        result.add_error(
            f"Found {len(empty_cols)} completely empty column(s): {names}"
        )

    # Completely empty rows (all NaN)
    empty_rows = df.index[df.isna().all(axis=1)].tolist()
    if empty_rows:
        display_idx = [str(i) for i in empty_rows[:5]]
        result.add_error(
            f"Found {len(empty_rows)} completely empty row(s) at "
            f"index: {', '.join(display_idx)}"
            + (" ..." if len(empty_rows) > 5 else "")
        )


def _check_duplicate_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Detect duplicate feature names."""
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        names = (
            ", ".join(str(d) for d in dupes[:5])
            + (" ..." if len(dupes) > 5 else "")
        )
        result.add_error(
            f"Found {len(dupes)} duplicate column name(s): {names}. "
            "Please remove duplicate features."
        )


def _check_feature_schema(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate that column names match the training scaler's expected schema.

    The StandardScaler artefact stores ``feature_names_in_`` when it was
    fitted on a DataFrame.  If that attribute is available, we compare
    it against the uploaded columns.  If the attribute is missing (e.g.
    the scaler was fitted on a numpy array), we emit a warning instead
    of a hard error.

    Limitation
    ----------
    If feature names differ only in quoting style (e.g. ``gene_1`` vs
    ``"gene_1"``) or ordering, we report it but do **not** silently
    reorder.
    """
    try:
        from core.predictor import load_artifacts

        artifacts = load_artifacts()
        scaler = artifacts["scaler"]

        from utils.helpers import get_expected_feature_names
        expected_names = set(get_expected_feature_names())
        actual_names = set(df.columns)

        if actual_names == expected_names:
            return  # perfect match — no issues

        missing = expected_names - actual_names
        extra = actual_names - expected_names

        if missing and extra:
            result.add_error(
                f"Feature schema mismatch: {len(missing)} expected feature(s) "
                f"are missing and {len(extra)} unexpected feature(s) were "
                f"found. The uploaded CSV does not match the training schema."
            )
        elif missing:
            result.add_error(
                f"Missing {len(missing)} expected feature(s) from the "
                f"training schema (e.g. {', '.join(sorted(missing)[:3])})."
            )
        elif extra:
            result.add_warning(
                f"Found {len(extra)} extra feature(s) not in the training "
                f"schema (e.g. {', '.join(sorted(extra)[:3])}). "
                "These will be present but the pipeline may not use them."
            )

    except Exception:
        # If artefacts can't be loaded (e.g. during unit tests), skip schema check
        result.add_warning(
            "Could not load training artefacts for schema validation. "
            "Feature name validation was skipped."
        )


def _check_numeric_values(df: pd.DataFrame, result: ValidationResult) -> None:
    """Ensure all gene-expression values are numeric.

    We attempt to convert every column to ``float64``.  Columns that
    contain strings or other non-numeric data are flagged.
    """
    # Fast path: if all columns are already numeric dtypes, skip the slow
    # per-column pd.to_numeric check.
    non_numeric_cols_mask = ~df.columns.to_series().apply(
        lambda col: pd.api.types.is_numeric_dtype(df[col])
    )
    if not non_numeric_cols_mask.any():
        return  # all columns are numeric — nothing to check

    # Slow path: some columns are object/string — identify which ones fail
    non_numeric_cols: list[str] = []
    for col in df.columns[non_numeric_cols_mask]:
        try:
            pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            non_numeric_cols.append(str(col))

    if non_numeric_cols:
        names = (
            ", ".join(non_numeric_cols[:5])
            + (" ..." if len(non_numeric_cols) > 5 else "")
        )
        count = len(non_numeric_cols)
        result.invalid_value_count += count
        result.add_error(
            f"Found {count} column(s) with non-numeric values: {names}. "
            "All gene-expression values must be numeric."
        )


def _check_missing_values(df: pd.DataFrame, result: ValidationResult) -> None:
    """Detect NaN/null values and report their count and locations."""
    missing_count = int(df.isna().sum().sum())
    result.missing_value_count = missing_count

    if missing_count == 0:
        return

    # Find which columns have missing values
    cols_with_missing = df.columns[df.isna().any()].tolist()
    col_names = [str(c) for c in cols_with_missing[:5]]
    col_info = ", ".join(col_names) + (" ..." if len(cols_with_missing) > 5 else "")

    result.add_error(
        f"Found {missing_count:,} missing value(s) across "
        f"{len(cols_with_missing)} column(s): {col_info}. "
        "Gene-expression data must not contain missing values."
    )


def _check_infinite_values(df: pd.DataFrame, result: ValidationResult) -> None:
    """Detect positive and negative infinity values."""
    # Only check numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    inf_mask = np.isinf(numeric_df.values)
    inf_count = int(inf_mask.sum())

    if inf_count == 0:
        return

    result.invalid_value_count += inf_count

    # Identify which columns have infinities
    inf_cols_mask = inf_mask.any(axis=0)
    inf_cols = numeric_df.columns[inf_cols_mask].tolist()
    col_names = [str(c) for c in inf_cols[:5]]
    col_info = ", ".join(col_names) + (" ..." if len(inf_cols) > 5 else "")

    result.add_error(
        f"Found {inf_count} infinite value(s) in "
        f"{len(inf_cols)} column(s): {col_info}. "
        "Please remove or replace infinite values."
    )


def _check_single_sample(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate that exactly one patient/sample row is present."""
    if df.shape[0] == 1:
        return

    if df.shape[0] == 0:
        result.add_error("No patient samples found in the uploaded file.")
    elif df.shape[0] > 1:
        result.add_error(
            f"Expected exactly 1 patient sample, but received "
            f"{df.shape[0]} rows. Upload a CSV with a single patient."
        )


# ── File-level readability check (pre-DataFrame) ────────────────────────────


def validate_file_content(raw_bytes: bytes) -> tuple[bool, str]:
    """Quick check that uploaded bytes look like a valid CSV.

    This runs *before* ``pd.read_csv`` and catches the most common
    failure modes (empty file, binary garbage).

    Parameters
    ----------
    raw_bytes : bytes
        Raw content of the uploaded file.

    Returns
    -------
    (ok, message) : tuple[bool, str]
        ``ok`` is ``True`` if the content looks parsable.
        ``message`` is an error string when ``ok`` is ``False``.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        return False, "The uploaded file is empty (0 bytes)."

    # Attempt to decode as UTF-8 (most common CSV encoding)
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except UnicodeDecodeError:
            return False, (
                "The file could not be decoded as UTF-8 or Latin-1. "
                "Please ensure the CSV is saved in a supported encoding."
            )

    stripped = text.strip()
    if not stripped:
        return False, "The uploaded file contains only whitespace."

    # Should have at least one header-like line
    lines = stripped.splitlines()
    if len(lines) < 1:
        return False, "The file appears to have no content."

    return True, ""
