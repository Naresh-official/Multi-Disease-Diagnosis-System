"""
Canonical prediction pipeline.

This is the **single source of truth** for transforming a raw
patient gene-expression CSV row into the final 80-feature vector
that the DNN expects.  Every page that needs to predict must call
:func:`preprocess` — never reimplement the steps inline.

Pipeline
--------
raw row  →  StandardScaler  →  Variance Threshold
         →  PSO column select  →  ACO column select
         →  ``np.ndarray(1, 80)``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import APP_CONFIG
from core.predictor import load_artifacts


def preprocess(raw_input: pd.DataFrame) -> np.ndarray:
    """Apply the full preprocessing chain to a single-patient DataFrame.

    Parameters
    ----------
    raw_input : pd.DataFrame
        Exactly **one row** of raw gene-expression values with
        ``TOTAL_FEATURES`` columns.

    Returns
    -------
    np.ndarray of shape ``(1, SELECTED_FEATURES)`` ready for the DNN.

    Raises
    ------
    ValueError
        If the input does not contain exactly one row or the expected
        number of features.
    """
    if raw_input.shape[0] != 1:
        raise ValueError(
            f"Expected exactly 1 patient row, got {raw_input.shape[0]}."
        )

    artifacts = load_artifacts()

    X = raw_input.values.astype(np.float64)  # (1, N)

    # 1) StandardScaler
    X_scaled: np.ndarray = artifacts["scaler"].transform(X)

    # 2) Variance threshold
    X_var: np.ndarray = artifacts["selector"].transform(X_scaled)

    # 3) PSO feature selection (column indices)
    X_pso: np.ndarray = X_var[:, artifacts["pso_features"]]

    # 4) ACO feature selection (column indices from the PSO subset)
    X_final: np.ndarray = X_pso[:, artifacts["aco_features"]]  # (1, 80)

    return X_final
