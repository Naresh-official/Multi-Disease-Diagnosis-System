"""
Model loading and prediction.

Heavy artefacts (Keras model, scikit-learn transformers, numpy arrays)
are loaded **once** via ``@st.cache_resource`` so that Streamlit reruns
do not pay the I/O cost again.  Every page that needs predictions
imports ``load_artifacts`` and ``predict`` from here — there is no
duplicated loading logic anywhere else in the codebase.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

from core.config import APP_CONFIG


@st.cache_resource
def load_artifacts() -> dict[str, Any]:
    """Load all model artefacts into memory (cached across reruns).

    Returns
    -------
    dict with keys: model, scaler, selector, pso_features,
    aco_features, x_train_fs
    """
    model = tf.keras.models.load_model(str(APP_CONFIG.model_path))
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    selector = joblib.load(str(APP_CONFIG.variance_selector_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))
    x_train_fs = np.load(str(APP_CONFIG.x_train_fs_path))

    return {
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "pso_features": pso_features,
        "aco_features": aco_features,
        "x_train_fs": x_train_fs,
    }


def predict(artifacts: dict[str, Any], x_final: np.ndarray) -> np.ndarray:
    """Run the DNN forward pass and return the softmax probabilities.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`load_artifacts`.
    x_final : np.ndarray
        Pre-processed feature matrix of shape ``(1, SELECTED_FEATURES)``.

    Returns
    -------
    np.ndarray of shape ``(NUM_CLASSES,)`` — class probabilities.
    """
    return artifacts["model"].predict(x_final)[0]
