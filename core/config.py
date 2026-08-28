"""
Application configuration.

Centralises every tunable parameter so that page modules and
the prediction pipeline never import raw paths or magic numbers
directly — they import ``APP_CONFIG`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from utils.constants import (
    ACO_FEATURES_PATH,
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DISEASE_MAP,
    HIGH_CONFIDENCE_THRESHOLD,
    LIME_NUM_FEATURES,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    NUM_CLASSES,
    PSO_FEATURES_PATH,
    SCALER_PATH,
    SELECTED_FEATURES,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_MAX_DISPLAY,
    TOTAL_FEATURES,
    VARIANCE_SELECTOR_PATH,
    X_TRAIN_FS_PATH,
)


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    # Paths
    artifacts_dir: Path = ARTIFACTS_DIR
    model_path: Path = MODEL_PATH
    scaler_path: Path = SCALER_PATH
    variance_selector_path: Path = VARIANCE_SELECTOR_PATH
    pso_features_path: Path = PSO_FEATURES_PATH
    aco_features_path: Path = ACO_FEATURES_PATH
    x_train_fs_path: Path = X_TRAIN_FS_PATH

    # Feature dimensions
    total_features: int = TOTAL_FEATURES
    selected_features: int = SELECTED_FEATURES

    # Class metadata
    disease_map: dict[int, str] = field(default_factory=lambda: dict(DISEASE_MAP))
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    num_classes: int = NUM_CLASSES

    # XAI defaults
    shap_background_samples: int = SHAP_BACKGROUND_SAMPLES
    shap_max_display: int = SHAP_MAX_DISPLAY
    lime_num_features: int = LIME_NUM_FEATURES

    # Confidence thresholds
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD


# Singleton — import this everywhere.
APP_CONFIG = AppConfig()
