"""
Centralized constants for the Explainable Multi-Disease Diagnosis System.

All disease labels, feature dimensions, artifact paths, and threshold
values live here so that every module references a single source of truth.
"""

from pathlib import Path

# ── Project root (this file lives in utils/, project root is one level up) ──
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Artifact paths ──────────────────────────────────────────────────────────
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

MODEL_PATH: Path = ARTIFACTS_DIR / "hybrid_pso_aco_dl_model.keras"
SCALER_PATH: Path = ARTIFACTS_DIR / "scaler.pkl"
VARIANCE_SELECTOR_PATH: Path = ARTIFACTS_DIR / "variance_selector.pkl"
PSO_FEATURES_PATH: Path = ARTIFACTS_DIR / "pso_features.pkl"
ACO_FEATURES_PATH: Path = ARTIFACTS_DIR / "aco_features.pkl"
X_TRAIN_FS_PATH: Path = ARTIFACTS_DIR / "X_train_fs.npy"

# ── Feature dimensions ──────────────────────────────────────────────────────
TOTAL_FEATURES: int = 10_571     # raw gene-expression probes after scaling
SELECTED_FEATURES: int = 80      # final biomarkers after PSO + ACO

# ── Disease / class metadata ────────────────────────────────────────────────
DISEASE_MAP: dict[int, str] = {
    0: "Coronary Artery Disease",
    1: "Autoimmune Disease",
    2: "Alzheimer's Disease",
    3: "Parkinson's Disease",
}

DISEASE_SHORT_NAMES: dict[int, str] = {
    0: "CAD",
    1: "Autoimmune",
    2: "Alzheimer's",
    3: "Parkinson's",
}

CLASS_NAMES: list[str] = list(DISEASE_MAP.values())
NUM_CLASSES: int = len(DISEASE_MAP)

# ── GEO dataset references ──────────────────────────────────────────────────
GEO_DATASETS: dict[str, str] = {
    "Coronary Artery Disease": "GSE66360",
    "Autoimmune Disease": "GSE50772",
    "Alzheimer's Disease": "GSE48350",
    "Parkinson's Disease": "GSE6613",
}

# ── Explanation defaults ────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES: int = 50
SHAP_MAX_DISPLAY: int = 10
LIME_NUM_FEATURES: int = 10

# ── Confidence thresholds (for future use) ─────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD: float = 0.90
LOW_CONFIDENCE_THRESHOLD: float = 0.50
