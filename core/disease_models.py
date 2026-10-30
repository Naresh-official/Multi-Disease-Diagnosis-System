"""Loading and prediction helpers for disease-specific models."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.feature_selectors import RankedFeatureSelector  # Ensure unpickling works
from core.geo import parse_series_matrix as _parse_series_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "disease_models"

DISEASE_NAME_MAP = {
    "cad": "Coronary Artery Disease / AMI",
    "asthma": "Asthma",
    "alzheimers": "Alzheimer's Disease",
    "lung_adenocarcinoma": "Lung Adenocarcinoma",
    "heart_failure": "Heart Failure",
}


def available_disease_models() -> list[dict[str, Any]]:
    """Return the trained disease models and their metadata."""
    models = []
    if not MODEL_ROOT.exists():
        return models
    for folder in sorted(MODEL_ROOT.iterdir()):
        info_path = folder / "training_info.json"
        if folder.is_dir() and (folder / "model.joblib").is_file() and info_path.is_file():
            try:
                with info_path.open(encoding="utf-8") as f:
                    info = json.load(f)
                display_name = info.get("disease_name", DISEASE_NAME_MAP.get(folder.name, folder.name.title()))
                models.append({
                    "slug": folder.name,
                    "disease": display_name,
                    **info
                })
            except Exception:
                pass
    return models


@st.cache_resource(show_spinner=False)
def load_disease_model(slug: str) -> tuple[Any, list[str], dict[str, Any]]:
    """Load one binary model, its exact probe schema, and training metadata."""
    folder = MODEL_ROOT / slug
    model = joblib.load(folder / "model.joblib")
    with (folder / "feature_schema.json").open(encoding="utf-8") as f:
        schema = json.load(f)
    with (folder / "training_info.json").open(encoding="utf-8") as f:
        info = json.load(f)

    if isinstance(schema, dict) and "features" in schema:
        features = schema["features"]
    elif isinstance(schema, list):
        features = schema
    else:
        raise ValueError(f"Invalid feature schema format for {slug}")

    return model, features, info


def parse_expression_input(
    file_bytes: bytes, filename: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse an uploaded CSV or GEO Series Matrix file into an expression DataFrame
    with sample IDs as index and probe IDs as columns.
    """
    fname_lower = filename.lower()
    if fname_lower.endswith((".txt", ".txt.gz", ".gz")):
        suffix = ".txt.gz" if fname_lower.endswith(".gz") else ".txt"
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            matrix, sample_ids, _, titles = _parse_series_matrix(tmp_path)
            return matrix, [f"{s} — {t}" for s, t in zip(sample_ids, titles)]
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        # CSV parsing
        df = pd.read_csv(io.BytesIO(file_bytes))
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")

        # Check if first column is an index/sample_id
        first_col = df.columns[0]
        first_col_str = str(first_col).strip().lower()
        if (
            first_col_str in {"sample_id", "id", "sample", "patient_id", "patient", "gsm"}
            or df.iloc[:, 0].dtype == object
            or not str(first_col).endswith(("_at", "_s_at", "_x_at", "_g_at"))
            and not str(first_col).startswith(("ILMN_", "20", "21", "22", "10", "11", "12", "78", "79"))
        ):
            # Check if values in first column look like sample IDs
            first_col_vals = df.iloc[:, 0].astype(str)
            if any(val.startswith(("GSM", "Muse_", "A_", "indiv", "05L", "C0", "P0")) for val in first_col_vals):
                df = df.set_index(df.columns[0])
            elif not np.issubdtype(df.iloc[:, 0].dtype, np.number):
                df = df.set_index(df.columns[0])

        sample_ids = [str(idx) for idx in df.index]
        return df, sample_ids


def predict_samples(
    model: Any, expected_features: list[str], df_samples: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Reindex DataFrame according to model expected features and predict.
    Returns (predictions, probabilities, top_biomarkers).
    """
    # Reindex columns to match model's expected probe schema
    aligned = df_samples.reindex(columns=expected_features)
    aligned_numeric = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    preds = model.predict(aligned_numeric)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(aligned_numeric)
    else:
        probs = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            probs[i, p] = 1.0

    # Extract selected biomarker feature names from the selector step
    selected_probes = []
    if "selector" in model.named_steps:
        mask = model.named_steps["selector"].get_support()
        selected_probes = [feat for feat, keep in zip(expected_features, mask) if keep]

    return preds, probs, selected_probes
