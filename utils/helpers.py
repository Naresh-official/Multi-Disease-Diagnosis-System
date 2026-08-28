"""
Shared helper utilities used across the application.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_percentage(value: float) -> str:
    """Return a float (0–1) formatted as a percentage string, e.g. '95.32%'."""
    return f"{value * 100:.2f}%"


def build_probability_dataframe(
    probabilities: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Create a tidy DataFrame mapping class names to their probabilities."""
    return pd.DataFrame({
        "Disease": class_names,
        "Probability (%)": probabilities * 100,
    })


def gene_feature_names(n: int) -> list[str]:
    """Return generic feature names ['Gene_0', …, 'Gene_{n-1}']."""
    return [f"Gene_{i}" for i in range(n)]


def get_expected_feature_names() -> list[str]:
    """Get the expected 10,571 probe IDs by intersecting all GSE files.

    The StandardScaler was trained on numpy arrays, so feature_names_in_
    is unavailable.  This function re-derives the expected probe names
    by parsing the raw GSE Series Matrix files and intersecting their
    probe sets.
    """
    import gzip
    from pathlib import Path

    from utils.constants import GSE_FILES, PROJECT_ROOT

    all_probe_sets: list[set[str]] = []
    for fname in GSE_FILES.values():
        fpath = str(PROJECT_ROOT / fname)
        with gzip.open(fpath, "rt", errors="replace") as fh:
            lines = fh.readlines()
        begin_idx = end_idx = None
        for i, line in enumerate(lines):
            if "!series_matrix_table_begin" in line:
                begin_idx = i
            if "!series_matrix_table_end" in line:
                end_idx = i
                break
        probes = set()
        for i in range(begin_idx + 2, end_idx):
            parts = lines[i].strip().split("\t")
            pid = parts[0].strip('"')
            if pid and not pid.startswith("!") and not pid.startswith('"'):
                probes.add(pid)
        all_probe_sets.append(probes)

    common = all_probe_sets[0]
    for ps in all_probe_sets[1:]:
        common &= ps
    return sorted(common)
