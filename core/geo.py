"""Parser for plain-text GEO Series Matrix expression files."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Return expression (samples × probes), GSM IDs, annotations, and titles."""
    metadata_rows: list[list[str]] = []
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as stream:
        for line in stream:
            if line.strip().lower() == "!series_matrix_table_begin":
                header = next(csv.reader([next(stream)], delimiter="\t"))
                sample_ids = [value.strip('"') for value in header[1:]]
                break
            if line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                metadata_rows.append([value.strip('"').strip() for value in row])
        else:
            raise ValueError(f"No expression table found in {path}")

        feature_rows: dict[str, list[float]] = {}
        for line in stream:
            if line.strip().lower() == "!series_matrix_table_end":
                break
            row = next(csv.reader([line], delimiter="\t"))
            if len(row) < 2:
                continue
            probe = row[0].strip('"').strip()
            if not probe or probe.startswith("!") or probe == "ID_REF":
                continue
            values = []
            for item in row[1:len(sample_ids) + 1]:
                try:
                    values.append(float(item))
                except (ValueError, TypeError):
                    values.append(np.nan)
            if len(values) != len(sample_ids):
                continue
            if probe in feature_rows:
                old = np.asarray(feature_rows[probe], dtype=float)
                new = np.asarray(values, dtype=float)
                with np.errstate(invalid="ignore"):
                    feature_rows[probe] = np.nanmean(np.vstack([old, new]), axis=0).tolist()
            else:
                feature_rows[probe] = values

    metadata = [""] * len(sample_ids)
    titles = sample_ids.copy()
    for row in metadata_rows:
        if row and row[0].lower() == "!sample_title":
            titles = (row[1:len(sample_ids) + 1] + sample_ids)[:len(sample_ids)]
        if row and row[0].lower() in {
            "!sample_title", "!sample_source_name_ch1", "!sample_characteristics_ch1"
        }:
            for idx, item in enumerate(row[1:len(sample_ids) + 1]):
                metadata[idx] += " | " + item

    matrix = pd.DataFrame.from_dict(feature_rows, orient="index", columns=sample_ids).T
    matrix.index.name = "sample_id"
    return matrix, sample_ids, metadata, [title.strip() for title in titles]
