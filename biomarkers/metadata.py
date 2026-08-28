"""
Biomarker gene-name and pathway metadata.

IMPORTANT: The feature identifiers in this project are Affymetrix
microarray probe IDs (e.g. ``219170_at``), NOT gene symbols.  This
module documents the identifier type and provides utilities for
looking up probe-to-gene mappings if a reliable source is available.

No biological annotations are fabricated.  If no mapping is available,
the system uses the raw probe IDs and clearly states that annotation
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from core.config import APP_CONFIG


@dataclass
class ProbeInfo:
    """Information about a single Affymetrix probe.

    Attributes
    ----------
    probe_id : str
        The Affymetrix probe ID (e.g. ``'219170_at'``).
    platform : str
        The microarray platform (e.g. ``'HG-U133_Plus_2'``).
    has_gene_mapping : bool
        Whether a reliable gene symbol mapping is available.
    gene_symbol : str or None
        The gene symbol if available, else ``None``.
    gene_description : str or None
        A brief description if available, else ``None``.
    """

    probe_id: str
    platform: str = "Affymetrix HG-U133 Plus 2.0"
    has_gene_mapping: bool = False
    gene_symbol: str | None = None
    gene_description: str | None = None


def get_identifier_type() -> str:
    """Return a description of the feature identifier type.

    Returns
    -------
    str
        Always returns ``'Affymetrix probe ID'`` for this project.
    """
    return "Affymetrix probe ID"


def get_identifier_format_description() -> str:
    """Return a human-readable description of the identifier format.

    Returns
    -------
    str
        Description of what the probe IDs look like and what they mean.
    """
    return (
        "The feature identifiers in this system are Affymetrix microarray "
        "probe IDs (e.g. '219170_at'). These identifiers correspond to "
        "specific oligonucleotide probes on the Affymetrix HG-U133 Plus 2.0 "
        "gene chip. Each probe targets a specific transcript region of a gene. "
        "The suffix '_at' indicates a unique target, '_s_at' indicates a "
        "shared target (cross-hybridizing probes), and '_x_at' indicates "
        "a cross-hybridizing probe."
    )


def get_all_probe_ids() -> list[str]:
    """Get all 80 selected probe IDs (cleaned of quotes).

    Returns
    -------
    list of str
        The 80 Affymetrix probe IDs.
    """
    scaler = joblib.load(str(APP_CONFIG.scaler_path))
    pso_features = joblib.load(str(APP_CONFIG.pso_features_path))
    aco_features = joblib.load(str(APP_CONFIG.aco_features_path))

    from utils.helpers import get_expected_feature_names
    all_names = get_expected_feature_names()
    pso_names = [all_names[i] for i in pso_features]
    final_names = [pso_names[i] for i in aco_features]

    return [name.strip('"') for name in final_names]


def get_probe_info(probe_id: str) -> ProbeInfo:
    """Get information about a specific probe.

    NOTE: No biological annotation is fabricated.  The returned
    ProbeInfo has ``has_gene_mapping=False`` unless a reliable
    external mapping is loaded.

    Parameters
    ----------
    probe_id : str
        Affymetrix probe ID.

    Returns
    -------
    ProbeInfo
    """
    return ProbeInfo(
        probe_id=probe_id,
        platform="Affymetrix HG-U133 Plus 2.0",
        has_gene_mapping=False,
        gene_symbol=None,
        gene_description=None,
    )


def get_annotation_availability() -> str:
    """Return a statement about biological annotation availability.

    Returns
    -------
    str
        Clear statement that annotation is not available.
    """
    return (
        "Biological annotation (gene symbols, pathway information) is NOT "
        "available for the probe identifiers in this system. The feature "
        "identifiers are raw Affymetrix probe IDs. To obtain gene annotations, "
        "an external mapping database (e.g. BioMart, GEO query, or Affymetrix "
        "CDF file) would be required. No gene functions or disease associations "
        "are fabricated in this system."
    )
