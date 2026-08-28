"""
PDF report generator for the Explainable Multi-Disease Diagnosis System.

Generates a professional, research-oriented PDF report from a
single prediction.  The report is intended for university
demonstration, research analysis, and educational decision-support
purposes only — it is NOT a clinical diagnosis document.

All patient data is handled in-memory and never persisted to disk.
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.figure
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.config import APP_CONFIG
from explainability.shap_explainer import extract_top_features as shap_top
from utils.helpers import format_percentage
from utils.uncertainty import UncertaintyResult


# ── Colour palette ──────────────────────────────────────────────────────────

_PRIMARY = colors.HexColor("#1a5276")
_ACCENT = colors.HexColor("#2e86c1")
_LIGHT_BG = colors.HexColor("#eaf2f8")
_SUCCESS = colors.HexColor("#27ae60")
_WARNING = colors.HexColor("#e67e22")
_DARK_TEXT = colors.HexColor("#2c3e50")
_HEADER_BG = colors.HexColor("#1a5276")


# ── Data containers (passed into the generator) ─────────────────────────────


class SampleInfo:
    """Metadata about the uploaded sample."""

    def __init__(
        self,
        sample_id: str = "N/A",
        n_input_features: int = APP_CONFIG.total_features,
        n_selected_features: int = APP_CONFIG.selected_features,
        analysis_time: datetime | None = None,
    ) -> None:
        self.sample_id = sample_id
        self.n_input_features = n_input_features
        self.n_selected_features = n_selected_features
        self.analysis_time = analysis_time or datetime.now(timezone.utc)


class PredictionResult:
    """Container for model prediction outputs."""

    def __init__(
        self,
        predicted_disease: str,
        predicted_class: int,
        probabilities: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        self.predicted_disease = predicted_disease
        self.predicted_class = predicted_class
        self.probabilities = probabilities
        self.class_names = class_names or APP_CONFIG.class_names


# ── Styles ──────────────────────────────────────────────────────────────────


def _build_styles() -> dict[str, ParagraphStyle]:
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=22,
        leading=26,
        textColor=_PRIMARY,
        spaceAfter=4 * mm,
        alignment=TA_CENTER,
    )

    styles["Subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontSize=12,
        leading=14,
        textColor=_ACCENT,
        spaceAfter=6 * mm,
        alignment=TA_CENTER,
    )

    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontSize=14,
        leading=17,
        textColor=_PRIMARY,
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
        borderPadding=(0, 0, 2, 0),
    )

    styles["Body"] = ParagraphStyle(
        "ReportBody",
        parent=base["Normal"],
        fontSize=10,
        leading=13,
        textColor=_DARK_TEXT,
        spaceAfter=2 * mm,
    )

    styles["BodyBold"] = ParagraphStyle(
        "ReportBodyBold",
        parent=styles["Body"],
        fontName="Helvetica-Bold",
    )

    styles["Small"] = ParagraphStyle(
        "ReportSmall",
        parent=base["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.grey,
    )

    styles["Disclaimer"] = ParagraphStyle(
        "Disclaimer",
        parent=base["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )

    styles["Centered"] = ParagraphStyle(
        "Centered",
        parent=styles["Body"],
        alignment=TA_CENTER,
    )

    styles["PipelineStep"] = ParagraphStyle(
        "PipelineStep",
        parent=base["Normal"],
        fontSize=10,
        leading=13,
        textColor=_DARK_TEXT,
        alignment=TA_CENTER,
        spaceBefore=1 * mm,
        spaceAfter=1 * mm,
    )

    return styles


# ── Table helpers ───────────────────────────────────────────────────────────


def _make_table(
    data: list[list[str]],
    col_widths: list[float] | None = None,
    header: bool = True,
) -> Table:
    """Build a styled Platypus Table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)

    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("TEXTCOLOR", (0, 0), (-1, -1), _DARK_TEXT),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
    ]

    if header:
        style_cmds.extend([
            ("BACKGROUND", (0, 0), (-1, 0), _PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
        ])

    # Alternate row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _LIGHT_BG))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Image helper ────────────────────────────────────────────────────────────


def _fig_to_image(
    fig: matplotlib.figure.Figure,
    width: float = 16 * cm,
    caption: str = "",
    styles: dict[str, ParagraphStyle] | None = None,
) -> list[Any]:
    """Convert a matplotlib figure to a reportlab Image element.

    Uses a temporary file that is closed immediately (reportlab reads
    it before the context-manager cleans up, so no leaked files).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    img = Image(buf, width=width, height=width * 0.6)
    img.hAlign = "CENTER"

    elements: list[Any] = [img]
    if caption and styles:
        elements.append(Spacer(1, 2 * mm))
        elements.append(Paragraph(caption, styles["Small"]))

    return elements


# ── Section builders ────────────────────────────────────────────────────────


def _build_header(
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Page 1 header block."""
    return [
        Spacer(1, 8 * mm),
        Paragraph("Explainable Multi-Disease Diagnosis System", styles["Title"]),
        Paragraph(
            "Gene Expression-Based Multi-Disease Prediction",
            styles["Subtitle"],
        ),
        Spacer(1, 2 * mm),
    ]


def _build_sample_info(
    info: SampleInfo,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Sample metadata section."""
    elements: list[Any] = [
        Paragraph("Sample Information", styles["SectionHeading"]),
    ]

    data = [
        ["Property", "Value"],
        ["Sample ID", info.sample_id],
        ["Analysis Date (UTC)", info.analysis_time.strftime("%Y-%m-%d %H:%M:%S")],
        ["Input Features", f"{info.n_input_features:,}"],
        ["Selected Biomarkers", str(info.n_selected_features)],
    ]
    elements.append(_make_table(data, col_widths=[6 * cm, 10 * cm]))
    elements.append(Spacer(1, 4 * mm))
    return elements


def _build_prediction(
    prediction: PredictionResult,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Prediction result section."""
    elements: list[Any] = [
        Paragraph("Model Prediction", styles["SectionHeading"]),
    ]

    # Predicted disease highlight
    max_prob = float(np.max(prediction.probabilities))
    elements.append(
        Paragraph(
            f'<b>Predicted Disease:</b> {prediction.predicted_disease}',
            styles["BodyBold"],
        )
    )
    elements.append(
        Paragraph(
            f'<b>Highest Model Probability:</b> {format_percentage(max_prob)}',
            styles["Body"],
        )
    )
    elements.append(Spacer(1, 3 * mm))

    # Probability distribution table
    prob_data = [["Disease", "Probability"]]
    for name, prob in zip(prediction.class_names, prediction.probabilities):
        prob_data.append([name, format_percentage(float(prob))])

    elements.append(_make_table(prob_data, col_widths=[8 * cm, 5 * cm]))
    elements.append(Spacer(1, 4 * mm))
    return elements


def _build_pipeline(
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Model pipeline diagram."""
    elements: list[Any] = [
        Paragraph("Model Pipeline", styles["SectionHeading"]),
    ]

    steps = [
        f"{APP_CONFIG.total_features:,} input features",
        "Variance Filtering",
        "PSO (Particle Swarm Optimisation)",
        "ACO (Ant Colony Optimisation)",
        f"{APP_CONFIG.selected_features} selected biomarkers",
        "Deep Neural Network (DNN)",
        "4-class softmax prediction",
    ]

    # Build a visual pipeline as a simple table
    pipeline_data = [["Step", "Description"]]
    for i, step in enumerate(steps, 1):
        arrow = " ↓" if i < len(steps) else ""
        pipeline_data.append([str(i), f"{step}{arrow}"])

    elements.append(_make_table(pipeline_data, col_widths=[2 * cm, 14 * cm]))
    elements.append(Spacer(1, 4 * mm))
    return elements


def _build_explainability(
    shap_fig: matplotlib.figure.Figure | None,
    lime_fig: matplotlib.figure.Figure | None,
    shap_values: np.ndarray | None,
    pred_class: int,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """SHAP and LIME explanation charts."""
    elements: list[Any] = [
        Paragraph("Explainability Analysis", styles["SectionHeading"]),
        Paragraph(
            "The following visualisations show which gene features most "
            "influenced the model's prediction for this sample.",
            styles["Body"],
        ),
    ]

    if shap_fig is not None:
        elements.append(Spacer(1, 2 * mm))
        elements.append(
            Paragraph("<b>SHAP Explanation</b>", styles["Body"])
        )
        elements.extend(
            _fig_to_image(
                shap_fig,
                width=15 * cm,
                caption=(
                    "SHAP bar chart: feature contributions to the predicted "
                    "class.  Positive values push toward the prediction; "
                    "negative values push against it."
                ),
                styles=styles,
            )
        )
        elements.append(Spacer(1, 4 * mm))

    if lime_fig is not None:
        elements.append(
            Paragraph("<b>LIME Explanation</b>", styles["Body"])
        )
        elements.extend(
            _fig_to_image(
                lime_fig,
                width=15 * cm,
                caption=(
                    "LIME bar chart: local interpretable explanations "
                    "showing feature contributions to the prediction."
                ),
                styles=styles,
            )
        )
        elements.append(Spacer(1, 4 * mm))

    return elements


def _build_biomarkers(
    shap_values: np.ndarray | None,
    pred_class: int,
    top_n: int = 10,
    styles: dict[str, ParagraphStyle] | None = None,
) -> list[Any]:
    """Top biomarkers table."""
    if shap_values is None or styles is None:
        return []

    top_features = shap_top(shap_values, pred_class, top_n=top_n)

    elements: list[Any] = [
        Paragraph("Top Biomarkers", styles["SectionHeading"]),
        Paragraph(
            f"Top {top_n} gene features ranked by SHAP importance for "
            f"the predicted class.",
            styles["Body"],
        ),
    ]

    data = [["Rank", "Feature", "Importance", "Direction"]]
    for feat in top_features:
        direction_label = (
            "Positive (+)" if feat["direction"] == "positive" else "Negative (−)"
        )
        data.append([
            str(feat["rank"]),
            feat["feature_name"],
            f'{feat["importance"]:.4f}',
            direction_label,
        ])

    elements.append(
        _make_table(data, col_widths=[1.5 * cm, 4 * cm, 3.5 * cm, 4 * cm])
    )
    elements.append(Spacer(1, 4 * mm))
    return elements


def _build_uncertainty(
    uncertainty: UncertaintyResult | None,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Uncertainty analysis section."""
    if uncertainty is None:
        return []

    elements: list[Any] = [
        Paragraph("Uncertainty Analysis", styles["SectionHeading"]),
    ]

    data = [
        ["Metric", "Value"],
        ["Maximum Probability", format_percentage(uncertainty.max_probability)],
        ["Confidence Level", uncertainty.confidence_level.value],
        ["Prediction Entropy", f"{uncertainty.entropy:.3f} bits"],
        ["Normalised Entropy", f"{uncertainty.normalised_entropy:.3f}"],
        ["Probability Sum", f"{uncertainty.probability_sum:.4f}"],
    ]

    elements.append(_make_table(data, col_widths=[6 * cm, 6 * cm]))

    if uncertainty.is_low_confidence:
        elements.append(Spacer(1, 3 * mm))
        elements.append(
            Paragraph(
                '<font color="#e67e22"><b>Note:</b> This is a low-confidence '
                "prediction. The model does not strongly distinguish between "
                "available disease classes for this sample. Results should be "
                "interpreted with caution.</font>",
                styles["Body"],
            )
        )

    elements.append(Spacer(1, 4 * mm))
    return elements


def _build_disclaimer(styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Research disclaimer footer."""
    return [
        Spacer(1, 6 * mm),
        Paragraph(
            "—" * 40,
            styles["Disclaimer"],
        ),
        Paragraph(
            "<b>Research Disclaimer</b>",
            styles["Disclaimer"],
        ),
        Paragraph(
            "This system is intended for research and educational "
            "decision-support purposes only and is not a substitute for "
            "professional medical diagnosis. The model predictions and "
            "confidence scores shown are model-internal probabilities and "
            "have NOT been clinically validated. They should NOT be used "
            "for medical diagnosis, treatment decisions, or as a substitute "
            "for professional medical advice.",
            styles["Disclaimer"],
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
            "| Explainable Multi-Disease Diagnosis System",
            styles["Small"],
        ),
    ]


# ── Page number callback ────────────────────────────────────────────────────


def _add_page_number(canvas: Any, doc: Any) -> None:
    """Draw page number in the footer of every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, text)
    canvas.restoreState()


# ── Main generator ──────────────────────────────────────────────────────────


def generate_diagnosis_report(
    sample_info: SampleInfo,
    prediction: PredictionResult,
    probabilities: np.ndarray,
    shap_image: matplotlib.figure.Figure | None = None,
    lime_image: matplotlib.figure.Figure | None = None,
    shap_values: np.ndarray | None = None,
    biomarkers: list[dict[str, Any]] | None = None,
    uncertainty: UncertaintyResult | None = None,
) -> bytes:
    """Generate a complete PDF diagnosis report and return as bytes.

    Parameters
    ----------
    sample_info : SampleInfo
        Metadata about the uploaded sample.
    prediction : PredictionResult
        Model prediction result.
    probabilities : np.ndarray
        Raw softmax probabilities.
    shap_image : matplotlib.figure.Figure, optional
        SHAP explanation chart.
    lime_image : matplotlib.figure.Figure, optional
        LIME explanation chart.
    shap_values : np.ndarray, optional
        Raw SHAP values for biomarker extraction.
    biomarkers : list of dict, optional
        Pre-computed biomarker list (overrides SHAP extraction).
    uncertainty : UncertaintyResult, optional
        Uncertainty analysis result.

    Returns
    -------
    bytes
        The complete PDF file content.
    """
    styles = _build_styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
        title="Diagnosis Report — Explainable Multi-Disease Diagnosis System",
        author="Explainable Multi-Disease Diagnosis System",
    )

    story: list[Any] = []

    # ── Page 1: Header + Sample Info + Prediction ──────────────────────
    story.extend(_build_header(styles))
    story.extend(_build_sample_info(sample_info, styles))
    story.extend(_build_prediction(prediction, styles))

    # ── Pipeline ───────────────────────────────────────────────────────
    story.extend(_build_pipeline(styles))

    # ── Explainability ─────────────────────────────────────────────────
    story.extend(
        _build_explainability(
            shap_image, lime_image, shap_values,
            prediction.predicted_class, styles,
        )
    )

    # ── Biomarkers ─────────────────────────────────────────────────────
    story.extend(
        _build_biomarkers(
            shap_values, prediction.predicted_class,
            top_n=10, styles=styles,
        )
    )

    # ── Uncertainty ────────────────────────────────────────────────────
    story.extend(_build_uncertainty(uncertainty, styles))

    # ── Disclaimer ─────────────────────────────────────────────────────
    story.extend(_build_disclaimer(styles))

    # ── Build PDF ──────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
