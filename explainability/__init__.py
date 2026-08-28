"""Explainable-AI modules (SHAP, LIME, comparison)."""

from explainability.shap_explainer import explain as shap_explain
from explainability.lime_explainer import explain as lime_explain
from explainability.comparison import compare_explanations, ComparisonResult

__all__ = [
    "shap_explain",
    "lime_explain",
    "compare_explanations",
    "ComparisonResult",
]
