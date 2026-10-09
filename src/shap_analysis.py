"""
shap_analysis.py — per-attack-class SHAP interpretability (roadmap task 3.1b, contribution C4).

Run on the best-performing model(s) from the leakage-free-group-split experiment
(2.4b), on both original and WTMC2021-corrected data. Compare feature attributions
for attack classes that degrade heavily under the leakage-free split vs. classes
that stay stable — this is the paper's interpretability contribution, not attempted
by Moczkodan & Ragab 2026 or Bouke et al. 2026 in this exact corrected-vs-original
setting (see novelty-check-1-6-findings memory).

SUPERSEDED (24 Jul): implemented as src/run_shap_e4.py — use that. This stub is kept
only because the mounted folder blocks deletes from the sandbox; safe to delete manually.
"""

import shap


def compute_shap_values(model, X_sample):
    raise NotImplementedError("TODO: task 3.1b")


def per_class_shap_comparison(model, X_test, y_test, class_labels: list[str]):
    """Compare mean |SHAP value| per feature, split by attack class."""
    raise NotImplementedError("TODO: task 3.1b")
