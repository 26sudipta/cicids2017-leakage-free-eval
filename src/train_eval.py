"""
train_eval.py — train classifiers and compute imbalance-aware metrics.

Models (matching the base-paper / Maseer 2021 baseline family, per roadmap task 2.1):
    Logistic Regression, Decision Tree, Random Forest, k-NN, SVM (LinearSVC).

Metrics reported — never accuracy alone (see base-paper-critique memory):
    per-class precision / recall / F1, macro-F1, false-positive rate (FPR),
    detection rate (DR), confusion matrix.

Run this once per (dataset version x split type) combination:
    dataset version in {original, wtmc2021_corrected, lycos_ids2017}
    split type      in {random, naive_chronological, leakage_free_group}

Save every result row to results/metrics.csv — never retype numbers by hand
(see Paper_Roadmap_and_TODO.md "Rules that prevent the base paper's mistakes").

TODO: implement — not yet run.
"""

import pandas as pd


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """Train one model, return a dict of metrics ready to append to results/metrics.csv."""
    raise NotImplementedError("TODO: task 2.1-2.4b")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see Week 2 tasks in Paper_Roadmap_and_TODO.md")
