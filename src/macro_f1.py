"""
Canonical macro-F1 for this study.  Added 26 Jul 2026.

WHY THIS FILE EXISTS
--------------------
The paper (Eq. 2) defines macro-F1 as the mean per-class F1 over

    C+ = { c : n_c^test > 0 }

that is, only over the classes the split actually placed in the test partition.

`sklearn.metrics.f1_score(y_true, y_pred, average="macro")` does NOT do that.
Called without `labels=`, it averages over the *union* of the labels appearing
in y_true and y_pred. Whenever a model predicts a class that has no test flows,
that class joins the denominator with F1 = 0 and drags the average down.

The union is model-dependent, so five models end up with five different
denominators and their macro-F1 values stop being comparable to each other.

THIS ACTUALLY HAPPENED.  Found 26 Jul 2026 while adding the time-sorted
results to Table I.  On the corrected calendar split the denominators were
10, 14, 14, 13 and 14 across the five models, where |C+| = 8 throughout.  The
published E3 column had to be recomputed:

    model               was     now
    RandomForest        0.089   0.111
    DecisionTree        0.066   0.115
    LogisticRegression  0.066   0.115
    KNN                 0.073   0.119
    LinearSVM_SGD       0.065   0.113

E1 and E2 escaped only by luck: a stratified random split puts all 15 classes
in the test partition, so union == C+ == 15 and the two definitions coincide.
E4 was correct in 14 of 15 model-seed cells; the exception was DecisionTree
at seed 2, where the model predicted Infiltration even though Infiltration had
no test flows that seed, giving denominator 15 against |C+| = 14.

RULE
----
Use macro_f1_cplus() for any number that will be reported.  Never call
f1_score(..., average="macro") without an explicit labels= argument.

Weighted F1 is mathematically immune to this, because a class with zero test
support carries zero weight.  It is pinned here anyway so that both aggregates
are visibly computed over the same label set.
"""

import numpy as np
from sklearn.metrics import f1_score

__all__ = ["cplus", "macro_f1_cplus", "weighted_f1_cplus", "audit_macro"]


def cplus(y_true):
    """C+, the sorted array of classes with at least one test flow."""
    return np.unique(y_true)


def macro_f1_cplus(y_true, y_pred):
    """
    Macro-F1 per Eq. (2): mean per-class F1 over C+.

    Returns (score, n_cplus) so the caller can record the denominator alongside
    the number.  Log n_cplus.  A macro-F1 without its |C+| is not interpretable
    on a degenerate split.
    """
    present = cplus(y_true)
    score = f1_score(y_true, y_pred, labels=present,
                     average="macro", zero_division=0)
    return float(score), int(present.size)


def weighted_f1_cplus(y_true, y_pred):
    """Support-weighted F1 over the same label set.  Pinned for consistency."""
    present = cplus(y_true)
    return float(f1_score(y_true, y_pred, labels=present,
                          average="weighted", zero_division=0))


def audit_macro(y_true, y_pred):
    """
    Diagnostic.  Returns a dict comparing the correct value against the
    sklearn default, so a run can shout when the two disagree.

    Call this in a guard rather than trusting that the labels= argument
    survived a future refactor.
    """
    present = cplus(y_true)
    union = np.union1d(present, np.unique(y_pred))
    correct = f1_score(y_true, y_pred, labels=present,
                       average="macro", zero_division=0)
    default = f1_score(y_true, y_pred, labels=union,
                       average="macro", zero_division=0)
    return {
        "macro_f1_cplus": float(correct),
        "n_cplus": int(present.size),
        "macro_f1_sklearn_default": float(default),
        "n_union": int(union.size),
        "diverges": bool(present.size != union.size),
    }
