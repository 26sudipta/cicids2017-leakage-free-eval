"""
Figure: per-class F1 delta from the E2 random split to the E4 leakage-free group split.

This is Fig. 1 of the ICISET 2026 paper (\\label{fig:delta}).

Written 26 Jul 2026 to replace the original one-off plotting code, which was never
committed. The figure it produces reproduces the 25 Jul PNG value-for-value; the only
intended change is the colourbar label, which said "E3b" before the experiment was
renamed E3b -> E4. Keeping this script in the repo means a future label or metric
change does not require reverse-engineering the plot from pixels again.

Input:   results/e4_vs_e2_perclass_comparison.csv   (written by eval_perclass.py)
Output:  <paper_dir>/fig_perclass_delta.png

Run:  python src/make_fig_perclass_delta.py
      PAPER_DIR=/path/to/Conference/Paper python src/make_fig_perclass_delta.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
PAPER_DIR = os.environ.get("PAPER_DIR") or os.path.join(
    HERE, "..", "..", "Conference", "Paper"
)
SRC_CSV = os.path.join(RESDIR, "e4_vs_e2_perclass_comparison.csv")
OUT_PNG = os.path.join(PAPER_DIR, "fig_perclass_delta.png")

# Row order: strongest models first, so the reader meets the three models that
# actually learn the micro-classes (RF, DT, LR) before the two that do not.
MODEL_ORDER = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN", "LinearSVM_SGD"]
MODEL_LABEL = {
    "RandomForest": "RF",
    "DecisionTree": "DT",
    "LogisticRegression": "LR",
    "KNN": "k-NN",
    "LinearSVM_SGD": "SVM",
}
# Long CICIDS2017 class names do not fit on a tick; these are the paper's short forms.
CLASS_LABEL = {
    "DoS Slowhttptest": "DoS Slowhttp",
    "Web Attack - Brute Force": "WA Brute Force",
    "Web Attack - XSS": "WA XSS",
    "Web Attack - Sql Injection": "WA SQL Inj.",
}

VMIN, VMAX = -1.0, 1.0          # symmetric, so white is exactly "no change"
TEXT_WHITE_ABOVE = 0.55         # annotation flips to white on the dark end of RdBu


def signed(v):
    """Format a delta the way the paper does: explicit sign, but plain 0.00 for no change."""
    if abs(v) < 5e-3:
        return "0.00"
    return f"{v:+.2f}"


def main():
    d = pd.read_csv(SRC_CSV)

    missing = [m for m in MODEL_ORDER if m not in set(d["model"])]
    if missing:
        raise SystemExit(f"models absent from {SRC_CSV}: {missing}")

    # Classes ordered by test-set support, high to low. Support is a property of the
    # split, not of the model, so take it from any one model's rows.
    support = (
        d.drop_duplicates(subset=["class"])
        .set_index("class")["test_n"]
        .sort_values(ascending=False)
    )
    classes = list(support.index)

    mat = (
        d.pivot(index="model", columns="class", values="d_f1")
        .reindex(index=MODEL_ORDER, columns=classes)
    )
    if mat.isna().any().any():
        raise SystemExit("missing model x class cells; cannot draw a complete heatmap")
    vals = mat.to_numpy()

    fig, ax = plt.subplots(figsize=(14.32, 4.70))
    im = ax.imshow(vals, cmap="RdBu", vmin=VMIN, vmax=VMAX, aspect="auto")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(
        [CLASS_LABEL.get(c, c) for c in classes], rotation=45, ha="right", fontsize=11
    )
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=13)
    ax.set_xlabel(
        "attack class (ordered by test-set support, high to low)", fontsize=13, labelpad=8
    )
    ax.tick_params(top=False, bottom=True, left=False, right=False, length=3)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(
                j, i, signed(v),
                ha="center", va="center", fontsize=10,
                color="white" if abs(v) > TEXT_WHITE_ABOVE else "black",
            )

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color("black")

    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.026)
    cbar.set_label(r"$\Delta$F1  (E4 $-$ E2)", fontsize=12, labelpad=10)
    cbar.set_ticks(np.arange(-1.0, 1.01, 0.25))
    cbar.outline.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG}")

    # Echo the plotted matrix so the numbers can be diffed against the CSV by eye.
    shown = pd.DataFrame(
        [[signed(v) for v in row] for row in vals],
        index=[MODEL_LABEL[m] for m in MODEL_ORDER],
        columns=[CLASS_LABEL.get(c, c) for c in classes],
    )
    print(shown.to_string())


if __name__ == "__main__":
    main()
