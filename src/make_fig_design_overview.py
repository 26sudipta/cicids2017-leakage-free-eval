"""
Figure: the evaluation grid as bars -- (a) label correction E1 vs E2, (b) split protocol
E2 vs E3 vs E4.

NOT REFERENCED BY THE PAPER. This figure was cut from the ICISET 2026 submission on
26 Jul 2026 for two reasons: it duplicated Table I at lower precision, and the version
generated on 25 Jul predated the Eq. (2) macro-F1 fix, so its E3 bars (0.09, 0.07, 0.07,
0.07, 0.06) and its decision-tree E4 bar (0.80) contradicted the table. The script is
kept, and reads every value from the results CSVs rather than hard-coding them, so the
figure cannot silently go stale again if the metric changes. Regenerate and re-add it
only if the page budget allows.

Two deliberate differences from the 25 Jul PNG:
  1. All bar values now come from the current CSVs, so they agree with Table I.
  2. The legend says E4 where it used to say E3b (experiment renamed 26 Jul).
Panel (b) plots the CALENDAR temporal split for E3, as the original did. Table I also
reports a time-sorted split; a single bar cannot show both, which was part of why the
figure was cut.

Inputs:  results/e1_baseline_results.csv
         results/e2_corrected_results.csv
         results/e3_corrected_calendar_results.csv
         results/e4_corrected_group_summary.csv
Output:  <paper_dir>/fig_design_overview.png

Run:  python src/make_fig_design_overview.py
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
OUT_PNG = os.path.join(PAPER_DIR, "fig_design_overview.png")

MODEL_ORDER = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN", "LinearSVM_SGD"]
MODEL_LABEL = {
    "RandomForest": "RF",
    "DecisionTree": "DT",
    "LogisticRegression": "LR",
    "KNN": "k-NN",
    "LinearSVM_SGD": "SVM",
}

C_E1, C_E2, C_E3, C_E4 = "#BFBFBF", "#4C72B0", "#C44E52", "#55A868"


def series(fname, value_col, index_col="model"):
    p = os.path.join(RESDIR, fname)
    if not os.path.exists(p):
        raise SystemExit(f"missing input: {p}")
    d = pd.read_csv(p).set_index(index_col)
    missing = [m for m in MODEL_ORDER if m not in d.index]
    if missing:
        raise SystemExit(f"{fname}: models absent {missing}")
    return d.loc[MODEL_ORDER, value_col].astype(float).to_numpy()


def annotate(ax, xs, ys, yerr=None):
    for k, (x, y) in enumerate(zip(xs, ys)):
        off = 0.012 + (0.0 if yerr is None else float(yerr[k]))
        ax.text(x, y + off, f"{y:.2f}", ha="center", va="bottom", fontsize=11)


def main():
    e1 = series("e1_baseline_results.csv", "macro_f1")
    e2 = series("e2_corrected_results.csv", "macro_f1")
    e3 = series("e3_corrected_calendar_results.csv", "macro_f1")
    e4 = series("e4_corrected_group_summary.csv", "macro_f1_mean")
    e4sd = series("e4_corrected_group_summary.csv", "macro_f1_std")

    labels = [MODEL_LABEL[m] for m in MODEL_ORDER]
    x = np.arange(len(labels))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.32, 5.40))

    # ---- panel (a): label correction, split held fixed -------------------------
    w = 0.38
    b1 = axa.bar(x - w / 2, e1, w, color=C_E1, edgecolor="black", linewidth=0.8)
    b2 = axa.bar(x + w / 2, e2, w, color=C_E2, edgecolor="black", linewidth=0.8)
    annotate(axa, x - w / 2, e1)
    annotate(axa, x + w / 2, e2)
    axa.set_title("(a) effect of label correction", fontsize=14, pad=12)
    axa.set_ylabel("Macro-F1", fontsize=13)

    # ---- panel (b): split protocol, labels held fixed (corrected) --------------
    w = 0.26
    b3 = axb.bar(x - w, e2, w, color=C_E2, edgecolor="black", linewidth=0.8)
    b4 = axb.bar(x, e3, w, color=C_E3, edgecolor="black", linewidth=0.8, hatch="///")
    b5 = axb.bar(
        x + w, e4, w, color=C_E4, edgecolor="black", linewidth=0.8,
        yerr=e4sd, capsize=3, error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    annotate(axb, x - w, e2)
    annotate(axb, x, e3)
    annotate(axb, x + w, e4, yerr=e4sd)
    axb.set_title("(b) effect of evaluation protocol", fontsize=14, pad=12)

    for ax in (axa, axb):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.legend(
        [b1, b3, b4, b5],
        [
            "E1: original labels, random split",
            "E2: corrected labels, random split",
            "E3: corrected labels, calendar temporal split (degenerate)",
            "E4: corrected labels, leakage-free group split",
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=12,
        bbox_to_anchor=(0.5, -0.02), columnspacing=2.4, handlelength=1.8,
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG}")

    tbl = pd.DataFrame(
        {"E1": e1, "E2": e2, "E3_cal": e3, "E4": e4, "E4_std": e4sd}, index=labels
    ).round(3)
    print(tbl.to_string())


if __name__ == "__main__":
    main()
