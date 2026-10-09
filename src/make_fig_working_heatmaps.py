"""
The two working per-class heatmaps in results/figures/:

  fig2_perclass_f1_heatmap.png         absolute per-class F1 under the E4 group split
  fig3_perclass_f1_change_heatmap.png  per-class F1 change, E4 minus E2

These are the analysis-bench versions, with full class and model names. The paper ships a
tighter descendant of the second one (`Conference/Paper/fig_perclass_delta.png`, built by
make_fig_perclass_delta.py); these two are for reading at the desk, not for the 6-page limit.

Written 26 Jul 2026. Both figures were first produced on 24 Jul by throwaway code that was
never committed, and both had "E3b" baked into their titles, so neither could be relabelled
after the E3b to E4 rename without regenerating them. Values come from the result CSVs, so
the titles and the numbers cannot drift apart again.

Inputs:  results/e4_corrected_group_perclass_<MODEL>.csv   (fig2)
         results/e4_vs_e2_perclass_comparison.csv          (fig3)
Output:  results/figures/fig2_perclass_f1_heatmap.png
         results/figures/fig3_perclass_f1_change_heatmap.png

Run:  python src/make_fig_working_heatmaps.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

MODEL_ORDER = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN", "LinearSVM_SGD"]
MODEL_LABEL = {
    "RandomForest": "Random Forest",
    "DecisionTree": "Decision Tree",
    "LogisticRegression": "Logistic Regression",
    "KNN": "k-Nearest Neighbors",
    "LinearSVM_SGD": "Linear SVM",
}
XLABEL = "Attack class  (ordered by test-set support, high → low)"


def text_colour(rgba):
    """Black on light cells, white on dark ones, decided by perceived luminance."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"


def draw(vals, classes, cmap, norm, title, cbar_label, fmt, outname, figsize, footnote=None):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=14)
    ax.set_xlabel(XLABEL, fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(top=False, bottom=True, left=False, right=False, length=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(
                j, i, fmt(v), ha="center", va="center", fontsize=11,
                color=text_colour(sm.to_rgba(v)),
            )

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color("black")

    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.02)
    cbar.set_label(cbar_label, fontsize=12, labelpad=10)
    cbar.outline.set_linewidth(1.0)

    if footnote:
        fig.text(0.5, 0.015, footnote, ha="center", fontsize=13, color="0.35")

    fig.tight_layout(rect=(0, 0.05 if footnote else 0, 1, 1))
    out = os.path.join(FIGDIR, outname)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig2():
    frames = {}
    for m in MODEL_ORDER:
        p = os.path.join(RESDIR, f"e4_corrected_group_perclass_{m}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")
        frames[m] = pd.read_csv(p).set_index("class")

    support = frames[MODEL_ORDER[0]]["test_support_mean"].sort_values(ascending=False)
    classes = list(support.index)
    vals = np.array([frames[m].loc[classes, "f1_mean"].to_numpy() for m in MODEL_ORDER])

    draw(
        vals, classes, "viridis", plt.Normalize(vmin=0.0, vmax=1.0),
        "Per-class F1 score under the leakage-free group split (E4, Engelen-corrected)",
        "F1 score", lambda v: f"{v:.2f}",
        "fig2_perclass_f1_heatmap.png", (17.74, 6.71),
    )
    return pd.DataFrame(vals, index=[MODEL_LABEL[m] for m in MODEL_ORDER], columns=classes)


def fig3():
    p = os.path.join(RESDIR, "e4_vs_e2_perclass_comparison.csv")
    d = pd.read_csv(p)
    support = d.drop_duplicates(subset=["class"]).set_index("class")["test_n"]
    classes = list(support.sort_values(ascending=False).index)
    mat = d.pivot(index="model", columns="class", values="d_f1").reindex(
        index=MODEL_ORDER, columns=classes
    )
    if mat.isna().any().any():
        raise SystemExit("missing model x class cells in the comparison CSV")
    vals = mat.to_numpy()

    # Asymmetric on purpose, matching the 24 Jul original: the drops reach -1.0 while the
    # largest gain is about +0.07, so a symmetric scale would wash the gains out entirely.
    vmax = max(float(np.nanmax(vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=vmax)

    def fmt(v):
        return "0.00" if abs(v) < 5e-3 else f"{v:+.2f}"

    draw(
        vals, classes, "RdBu", norm,
        "Change in per-class F1: leakage-free group split − random split  (E4 − E2)",
        "Δ F1  (red = drop under honest evaluation)", fmt,
        "fig3_perclass_f1_change_heatmap.png", (17.75, 7.13),
        footnote="Only the low-support micro-classes (right) lose F1; "
                 "the well-supported classes (left) are unchanged.",
    )
    return pd.DataFrame(
        [[fmt(v) for v in row] for row in vals],
        index=[MODEL_LABEL[m] for m in MODEL_ORDER], columns=classes,
    )


if __name__ == "__main__":
    print(fig2().round(2).to_string(), "\n")
    print(fig3().to_string())
