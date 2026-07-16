"""
2.2 (E2) - Full metrics for the CORRECTED dataset: per-class precision/recall/F1, FPR, and
confusion matrices. Identical logic to eval_metrics.py (E1); only the cache, class names, and
output filenames point at E2 so the E1 tables/figures are never overwritten.

For each model it retrains on the cached E2 prepped arrays, then saves:
  results/e2_perclass_<MODEL>.csv       per-class precision/recall/F1/support + FPR + macro/weighted
  results/e2_confusion_<MODEL>.csv      raw confusion matrix (counts)
  results/figures/confusion_e2_<MODEL>.png   row-normalised heatmap

Run:   python src/eval_metrics_e2.py                    # all 5 models
       MODELS=RandomForest python src/eval_metrics_e2.py   # just one (comma-separated)
       NPZ=/path/to/other.npz python src/eval_metrics_e2.py  # override the cache (used for tests)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
HERE = os.path.dirname(__file__)
RESDIR = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)
# E2 cache (tagged 'full' by run_e2_corrected.py). Override with NPZ=... for a smoke test.
NPZ = os.environ.get("NPZ", os.path.join(RESDIR, "e2_prepped_arrays_full.npz"))
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))

# Load the exact class names the E2 LabelEncoder produced (guarantees index->name is correct).
CLASSES_PATH = os.environ.get("CLASSES", os.path.join(RESDIR, "label_classes_e2.npy"))
CLASSES = list(np.load(CLASSES_PATH, allow_pickle=True))

ALL_MODELS = {
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=SEED),
}

wanted = [m for m in os.environ.get("MODELS", "").split(",") if m] or list(ALL_MODELS)

d = np.load(NPZ, allow_pickle=True)
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
present = np.unique(np.concatenate([ytr, yte]))
names = [CLASSES[i] for i in present]


def fpr_per_class(cm):
    """One-vs-rest false-positive rate per class: FP / (FP + TN). Low FPR matters in security."""
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    return fp / np.clip(fp + tn, 1, None)


for name in wanted:
    clf = ALL_MODELS[name]
    Xf, yf = Xtr, ytr
    if name == "KNN" and 0 < KNN_MAX_TRAIN < len(Xtr):
        Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                        stratify=ytr, random_state=SEED)
    print(f"[{name}] training on {len(Xf):,} rows ...", flush=True)
    clf.fit(Xf, yf)
    pred = clf.predict(Xte)

    # --- per-class table (+ FPR) ---
    rep = classification_report(yte, pred, labels=present, target_names=names,
                                digits=4, zero_division=0, output_dict=True)
    table = pd.DataFrame(rep).T
    cm = confusion_matrix(yte, pred, labels=present)
    fpr = fpr_per_class(cm)
    table["FPR"] = pd.Series(dict(zip(names, fpr)))
    table.to_csv(os.path.join(RESDIR, f"e2_perclass_{name}.csv"))

    # --- raw confusion matrix (counts) ---
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        os.path.join(RESDIR, f"e2_confusion_{name}.csv"))

    # --- row-normalised heatmap ---
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} - row-normalised confusion matrix (corrected CICIDS2017, WTMC2021)")
    for i in range(len(names)):
        for j in range(len(names)):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"confusion_e2_{name}.png"), dpi=130)
    plt.close(fig)

    macro = table.loc["macro avg"]
    print(f"[{name}] macro-F1={macro['f1-score']:.4f}  saved perclass+confusion CSV and figure",
          flush=True)

print("Done.")
