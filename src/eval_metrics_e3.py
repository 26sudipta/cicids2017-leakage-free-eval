"""
E3 full metrics: per-class precision/recall/F1 + support + FPR + confusion, for one
temporal-split condition. Mirrors eval_metrics_e2.py; the ONE addition is that it makes the
degeneracy explicit -- it reports, per class, whether the class had ZERO training support
(so the model could never learn it -> recall 0) or ZERO test support (not evaluated). Under a
naive temporal split many attack classes fall into one of these buckets; hiding that behind a
single macro-F1 would be dishonest, so we surface it.

Selection matches run_e3_temporal.py:  TAG = <dataset>_<split>, e.g. corrected_calendar.

Run:   TAG=corrected_calendar python src/eval_metrics_e3.py
       TAG=original_timesort MODELS=RandomForest python src/eval_metrics_e3.py
Outputs (per model):
   results/e3_<TAG>_perclass_<MODEL>.csv    per-class P/R/F1/support + FPR + train_support + status
   results/e3_<TAG>_confusion_<MODEL>.csv   raw confusion matrix (counts)
   results/figures/confusion_e3_<TAG>_<MODEL>.png
"""
import os, collections
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
# Eq. (2) macro-F1 over C+; see src/macro_f1.py for why rep["macro avg"] is wrong here.
from macro_f1 import macro_f1_cplus
from sklearn.model_selection import train_test_split

SEED = 42
HERE = os.path.dirname(__file__)
RESDIR = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

TAG = os.environ.get("TAG", "").strip().lower()
if not TAG:
    raise SystemExit("Set TAG=<dataset>_<split>, e.g. TAG=corrected_calendar")
CACHE_TAG = os.environ.get("CACHE_TAG", "full")
NPZ = os.environ.get("NPZ", os.path.join(RESDIR, f"e3_{TAG}_{CACHE_TAG}.npz"))
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))

d = np.load(NPZ, allow_pickle=True)
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
# Class names come FROM the npz (written by run_e3_temporal.py at split time), so the label
# indices in ytr/yte and the names can never come from different runs. Fall back to the old
# separate .npy only for npz files made before this change (and warn if it looks stale).
if "classes" in d.files:
    CLASSES = list(d["classes"])
else:
    legacy = os.environ.get("CLASSES", os.path.join(RESDIR, f"label_classes_e3_{TAG}.npy"))
    CLASSES = list(np.load(legacy, allow_pickle=True))
    print(f"WARNING: {os.path.basename(NPZ)} has no embedded classes; using {os.path.basename(legacy)}. "
          f"If you hit an index error, re-run run_e3_temporal.py for TAG={TAG} to refresh the cache.")
_maxidx = int(max(int(v) for v in np.concatenate([ytr, yte])))
if _maxidx >= len(CLASSES):
    raise SystemExit(
        f"Class list ({len(CLASSES)} names) is smaller than the max label index ({_maxidx}) in "
        f"{os.path.basename(NPZ)} — the arrays and class names are from different runs. "
        f"Re-run:  DATASET={TAG.split('_')[0]} SPLIT={TAG.split('_',1)[1]} python src/run_e3_temporal.py")

ALL_MODELS = {
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=SEED),
}
wanted = [m for m in os.environ.get("MODELS", "").split(",") if m] or list(ALL_MODELS)

# evaluate over EVERY class that appears in train OR test, so zero-support classes stay visible
present = np.unique(np.concatenate([ytr, yte]))
names = [CLASSES[i] for i in present]
train_support = collections.Counter(int(v) for v in ytr)
test_support = collections.Counter(int(v) for v in yte)


def status_of(cls_idx):
    tr = train_support.get(cls_idx, 0)
    te = test_support.get(cls_idx, 0)
    if tr == 0 and te > 0:
        return "ZERO_TRAIN (unlearnable)"
    if te == 0 and tr > 0:
        return "ZERO_TEST (not evaluated)"
    if tr == 0 and te == 0:
        return "ABSENT"
    return "ok"


def fpr_per_class(cm):
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    return fp / np.clip(fp + tn, 1, None)


for name in wanted:
    clf = ALL_MODELS[name]
    Xf, yf = Xtr, ytr
    if name == "KNN" and 0 < KNN_MAX_TRAIN < len(Xtr):
        try:
            Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                            stratify=ytr, random_state=SEED)
        except ValueError:
            Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                            random_state=SEED)
    print(f"[{name}] training on {len(Xf):,} rows ...", flush=True)
    clf.fit(Xf, yf)
    pred = clf.predict(Xte)

    rep = classification_report(yte, pred, labels=present, target_names=names,
                                digits=4, zero_division=0, output_dict=True)
    table = pd.DataFrame(rep).T
    # annotate each class row with train support + degeneracy status
    table["train_support"] = pd.Series({CLASSES[i]: train_support.get(int(i), 0) for i in present})
    table["status"] = pd.Series({CLASSES[i]: status_of(int(i)) for i in present})

    cm = confusion_matrix(yte, pred, labels=present)
    fpr = fpr_per_class(cm)
    table["FPR"] = pd.Series(dict(zip(names, fpr)))
    table.to_csv(os.path.join(RESDIR, f"e3_{TAG}_perclass_{name}.csv"))

    pd.DataFrame(cm, index=names, columns=names).to_csv(
        os.path.join(RESDIR, f"e3_{TAG}_confusion_{name}.csv"))

    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} - confusion (E3 {TAG})")
    for i in range(len(names)):
        for j in range(len(names)):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"confusion_e3_{TAG}_{name}.png"), dpi=130)
    plt.close(fig)

    # NOTE: table.loc["macro avg"] averages over `present` = classes in train OR
    # test, which is not Eq. (2). Keep `present` for the per-class rows, since the
    # ZERO_TEST rows are the disclosure the paper promises, but report macro-F1
    # over C+ only. See src/macro_f1.py.
    mf1, n_cplus = macro_f1_cplus(yte, pred)
    n_zero_train = sum(1 for i in present if status_of(int(i)).startswith("ZERO_TRAIN"))
    print(f"[{name}] macro-F1={mf1:.4f} (|C+|={n_cplus})  "
          f"({n_zero_train} classes had zero training support)", flush=True)

print("Done.")
