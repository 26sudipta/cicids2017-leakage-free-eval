"""
E4 evaluation: aggregate the group-by-five-tuple results across seeds.

Consumes the per-seed PREDICTION files written by run_e4_group.py (no retraining), so eval scores
the EXACT models the run trained -- no train/eval drift, and half the compute. Produces:

  results/e4_<dataset>_group_perclass_<MODEL>.csv   per-class precision/recall/F1 mean±std over
                                                     seeds + train/test support + zero-support flags
  results/e4_<dataset>_group_summary.csv            per-model macro-F1/acc/wF1/FPR/FNR mean±std,
                                                     with Δ macro-F1 vs the E2 random-split baseline
  results/figures/confusion_e4_<dataset>_<MODEL>.png  (first seed, for a visual)

Honesty rule: a class is only counted in the recall/precision/F1 stats for a seed where it actually
had TEST samples. A class present in train but absent from test (ZERO_TEST) is flagged, not scored as
a fake 0; a class present in test but absent from train (ZERO_TRAIN) is an evaluated-but-unlearnable
class and its recall 0 is real, so it is kept.

Run:  DATASET=corrected python src/eval_metrics_e4.py
      DATASET=corrected SEEDS=2,42,123 MODELS=RandomForest python src/eval_metrics_e4.py
"""
import os, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Eq. (2) macro-F1 over C+; see src/macro_f1.py for why rep["macro avg"] is wrong here.
from macro_f1 import macro_f1_cplus, weighted_f1_cplus

DATASET = os.environ.get("DATASET", "corrected").strip().lower()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "2,42,123").split(",") if s.strip()]
CACHE_TAG = os.environ.get("CACHE_TAG", "full")
HERE = os.path.dirname(__file__)
RESDIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)
MODELS_FILTER = [m for m in os.environ.get("MODELS", "").split(",") if m]
E2_RESULTS = os.environ.get("E2_RESULTS", os.path.join(RESDIR, "e2_corrected_results.csv"))
RESULTS_CSV = os.path.join(RESDIR, f"e4_{DATASET}_group_results.csv")


def load_preds(seed):
    p = os.path.join(RESDIR, f"e4_{DATASET}_group_seed{seed}_{CACHE_TAG}_preds.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run run_e4_group.py first (DATASET={DATASET}, seed {seed}).")
    return np.load(p, allow_pickle=True)


d0 = load_preds(SEEDS[0])
CLASSES = list(d0["classes"])
model_names = [k[len("pred_"):] for k in d0.files if k.startswith("pred_")]
if MODELS_FILTER:
    model_names = [m for m in model_names if m in MODELS_FILTER]
print(f"E4 eval  DATASET={DATASET}  seeds={SEEDS}  models={model_names}  classes={len(CLASSES)}")


BENIGN_IDX = int(CLASSES.index("BENIGN")) if "BENIGN" in CLASSES else -1


def fpr_fnr(yte, pred, benign_idx=BENIGN_IDX):
    """Identical definition to run_e4_group.py, kept here so the summary can be
    rebuilt from the saved predictions without reading the run's results CSV."""
    yte, pred = np.asarray(yte), np.asarray(pred)
    ben = yte == benign_idx
    atk = ~ben
    fpr = float(np.mean(pred[ben] != benign_idx)) if ben.any() else float("nan")
    fnr = float(np.mean(pred[atk] == benign_idx)) if atk.any() else float("nan")
    return fpr, fnr


# Per-seed aggregates recomputed from the saved predictions.  The summary used to
# be built from run_e4_group.py's results CSV, which meant a metric fix in this
# repo did not reach the summary until the whole 3-seed run was repeated.  That
# is how the DecisionTree seed-2 cell stayed at the pre-fix 0.7962 after
# src/macro_f1.py landed.  Predictions are the source of truth: they are the
# exact model outputs the run produced, so recomputing from them cannot drift.
seed_rows = []


def status_str(n_zero_train, n_zero_test, n_seeds):
    if n_zero_test >= n_seeds:  return "ZERO_TEST(all)"
    if n_zero_train >= n_seeds: return "ZERO_TRAIN(all)"
    if n_zero_train > 0:        return f"ZERO_TRAIN({n_zero_train}/{n_seeds})"
    if n_zero_test > 0:         return f"ZERO_TEST({n_zero_test}/{n_seeds})"
    return "ok"


# ---- per-class aggregation across seeds -------------------------------------
for name in model_names:
    prec = collections.defaultdict(list)
    rec = collections.defaultdict(list)
    f1 = collections.defaultdict(list)
    tr_sup = collections.defaultdict(list)
    te_sup = collections.defaultdict(list)
    zero_train = collections.Counter()
    zero_test = collections.Counter()
    macro_f1_seeds = []
    first_cm = None

    for seed in SEEDS:
        d = load_preds(seed)
        yte, ytr, pred = d["yte"], d["ytr"], d[f"pred_{name}"]
        present = np.unique(np.concatenate([ytr, yte]))
        names_present = [CLASSES[i] for i in present]
        rep = classification_report(yte, pred, labels=present, target_names=names_present,
                                    digits=4, zero_division=0, output_dict=True)
        trc = collections.Counter(int(v) for v in ytr)
        tec = collections.Counter(int(v) for v in yte)
        for i in present:
            cls = CLASSES[i]
            tr, te = trc.get(int(i), 0), tec.get(int(i), 0)
            if te > 0:                     # only score classes that were actually evaluated
                prec[cls].append(rep[cls]["precision"])
                rec[cls].append(rep[cls]["recall"])
                f1[cls].append(rep[cls]["f1-score"])
                tr_sup[cls].append(tr)
                te_sup[cls].append(te)
                if tr == 0:
                    zero_train[cls] += 1   # evaluated but unlearnable -> real recall 0
            else:
                zero_test[cls] += 1        # not evaluated this seed -> don't fake a 0
        # NOT rep["macro avg"]: that averages over `present` = train-or-test
        # classes, which is not Eq. (2). See src/macro_f1.py.
        mf1, n_cplus = macro_f1_cplus(yte, pred)
        macro_f1_seeds.append(mf1)
        fpr_s, fnr_s = fpr_fnr(yte, pred)
        seed_rows.append({"model": name, "seed": seed,
                          "accuracy": float(accuracy_score(yte, pred)),
                          "macro_f1": mf1, "n_cplus": n_cplus,
                          "weighted_f1": weighted_f1_cplus(yte, pred),
                          "fpr": fpr_s, "fnr": fnr_s})
        if seed == SEEDS[0]:
            first_cm = (present, confusion_matrix(yte, pred, labels=present))

    all_classes = sorted(set(list(prec) + list(zero_test) + list(zero_train)))
    recs = []
    for cls in all_classes:
        has = len(rec[cls]) > 0
        recs.append({
            "class": cls,
            "precision_mean": np.mean(prec[cls]) if has else np.nan,
            "precision_std": np.std(prec[cls]) if has else np.nan,
            "recall_mean": np.mean(rec[cls]) if has else np.nan,
            "recall_std": np.std(rec[cls]) if has else np.nan,
            "f1_mean": np.mean(f1[cls]) if has else np.nan,
            "f1_std": np.std(f1[cls]) if has else np.nan,
            "train_support_mean": np.mean(tr_sup[cls]) if has else 0.0,
            "test_support_mean": np.mean(te_sup[cls]) if has else 0.0,
            "status": status_str(zero_train.get(cls, 0), zero_test.get(cls, 0), len(SEEDS)),
        })
    tbl = pd.DataFrame(recs).set_index("class").round(4)
    tbl.to_csv(os.path.join(RESDIR, f"e4_{DATASET}_group_perclass_{name}.csv"))
    print(f"[{name}] macro-F1 {np.mean(macro_f1_seeds):.4f} ± {np.std(macro_f1_seeds):.4f}"
          f"  ({sum(1 for c in all_classes if zero_train.get(c,0)) } class(es) zero-train in >=1 seed)")

    # confusion figure (first seed, for a visual)
    present, cm = first_cm
    names_present = [CLASSES[i] for i in present]
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names_present))); ax.set_xticklabels(names_present, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names_present))); ax.set_yticklabels(names_present, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} - E4 group-by-5-tuple (seed {SEEDS[0]})")
    for i in range(len(names_present)):
        for j in range(len(names_present)):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"confusion_e4_{DATASET}_{name}.png"), dpi=130)
    plt.close(fig)

# ---- summary (recomputed from the saved predictions) + Δ vs E2 random split --
if seed_rows:
    r = pd.DataFrame(seed_rows)
    # Write the per-seed table back out too, so results CSV and summary can never
    # disagree about what the run produced.
    r.round(6).to_csv(os.path.join(RESDIR, f"e4_{DATASET}_group_results_recomputed.csv"),
                      index=False)
    if os.path.exists(RESULTS_CSV):
        old = pd.read_csv(RESULTS_CSV)
        if "macro_f1" in old.columns:
            chk = old.merge(r, on=["model", "seed"], suffixes=("_run", "_eval"))
            drift = chk[(chk["macro_f1_run"] - chk["macro_f1_eval"]).abs() > 1e-9]
            for _, d in drift.iterrows():
                print(f"  NOTE {d['model']} seed {int(d['seed'])}: results CSV says "
                      f"macro_f1={d['macro_f1_run']:.4f}, predictions give "
                      f"{d['macro_f1_eval']:.4f} (|C+|={int(d['n_cplus'])}). "
                      f"The run predates the Eq. (2) fix; using the prediction value.")
    pstd = lambda s: s.std(ddof=0)   # population std, consistent with the per-class np.std above
    agg = r.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", pstd),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", pstd),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_std=("weighted_f1", pstd),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", pstd),
        fnr_mean=("fnr", "mean"), fnr_std=("fnr", pstd),
        n_cplus_min=("n_cplus", "min"), n_cplus_max=("n_cplus", "max"),
        n_seeds=("seed", "nunique")).reset_index()
    if os.path.exists(E2_RESULTS):
        e2 = pd.read_csv(E2_RESULTS)
        mcol = "model" if "model" in e2.columns else None
        col = next((c for c in ("macro_f1", "macroF1", "macro-f1") if c in e2.columns), None)
        if mcol and col:
            e2m = e2[[mcol, col]].rename(columns={mcol: "model", col: "e2_macro_f1"})
            agg = agg.merge(e2m, on="model", how="left")
            agg["delta_macro_f1_vs_e2"] = (agg["macro_f1_mean"] - agg["e2_macro_f1"]).round(4)
    else:
        print(f"(E2 baseline {os.path.basename(E2_RESULTS)} not found; skipping Δ column)")
    agg = agg.round(4)
    agg.to_csv(os.path.join(RESDIR, f"e4_{DATASET}_group_summary.csv"), index=False)
    print("Summary:\n", agg.to_string(index=False))
else:
    print(f"(no {os.path.basename(RESULTS_CSV)}; run run_e4_group.py first)")
print("Done.")
