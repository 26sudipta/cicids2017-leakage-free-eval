"""
Task 3.9 (added 28 Jul 2026): bootstrap confidence intervals for the E4
leakage-free group split.

WHY THIS AND NOT THE TESTS THE REVIEW ASKED FOR
-----------------------------------------------
A reviewer suggested (a) McNemar E2 vs E4 and (b) a paired Wilcoxon over the
three seeds. Neither is applicable here:

  * McNemar needs paired predictions on the SAME instances. E2's test set is a
    stratified random 30% of the corrected data; E4's is a group-based
    partition that differs per seed (551,198 / 555,196 / 556,550 rows). There
    are no shared instances, so no 2x2 discordance table exists.
  * Wilcoxon signed-rank at n=3 has a minimum two-sided exact p of 2/2^3 =
    0.25, so it cannot reach p<0.05 by construction. (n=5 across models gives
    0.0625; you need n>=6.) run_e2_corrected.py also has no seed loop, so
    there are no per-seed E2 values to pair against.

What IS valid, and what this script does: a CLUSTER bootstrap over the
conversation groups of the E4 test partition. Resampling flows independently
would be wrong in this paper specifically, because conversation-level
dependence is the phenomenon under study; an i.i.d. row bootstrap would treat
correlated flows as independent evidence and report intervals that are too
narrow. We resample whole five-tuple groups with replacement instead.

METHOD
------
TP/FP/FN are additive over rows, so they are additive over groups. We
precompute per-group, per-class (TP, FP, FN), then each bootstrap replicate is
a multiplicity vector over groups times those matrices. No index materialising,
no per-replicate re-scoring.

TWO MACRO-F1 COLUMNS, AND WHY
-----------------------------
Applying Eq. (2) literally inside the bootstrap (average over C+ = classes with
n_c^test > 0 IN THAT REPLICATE) biases the result upward: a class held in a
single conversation drops out of many replicates, and the classes that drop out
are exactly the ones scoring 0. At seed 42 the random forest reads 0.869 that
way against a point estimate of 0.840. That number is reported as
`macro_f1_cplus` for transparency, but it is not the one to quote.

`macro_f1_well` averages over a FIXED set: the eleven well-supported classes,
defined as all classes except the four micro-classes the paper already isolates
(Heartbleed, Infiltration, Web Attack Sql Injection, Web Attack XSS). No
dropout, no moving denominator, and it is the set C1 actually makes a claim
about.

The four micro-classes are reported per-class only, and their intervals are
degenerate by construction (a one-conversation class is either wholly in a
replicate or wholly out). That degeneracy is itself evidence for C3: those
classes cannot support inference at any sample size the split can give them.

GUARD: the reconstructed split is verified element-wise against the saved
predictions (yte) before anything is computed. If the reconstruction does not
reproduce the exact test partition the script aborts rather than reporting
numbers from a different split.

Inputs:  results/cleaned_e4_corrected_full.pkl        (cached clean frame)
         results/e4_corrected_group_seed{S}_full_preds.npz
Output:  results/e4_bootstrap_ci.csv
         results/e4_bootstrap_ci_perclass.csv

Run:  python src/bootstrap_ci_e4.py
Env:  B=<replicates, default 1000>   SEEDS=2,42,123   ALPHA=0.05
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
PICKLE = os.path.join(RESDIR, "cleaned_e4_corrected_full.pkl")

SEEDS = [int(s) for s in os.environ.get("SEEDS", "2,42,123").split(",") if s.strip()]
B = int(os.environ.get("B", "1000"))
ALPHA = float(os.environ.get("ALPHA", "0.05"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.30"))
GROUP_COL, LABEL_KEY = "__group__", "__label__"
MODELS = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN", "LinearSVM_SGD"]

# The four micro-classes the paper isolates in Sec. IV-C. Everything else is the
# eleven well-supported classes that C1 makes its claim about.
MICRO = {"Heartbleed", "Infiltration", "Web Attack - Sql Injection", "Web Attack - XSS"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_group_counts(y_true, y_pred, grp_idx, n_groups, n_classes):
    """(TP, FP, FN) summed within each group, shape (n_groups, n_classes) each."""
    tp = np.zeros((n_groups, n_classes), dtype=np.int32)
    fp = np.zeros((n_groups, n_classes), dtype=np.int32)
    fn = np.zeros((n_groups, n_classes), dtype=np.int32)
    hit = y_true == y_pred
    # true positives: correct rows credited to their class
    np.add.at(tp, (grp_idx[hit], y_true[hit]), 1)
    # false positives: wrong rows credited to the PREDICTED class
    np.add.at(fp, (grp_idx[~hit], y_pred[~hit]), 1)
    # false negatives: wrong rows charged to the TRUE class
    np.add.at(fn, (grp_idx[~hit], y_true[~hit]), 1)
    return tp, fp, fn


def f1_from_counts(TP, FP, FN):
    """Row-wise F1 over count matrices of shape (B, n_classes). 0/0 -> 0."""
    denom = 2.0 * TP + FP + FN
    out = np.zeros_like(denom, dtype=np.float64)
    nz = denom > 0
    out[nz] = (2.0 * TP[nz]) / denom[nz]
    return out


def main():
    if not os.path.exists(PICKLE):
        raise FileNotFoundError(
            f"{PICKLE} missing. It is the cached clean frame written by run_e4_group.py; "
            f"re-run E4 (or set OUT_DIR) before running this script.")

    log(f"loading {os.path.basename(PICKLE)} (this is the slow part)")
    df = pd.read_pickle(PICKLE)
    y_raw = df[LABEL_KEY].values
    groups_all = df[GROUP_COL].values
    n_rows = len(df)
    del df  # features are not needed; only labels and group keys

    le = LabelEncoder().fit(y_raw)
    y_enc = le.transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)
    missing = MICRO - set(classes)
    if missing:
        raise AssertionError(f"micro-class names not found in the label set: {sorted(missing)}")
    well_idx = np.array([i for i, c in enumerate(classes) if c not in MICRO])
    log(f"{n_rows:,} rows, {n_classes} classes, {pd.unique(groups_all).size:,} five-tuple groups")

    rows, prows = [], []
    for seed in SEEDS:
        npz_path = os.path.join(RESDIR, f"e4_corrected_group_seed{seed}_full_preds.npz")
        if not os.path.exists(npz_path):
            log(f"!! seed {seed}: {os.path.basename(npz_path)} missing, skipping")
            continue
        d = np.load(npz_path, allow_pickle=True)

        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        _, te = next(gss.split(np.zeros((n_rows, 1), dtype=np.int8), y_enc, groups=groups_all))

        # --- GUARD: reconstruction must reproduce the exact saved partition ---
        yte_saved = d["yte"]
        if len(te) != len(yte_saved) or not np.array_equal(y_enc[te], yte_saved):
            raise AssertionError(
                f"seed {seed}: reconstructed test partition does not match the saved "
                f"predictions (n={len(te)} vs {len(yte_saved)}). Refusing to bootstrap "
                f"against a different split.")
        if not np.array_equal(np.asarray(d["classes"], dtype=object), classes.astype(object)):
            raise AssertionError(f"seed {seed}: class order differs from the saved encoder")
        log(f"seed {seed}: split reconstructed and verified against saved preds "
            f"({len(te):,} test rows)")

        yte = y_enc[te]
        g_te = groups_all[te]
        _, grp_idx = np.unique(g_te, return_inverse=True)
        n_groups = grp_idx.max() + 1
        support = np.bincount(yte, minlength=n_classes)
        log(f"seed {seed}: {n_groups:,} test conversations")

        rng = np.random.default_rng(seed)
        # one shared set of replicate multiplicities per seed, so model CIs are comparable
        mult = np.zeros((B, n_groups), dtype=np.float32)
        for b in range(B):
            draw = rng.integers(0, n_groups, size=n_groups)
            mult[b] = np.bincount(draw, minlength=n_groups)

        for model in MODELS:
            key = f"pred_{model}"
            if key not in d.files:
                log(f"  seed {seed}: {key} absent, skipping")
                continue
            ypred = d[key].astype(np.int64)
            tp, fp, fn = per_group_counts(yte, ypred, grp_idx, n_groups, n_classes)

            TP = mult @ tp.astype(np.float32)
            FP = mult @ fp.astype(np.float32)
            FN = mult @ fn.astype(np.float32)
            f1 = f1_from_counts(TP, FP, FN)                    # (B, n_classes)
            present = (TP + FN) > 0                            # n_c^test > 0 in replicate

            # (a) Eq. (2) applied literally: moving denominator, biased high.
            macro_c = np.where(present, f1, 0.0).sum(1) / np.maximum(present.sum(1), 1)
            # (b) fixed set of eleven well-supported classes: the quotable one.
            macro_w = f1[:, well_idx].mean(1)

            q = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
            wlo, whi = np.percentile(macro_w, q)
            clo, chi = np.percentile(macro_c, q)
            rows.append(dict(seed=seed, model=model,
                             macro_f1_well=float(macro_w.mean()),
                             well_ci_lo=float(wlo), well_ci_hi=float(whi),
                             well_ci_width=float(whi - wlo),
                             macro_f1_cplus=float(macro_c.mean()),
                             cplus_ci_lo=float(clo), cplus_ci_hi=float(chi),
                             n_well=len(well_idx), n_groups=int(n_groups), B=B))
            log(f"  seed {seed} {model:20s} macro-F1(11 well-supported) "
                f"{macro_w.mean():.4f} [{wlo:.4f}, {whi:.4f}] width {whi - wlo:.4f}")

            for c in range(n_classes):
                col = f1[:, c]
                plo, phi = np.percentile(col, q)
                prows.append(dict(seed=seed, model=model, cls=classes[c],
                                  support=int(support[c]),
                                  micro=bool(classes[c] in MICRO),
                                  absent_frac=float(1.0 - present[:, c].mean()),
                                  f1_mean=float(col.mean()),
                                  ci_lo=float(plo), ci_hi=float(phi),
                                  ci_width=float(phi - plo)))
            del tp, fp, fn, TP, FP, FN, f1
        del mult

    out = os.path.join(RESDIR, "e4_bootstrap_ci.csv")
    pout = os.path.join(RESDIR, "e4_bootstrap_ci_perclass.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(prows).to_csv(pout, index=False)
    log(f"wrote {out}")
    log(f"wrote {pout}")


if __name__ == "__main__":
    main()
