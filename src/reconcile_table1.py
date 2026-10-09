"""
Reconcile the Sec. IV-C per-class decomposition against Table I, and recompute
every E2 -> E4 Delta on a class set common to all three seeds.

Written 27 Jul 2026 for P0 items 1 and 2 of IMPLEMENTATION_PLAN_26Jul2026.md.

WHY THIS EXISTS
---------------
Table I reports E4 as the mean of three per-seed macro-F1 values, each computed
over that seed's own C+ (Eq. 2).  |C+| is 14 at seed 2 and 15 at seeds 42 and
123, because the group split leaves Infiltration with zero test flows at seed
2.  Delta = E4 - E2 therefore subtracts two averages taken over DIFFERENT class
sets.  A mean of ratios is not the ratio of means, so:

  * the Sec. IV-C decomposition (per-class three-seed mean F1, summed over 15
    classes) gave -0.105 for the random forest against Table I's -0.099;
  * k-NN's Delta came out POSITIVE (+0.010), the only positive cell in the
    table, purely because seed 2 skipped a class k-NN scores 0.000 on.

On the 14 classes tested at every seed the two definitions are identical by
linearity, and the k-NN sign flips to -0.007.

NO RETRAINING.  Everything here is computed from the saved per-seed prediction
files, so it can be re-run anywhere in seconds.  F1 is implemented directly
from confusion counts (zero_division=0) instead of via sklearn, so the script
runs without the project venv; correctness is enforced by asserting that every
recomputed per-seed macro-F1 matches e4_corrected_group_results.csv to 1e-9.

Usage:  python3 src/reconcile_table1.py        (from the repo root)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
SEEDS = [2, 42, 123]
MODELS = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN",
          "LinearSVM_SGD"]


def per_class_f1(y_true, y_pred, n_classes):
    """F1 per class index, zero_division=0.  NaN where the class has no test flows."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        n_true = int((y_true == c).sum())
        if n_true == 0:
            continue                      # not in C+, F1 undefined
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = n_true - tp
        denom = 2 * tp + fp + fn
        out[c] = 0.0 if denom == 0 else 2 * tp / denom
    return out


def main():
    classes, f1_e4, test_sup = None, {}, {}
    for seed in SEEDS:
        d = np.load(RES / f"e4_corrected_group_seed{seed}_full_preds.npz",
                    allow_pickle=True)
        if classes is None:
            classes = [str(c) for c in d["classes"]]
        yte = d["yte"]
        test_sup[seed] = np.array([(yte == c).sum() for c in range(len(classes))])
        for m in MODELS:
            f1_e4[(m, seed)] = per_class_f1(yte, d[f"pred_{m}"], len(classes))

    # --- guard: refuse to report anything unless we reproduce the published run
    pub = pd.read_csv(RES / "e4_corrected_group_results.csv")
    worst = 0.0
    for _, r in pub.iterrows():
        v = f1_e4[(r["model"], int(r["seed"]))]
        assert int(np.isfinite(v).sum()) == r["n_cplus"]
        worst = max(worst, abs(np.nanmean(v) - r["macro_f1"]))
    assert worst < 1e-9, f"recomputed F1 disagrees with published run by {worst:.2e}"
    print(f"GUARD PASS  all 15 per-seed macro-F1 values reproduced "
          f"(worst |diff| {worst:.2e})\n")

    f1_e2 = {m: np.array([pd.read_csv(RES / f"e2_perclass_{m}.csv", index_col=0)
                          .loc[c, "f1-score"] for c in classes], dtype=float)
             for m in MODELS}

    common = np.where([all(np.isfinite(f1_e4[(MODELS[0], s)][c]) for s in SEEDS)
                       for c in range(len(classes))])[0]
    absent = [classes[c] for c in range(len(classes)) if c not in common]
    print(f"Tested at all three seeds: {len(common)}/{len(classes)}.  "
          f"Not tested at every seed: {absent}")
    for a in absent:
        i = classes.index(a)
        print(f"   {a} test support by seed: "
              f"{ {s: int(test_sup[s][i]) for s in SEEDS} }")

    print(f"\n{'model':<20} {'E2':>7} {'E4 pub':>8} {'D pub':>8} | "
          f"{'E2(14)':>7} {'E4(14)':>7} {'std':>6} {'D(14)':>8} {'decomp/14':>10}")
    rows = []
    for m in MODELS:
        e2 = np.nanmean(f1_e2[m])
        e4 = np.mean([np.nanmean(f1_e4[(m, s)]) for s in SEEDS])
        e2c = np.mean(f1_e2[m][common])
        per_seed = [np.mean(f1_e4[(m, s)][common]) for s in SEEDS]
        e4c = np.mean(per_seed)
        mean_e4_class = np.array([np.nanmean([f1_e4[(m, s)][c] for s in SEEDS])
                                  for c in range(len(classes))])
        decomp = np.sum((mean_e4_class - f1_e2[m])[common]) / len(common)
        rows.append(dict(model=m, e2=e2, e4_published=e4, delta_published=e4 - e2,
                         e2_common=e2c, e4_common=e4c,
                         e4_common_std=np.std(per_seed), delta_common=e4c - e2c,
                         decomposition_common=decomp,
                         decomposition_all15=np.nansum(mean_e4_class - f1_e2[m])
                         / len(classes)))
        print(f"{m:<20} {e2:7.4f} {e4:8.4f} {e4-e2:+8.4f} | {e2c:7.4f} {e4c:7.4f} "
              f"{np.std(per_seed):6.4f} {e4c-e2c:+8.4f} {decomp:+10.4f}")
    print("\nD(14) and decomp/14 agree by construction: on a constant class set "
          "the mean of\nper-class differences IS the difference of macro means. "
          "They diverge only when\n|C+| varies, which is the whole point.")

    out = RES / "table1_common_denominator.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    # ---- Sec. IV-D: the Moczkodan 13-class yardstick -----------------------
    # Added 30 Jul 2026.  Sec. IV-D says "scored on their thirteen our random
    # forest drops 0.036 per (1)".  That number was computed by hand and had no
    # script, which made the one externally-facing comparison in the paper the
    # only number a reader could not reproduce.  It is also CONVENTION-DEPENDENT,
    # which matters because this script exists to police exactly that:
    #
    #   Eq. (1) per seed over the classes that seed tested   -0.0359  <- printed
    #   constant 12-class denominator (drop Infiltration)    -0.0223
    #   mean of per-class three-seed differences             -0.0445
    #
    # The paper quotes the Eq. (1) value, so the paper's own definition is
    # applied consistently.  The spread is reported here so nobody has to guess
    # which one was meant, and so a reviewer asking "which denominator?" gets
    # the same three numbers we did.
    MOCZ_DROPPED = ["Heartbleed", "DoS Slowhttptest"]   # their <2-window rule
    keep = [i for i, c in enumerate(classes) if c not in MOCZ_DROPPED]
    print(f"\nMoczkodan 13-class yardstick (dropping {MOCZ_DROPPED}):")
    yrows = []
    for m in MODELS:
        e2_13 = float(np.mean(f1_e2[m][keep]))
        per_seed = [float(np.nanmean(f1_e4[(m, s)][keep])) for s in SEEDS]
        eq1 = float(np.mean(per_seed)) - e2_13
        keep12 = [i for i in keep if classes[i] != "Infiltration"]
        const = (float(np.mean([np.mean(f1_e4[(m, s)][keep12]) for s in SEEDS]))
                 - float(np.mean(f1_e2[m][keep12])))
        mean_cls = np.array([np.nanmean([f1_e4[(m, s)][c] for s in SEEDS])
                             for c in range(len(classes))])
        percls = float(np.nanmean((mean_cls - f1_e2[m])[keep]))
        yrows.append(dict(model=m, e2_13=e2_13, e4_13=float(np.mean(per_seed)),
                          delta_eq1=eq1, delta_const12=const,
                          delta_perclass_mean=percls))
        print(f"  {m:<20} eq(1) {eq1:+.4f}   const-12 {const:+.4f}   "
              f"per-class mean {percls:+.4f}")
    yout = RES / "moczkodan13_yardstick.csv"
    pd.DataFrame(yrows).to_csv(yout, index=False)
    rf = next(r for r in yrows if r["model"] == "RandomForest")
    assert abs(rf["delta_eq1"] - (-0.036)) < 0.0005, (
        f"Sec. IV-D says the RF drops 0.036 on their thirteen; this run gives "
        f"{rf['delta_eq1']:.4f}. Fix the paper or the script, not both.")
    print(f"\nGUARD PASS  Sec. IV-D's 0.036 reproduced ({rf['delta_eq1']:+.4f})")
    print(f"wrote {yout}")


if __name__ == "__main__":
    main()
