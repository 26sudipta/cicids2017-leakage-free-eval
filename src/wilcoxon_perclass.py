"""
Per-class paired Wilcoxon signed-rank test, E2 against E4.

WHY THIS PAIRING AND NOT THE OTHER TWO
--------------------------------------
Two earlier Wilcoxon pairings were tried and rejected as underpowered by design
(see the header of bootstrap_ci_e4.py and the 30 Jul response document):

  - pair across the three E4 seeds: n = 3, exact two-sided p floor 2/2^3 = 0.25.
  - pair across the five models:    n = 5, floor 2/2^5 = 0.0625.

Neither can reach 0.05 whatever the data says. Pairing by CLASS is the third
option and is the only one with usable n: 15 classes, so up to 15 pairs.

WHAT THIS SCRIPT TESTS, AND WHAT IT DOES NOT
--------------------------------------------
A single test over all fifteen classes does NOT test the paper's claim (C1). It
tests "did per-class F1 shift systematically", and a significant answer there,
read alone, contradicts the headline. C1 is a claim about WHICH classes moved.
So the test is run stratified, on three class sets:

  all15   every class the split scores
  well11  the eleven well-supported classes
  micro4  Heartbleed, Infiltration, SQL injection, XSS

The result that speaks to C1 is the CONTRAST between all15 and well11, not
either p-value on its own. micro4 is reported for completeness only: at n = 4
the exact two-sided floor is 0.125, so it cannot reach significance even when
every class moves the same way, which for the tree models it does.

EXACT p RATHER THAN THE NORMAL APPROXIMATION
--------------------------------------------
scipy's default switches to a normal approximation with a tie correction once
ties are present, and these differences are full of ties (many are exactly
0.0000). Instead this computes the exact conditional distribution: rank the
absolute differences with average ranks, then enumerate all 2^n sign
assignments of that fixed rank vector. Under ties this permutation argument is
the correct exact test, not an approximation. n <= 15 so 2^n is at most 32768.
No scipy dependency, which also means this runs in environments where the
install is unavailable.

ZERO HANDLING, AND WHY IT MATTERS HERE
--------------------------------------
Standard Wilcoxon discards pairs with a zero difference. That is awkward for
this paper, because a class whose F1 did not move at all is exactly the
evidence C1 rests on, and discarding it shrinks n. For the random forest five
of the eleven well-supported classes have a difference of exactly 0.0000, so n
falls from 11 to 6, where the exact floor is 0.031, only just under 0.05. A
non-significant result there is weak evidence and the paper says so; the
cluster bootstrap remains the stronger evidence for the "holds steady" half.
Pratt's method, which ranks the zeros before dropping them, is computed as a
robustness check. Both are reported so nobody has to take the choice on trust.

KNOWN LIMITATIONS, STATED BECAUSE THEY AFFECT THE READING
---------------------------------------------------------
1. E2 is a single run. run_e2_corrected.py has no seed loop, so each pair is one
   E2 value against a three-seed E4 mean. The pairing is not symmetric in
   sampling effort.
2. The pairs are not equally informative. BENIGN's F1 rests on 445,110 test
   flows and SQL injection's on 4, yet the test weights their signs by rank
   alone.
3. Ten tests are run (five models, two main class sets). No single p-value
   should be leaned on; the pattern across models is the evidence. A Bonferroni
   correction at 0.05/5 would remove the random forest's all15 result and the
   linear SVM's well11 result alike, so uncorrected values are reported with
   the test count stated.

Input:  results/e4_vs_e2_perclass_comparison.csv  (e4_f1 is the three-seed mean)
Output: results/wilcoxon_perclass.csv
Run:    python src/wilcoxon_perclass.py
"""
import itertools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, "..", "results")
SRC = os.path.join(RESDIR, "e4_vs_e2_perclass_comparison.csv")
OUT = os.path.join(RESDIR, "wilcoxon_perclass.csv")

# The four classes Sec. IV-C identifies as micro-classes: at most 26 test flows
# at any seed. Names match the corrected release's label strings exactly.
MICRO = [
    "Heartbleed",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]

MODELS = ["RandomForest", "DecisionTree", "LogisticRegression", "KNN", "LinearSVM_SGD"]


def average_ranks(x):
    """Ranks of x, 1-based, tied values sharing their mean rank."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def exact_two_sided_p(ranks, diff):
    """
    Exact conditional p for the signed-rank statistic, by enumerating every
    sign assignment of a fixed rank vector. Returns (T, p).

    T is min(W+, W-). The p-value is the share of the 2^n equally likely sign
    assignments under the null whose statistic is at least as extreme as T.
    """
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    t_obs = min(w_pos, w_neg)
    n = len(ranks)
    hits = 0
    for signs in itertools.product([False, True], repeat=n):
        s = np.array(signs)
        if min(ranks[s].sum(), ranks[~s].sum()) <= t_obs + 1e-9:
            hits += 1
    return t_obs, hits / 2 ** n


def signed_rank(diff, zero_method):
    """
    zero_method='wilcox': drop zeros, then rank. The standard procedure.
    zero_method='pratt' : rank including zeros, then drop them from the
                          statistic. Keeps zeros' effect on the rank scale.
    """
    d = np.asarray(diff, dtype=float)
    if zero_method == "wilcox":
        kept = d[d != 0]
        if len(kept) == 0:
            return None
        return len(kept), exact_two_sided_p(average_ranks(np.abs(kept)), kept)
    if zero_method == "pratt":
        ranks = average_ranks(np.abs(d))
        keep = d != 0
        if keep.sum() == 0:
            return None
        return int(keep.sum()), exact_two_sided_p(ranks[keep], d[keep])
    raise ValueError(zero_method)


def main():
    df = pd.read_csv(SRC)

    rows = []
    for model in MODELS:
        m = df[df.model == model]
        sets = [
            ("all15", m),
            ("well11", m[~m["class"].isin(MICRO)]),
            ("micro4", m[m["class"].isin(MICRO)]),
        ]
        for label, sub in sets:
            # Sign convention: negative means E4 scored below E2.
            diff = (sub.e4_f1 - sub.e2_f1).values
            row = {
                "model": model,
                "class_set": label,
                "n_pairs": len(diff),
                "n_negative": int((diff < 0).sum()),
                "n_positive": int((diff > 0).sum()),
                "n_zero": int((diff == 0).sum()),
                "median_delta": round(float(np.median(diff)), 4),
                "mean_delta": round(float(diff.mean()), 4),
                "min_delta": round(float(diff.min()), 4),
            }
            for method in ("wilcox", "pratt"):
                res = signed_rank(diff, method)
                if res is None:
                    row[f"n_{method}"] = 0
                    row[f"T_{method}"] = np.nan
                    row[f"p_{method}"] = np.nan
                    row[f"p_floor_{method}"] = np.nan
                    continue
                n_eff, (t_obs, p) = res
                row[f"n_{method}"] = n_eff
                row[f"T_{method}"] = round(float(t_obs), 1)
                row[f"p_{method}"] = round(float(p), 4)
                # Smallest two-sided p reachable at this n. Above 0.05 means the
                # test cannot detect anything, whatever the data shows.
                row[f"p_floor_{method}"] = round(2 / 2 ** n_eff, 4)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    cols = [
        "model", "class_set", "n_pairs", "n_negative", "n_positive", "n_zero",
        "median_delta", "n_wilcox", "p_wilcox", "p_floor_wilcox", "p_pratt",
    ]
    print(out[cols].to_string(index=False))
    print(f"\nSaved -> {os.path.normpath(OUT)}")

    # Guards. If any of these trip, a number in the paper is wrong.
    rf11 = out[(out.model == "RandomForest") & (out.class_set == "well11")].iloc[0]
    assert rf11.n_wilcox == 6, f"RF well11 n changed to {rf11.n_wilcox}"
    assert rf11.p_floor_wilcox < 0.05, "RF well11 can no longer detect anything"
    for _, r in out[out.class_set == "micro4"].iterrows():
        if r.n_wilcox:
            assert r.p_floor_wilcox >= 0.05, (
                f"{r.model} micro4 floor dropped below 0.05, revisit the text "
                "that says micro4 cannot reach significance"
            )
    print("Guards passed.")


if __name__ == "__main__":
    main()
