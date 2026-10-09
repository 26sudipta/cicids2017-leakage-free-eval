# Rerun after the macro-F1 fix (26 Jul 2026)

## What was wrong

`f1_score(y_true, y_pred, average="macro")` with no `labels=` averages over the
**union of labels in y_true and y_pred**, not over $\mathcal{C}^{+}$ as the
paper's Eq. (2) defines. On a degenerate split those differ, and the union is
model-dependent, so five models got five different denominators.

Calendar-split denominators were 10, 14, 14, 13 and 14 where $|\mathcal{C}^{+}|$
was 8 throughout.

## What was changed

New `src/macro_f1.py` holds the canonical definition. Six files now import it:

| File | Was | Effect of the fix |
|---|---|---|
| `run_e3_temporal.py` | bare `average="macro"` | **numbers change** |
| `run_e4_group.py` | bare `average="macro"` | **one cell changes** (DT seed 2) |
| `run_e1_baseline.py` | bare `average="macro"` | no change, pinned for consistency |
| `run_e2_corrected.py` | bare `average="macro"` | no change, pinned for consistency |
| `eval_metrics_e3.py` | `rep["macro avg"]` over train-or-test classes | printed value changes |
| `eval_metrics_e4.py` | `rep["macro avg"]` over train-or-test classes | printed value changes |

Two extras:

- Every run now writes an `n_cplus` column and logs `(|C+|=n)` beside macro-F1.
  A macro-F1 on a degenerate split is not interpretable without its denominator.
- `run_e3_temporal.py` and `run_e4_group.py` call `audit_macro()` and print a
  `NOTE` line whenever the sklearn default would have given a different answer.
  If the fix is ever reverted by a refactor, the logs say so.

The per-class CSVs were always correct (`classification_report` was called with
an explicit `labels=`), so **Fig. 2, Fig. 3 and the whole per-class
decomposition are unaffected**. Only the aggregate row changes.

## Expected values after the rerun

These are already in the paper's Table I. If a rerun disagrees with this table,
something else is wrong. Stop and investigate rather than editing the paper.

| Model | E1 | E2 | E3 cal | E3 time | E4 | Δ |
|---|---|---|---|---|---|---|
| RandomForest | 0.847 | 0.939 | 0.111 | 0.427 | 0.840 ± 0.012 | −0.099 |
| DecisionTree | 0.865 | 0.896 | 0.115 | 0.443 | 0.821 ± 0.025 | −0.075 |
| LogisticRegression | 0.658 | 0.794 | 0.115 | 0.208 | 0.722 ± 0.013 | −0.072 |
| KNN | 0.671 | 0.697 | 0.119 | 0.437 | 0.708 ± 0.026 | +0.010 |
| LinearSVM_SGD | 0.365 | 0.510 | 0.113 | 0.199 | 0.498 ± 0.005 | −0.012 |

**Round from full precision, not from a CSV.** The summary CSV is written with
`.round(4)`. Rounding *that* again to 3 decimals double-rounds. It already cost
us one cell: LogisticRegression E4 std is 0.013490, which is 0.013, but via the
stored 0.0135 it became 0.014 in an earlier draft of Table I.

$|\mathcal{C}^{+}|$: 15 for E1, E2 and E4 (14 for E4 at seed 2), 8 for the
calendar split, 4 for the time-sorted split.

---

## Commands

Run from the repo root, `cicids2017-leakage-free-eval/`. `DATA_DIR` is
auto-resolved because `data/original/` and `data/corrected_wtmc2021/` are both
populated, so it is omitted below.

### Fastest path: verify without retraining (about 1 minute)

E4 saved its per-seed predictions, so the corrected E4 numbers can be
confirmed with no model fitting at all. Do this first.

```bash
cd cicids2017-leakage-free-eval
venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
import numpy as np
from macro_f1 import macro_f1_cplus
want = {"RandomForest":0.840, "DecisionTree":0.821, "LogisticRegression":0.722,
        "KNN":0.708, "LinearSVM_SGD":0.498}
for m, exp in want.items():
    v = [macro_f1_cplus(np.load(f"results/e4_corrected_group_seed{s}_full_preds.npz",
                                allow_pickle=True)["yte"],
                        np.load(f"results/e4_corrected_group_seed{s}_full_preds.npz",
                                allow_pickle=True)[f"pred_{m}"])[0] for s in (2,42,123)]
    got = round(float(np.mean(v)), 3)
    print(f"{'OK ' if got == exp else 'FAIL'} {m:<20} {got:.3f}  expected {exp:.3f}")
PY
```

### Full rerun of the affected experiments

Only E3 actually needs retraining. E1, E2 and E4 are numerically unchanged by
the fix (E4 apart from the one DT seed-2 cell, which the script above
confirms), so rerun them only if you want the `n_cplus` column present in every
results CSV.

**E3, the four temporal conditions.** These are the runs whose numbers changed.

```bash
cd cicids2017-leakage-free-eval

DATASET=corrected SPLIT=calendar venv/bin/python src/run_e3_temporal.py
DATASET=corrected SPLIT=timesort venv/bin/python src/run_e3_temporal.py
DATASET=original  SPLIT=calendar venv/bin/python src/run_e3_temporal.py
DATASET=original  SPLIT=timesort venv/bin/python src/run_e3_temporal.py

TAG=corrected_calendar venv/bin/python src/eval_metrics_e3.py
TAG=corrected_timesort venv/bin/python src/eval_metrics_e3.py
TAG=original_calendar  venv/bin/python src/eval_metrics_e3.py
TAG=original_timesort  venv/bin/python src/eval_metrics_e3.py
```

Roughly 20 to 45 minutes per condition on the full data, KNN dominating.

**E4, optional, only for the `n_cplus` column.**

```bash
DATASET=corrected venv/bin/python src/run_e4_group.py
DATASET=corrected venv/bin/python src/eval_metrics_e4.py
```

Three seeds by five models on 1.8M flows. Budget a couple of hours.

**E1 and E2, optional, same reason.**

```bash
venv/bin/python src/run_e1_baseline.py
venv/bin/python src/run_e2_corrected.py
venv/bin/python src/eval_metrics.py
venv/bin/python src/eval_metrics_e2.py
```

### Smoke test first if you want to check plumbing cheaply

```bash
MAX_ROWS=200000 DATASET=corrected SPLIT=calendar venv/bin/python src/run_e3_temporal.py
```

Numbers from a capped run will not match the table. This only proves the code
runs and the `(|C+|=n)` logging appears.

---

## What to check in the output

1. Every macro-F1 log line now ends with `(|C+|=n)`. Calendar should print
   `|C+|=8`, time-sorted `|C+|=4`, E1/E2 `|C+|=15`.
2. `NOTE` lines appear on the temporal runs, saying the sklearn default would
   have differed. That is the guard working, not a failure.
3. `results/e3_corrected_calendar_results.csv` should show macro_f1 near 0.111
   for RandomForest, not 0.089. If it still says 0.089 the import did not take
   effect, so check that you ran `src/run_e3_temporal.py` as a path (which puts
   `src/` on `sys.path`) rather than importing it from elsewhere.
4. Compare against the expected-values table above before touching the paper.
