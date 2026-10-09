# RUN_E3 — Temporal (chronological) split experiment

E3 is the **deliberately-degenerate temporal-split sanity check** (roadmap task 2.4). It reuses
the E1/E2 modelling pipeline unchanged and alters **only the train/test split** — from a random
stratified split to a time-based one. We expect macro-F1 to collapse (~0.50 or lower), which
replicates Moczkodan & Ragab (arXiv 2606.11098) and **motivates E4** (the leakage-free
group-by-five-tuple split, task 2.4b). A degenerate E3 is the intended, useful result — do not
"fix" it.

## The four conditions (one script, two env vars)

`src/run_e3_temporal.py` is parameterised by `DATASET` and `SPLIT`:

| Row | DATASET   | SPLIT    | Cleaning (unchanged from) | Split rule |
|-----|-----------|----------|---------------------------|-----------|
| 1   | original  | calendar | E1 (ML CSVs, 78 feats)    | train Mon+Tue+Wed / test Thu+Fri |
| 2   | original  | timesort | E1                        | global time order, first 80% train / last 20% test |
| 3   | corrected | calendar | E2 (WTMC, relabel+drop)   | train Mon+Tue+Wed / test Thu+Fri |
| 4   | corrected | timesort | E2                        | first 80% / last 20% by parsed Timestamp |

**Only the split changes vs. E1/E2.** Cleaning is byte-for-byte the E1 recipe for `original`
and the E2 recipe for `corrected`, so E1→E2→E3 is a clean controlled comparison.

## Data locations (set DATA_DIR)

- original ML CSVs (8 files): the same folder E1 used.
- corrected WTMC CSVs (5 files): the same folder E2 used.

## 0. One-time cleanup (delete stale smoke-test artifacts)

Smoke tests were run in the Cowork sandbox on tiny data slices; they left `*cap40000*` caches
and one meaningless `e3_corrected_calendar_*` result set. Delete them before the real run so
they can't be confused with full-data output:

```bash
cd cicids2017-leakage-free-eval
rm -f results/*e3*cap40000* \
      results/e3_corrected_calendar_results.csv \
      results/e3_corrected_calendar_perclass_*.csv \
      results/e3_corrected_calendar_confusion_*.csv \
      results/figures/confusion_e3_* \
      results/label_classes_e3_*
```

(The `cap40000` caches are already ignored by full runs — full runs use the `full` cache tag —
but removing them keeps `results/` clean.)

## 1. Run all four conditions (full data, on the laptop)

Each run: builds a cleaned-data cache per dataset (reused across both splits), the split+scaled
`.npz`, and a per-model results CSV. The sandbox can't load the full ~1 GB×N CSVs in its 45 s
window, which is why this runs locally (same as E2).

```bash
cd cicids2017-leakage-free-eval

# ---- original ----
DATASET=original  SPLIT=calendar  DATA_DIR=/path/to/original_ml_csvs  python src/run_e3_temporal.py
DATASET=original  SPLIT=timesort  DATA_DIR=/path/to/original_ml_csvs  python src/run_e3_temporal.py

# ---- corrected ----
DATASET=corrected SPLIT=calendar  DATA_DIR=/path/to/corrected_wtmc    python src/run_e3_temporal.py
DATASET=corrected SPLIT=timesort  DATA_DIR=/path/to/corrected_wtmc    python src/run_e3_temporal.py
```

Tips:
- `ONLY_MODELS=RandomForest,DecisionTree` to run a subset.
- `STAGE=prep` to build only the split cache and stop (fast; lets you inspect class support first).
- The cleaned cache `cleaned_e3_<dataset>_full.pkl` is shared by both splits of a dataset, so the
  second split of each dataset skips re-loading the CSVs.

## 2. Per-class metrics + confusion + degeneracy annotation

```bash
TAG=original_calendar  python src/eval_metrics_e3.py
TAG=original_timesort  python src/eval_metrics_e3.py
TAG=corrected_calendar python src/eval_metrics_e3.py
TAG=corrected_timesort python src/eval_metrics_e3.py
```

Each writes `results/e3_<TAG>_perclass_<MODEL>.csv` with two extra columns beyond the usual
precision/recall/F1/support/FPR:
- `train_support` — how many training rows that class had.
- `status` — `ok` / `ZERO_TRAIN (unlearnable)` / `ZERO_TEST (not evaluated)` / `ABSENT`.

`ZERO_TRAIN` means the class exists only on the test side of the time boundary, so the model
could never learn it → recall 0. That column is the paper's evidence; report it, don't hide it
behind macro-F1.

## Actual result (FULL data, 17 Jul 2026)

Full run on the complete datasets (original 2,499,543 flows after dedup; corrected 1,840,339).
RandomForest anchor (macro-F1 = the metric that matters on imbalanced IDS data):

| Condition                    | accuracy | macro-F1 |
|------------------------------|----------|----------|
| original, random 70/30 (E1)  | 0.999    | 0.847    |
| corrected, random 70/30 (E2) | 0.994    | 0.939    |
| original, calendar (E3)      | 0.774    | 0.069    |
| corrected, calendar (E3)     | 0.745    | 0.089    |
| original, timesort (E3)      | 0.560    | 0.145    |
| corrected, timesort (E3)     | 0.495    | 0.285    |

macro-F1 range across all 5 models: calendar 0.064–0.089; timesort 0.069–0.285. The story is
consistent across every model and both datasets:

- **Random → honest split = collapse.** macro-F1 drops from 0.85–0.94 to 0.06–0.29.
- **Calendar split (7 classes zero-train):** Bot, DDoS, Infiltration, PortScan and all three Web
  attacks live only Thu–Fri → never trained → macro-F1 floors near 0.06 (only BENIGN scores).
- **Timesort 80/20 (2 classes zero-train):** only DDoS + PortScan (latest Friday afternoon) are
  stranded past the cut, so more classes survive → macro-F1 0.08–0.29.
- **The collapse is caused by the SPLIT, not the correction:** original-calendar (0.069) ≈
  corrected-calendar (0.089). This answers the open question from novelty check 1.6c.
- **Even accuracy falls** (0.99 → 0.50–0.77) because large test-day classes (PortScan, DDoS) are
  dumped into BENIGN — so on this dataset even the headline metric is unsafe under honest evaluation.

Note on two macro-F1 denominators: `e3_*_results.csv` averages over classes present in the test
set; `eval_metrics_e3.py` per-class files average over ALL 15 classes (zero-test ones count as 0)
and so read lower for timesort (orig ~0.05–0.09, corr ~0.05–0.12). Pick one for Methods — the
all-15 version is the more conservative, harder-to-challenge number.

### chrono_rank fix (17 Jul) — affects original/timesort only
Original review found `chrono_rank` matched the last (generic) keyword instead of the most specific
one, collapsing all Thu files to rank 5 and all Fri files to rank 9. That made original/timesort's
within-Friday order alphabetical, not chronological. Fixed to pick the longest matching key (ranks
now 0,1,2,3,4,6,7,8). After the fix, original/timesort strands DDoS+PortScan (was Bot/PortScan),
matching corrected/timesort; RF acc 0.560 / macro-F1 0.145 (superseded the pre-fix 0.814 / 0.209).
Calendar splits and corrected/timesort were unaffected.

## Honesty caveats (for Methods / Limitations)

1. **Timesort ordering is exact only for corrected data.** The corrected WTMC CSVs carry a
   `Timestamp` column (`DD/MM/YYYY hh:mm:ss AM/PM`), so `timesort` sorts by true time. The original
   MachineLearningCVE CSVs have **no timestamp**, so `timesort` there orders by chronological *file*
   sequence (Mon, Tue, Wed, Thu-AM, Thu-PM, Fri-AM, Fri-PM-PortScan, Fri-PM-DDoS) then within-file
   row order. This is an approximation; state it. It does **not** change the degeneracy conclusion,
   which is driven by day-level attack scheduling, not sub-day ordering.
2. **Friday afternoon sub-order** (PortScan before DDoS) follows the published CICIDS2017 schedule;
   it only affects which of the two lands deepest in the last 20% and does not change the partition.
3. **Duplicate rows** are dropped exactly as in E1/E2 (keep first = earliest occurrence).
4. The **calendar split is the primary, fully-robust condition** — identical method on both
   datasets. Lead the E3 write-up on it; treat timesort as a corroborating second degenerate variant.

## What E3 feeds

E3's degeneracy is the on-page justification for **E4** (task 2.4b): a leakage-free
group-by-five-tuple split on corrected data — the paper's actual novel contribution.
