# RUN_E4 — leakage-free group-by-five-tuple evaluation (task 2.4b, the novel result)

E4 re-scores the **E2 corrected-data pipeline** under a **leakage-free group-by-five-tuple split**
instead of a random 70/30 split: whole conversations (five-tuples) are assigned to one side, so no
conversation straddles train/test. The split protocol follows Moczkodan and Ragab (2026).

Confirmed config: **corrected data first · 70/30 · 3 seeds {2, 42, 123}**.

## Prerequisites
- The corrected (WTMC2021) CSVs, see `data/README.md`. Point the script at them with `DATA_DIR=/full/path`.
- Python env with `pandas`, `numpy`, `scikit-learn`, `matplotlib` (the repo `venv/`).
- `results/e2_corrected_results.csv` present (so the summary can compute Δ macro-F1 vs E2). Optional.

## Run (laptop, full data — the ONLY commands you need)
```bash
cd cicids2017-leakage-free-eval
# 1) train all 5 models over the 3 group-split seeds (writes per-seed npz + predictions + results.csv)
DATASET=corrected python src/run_e4_group.py
# 2) aggregate per-class mean±std across seeds + summary with Δ vs E2
DATASET=corrected python src/eval_metrics_e4.py
```
Runtime: similar to one E2/E3 condition × 3 (roughly ~1–2 h on a laptop; RandomForest + KNN dominate).
`SEEDS`, `ONLY_MODELS`, `TEST_SIZE`, `KNN_MAX_TRAIN` are env-overridable if you need a partial run.

## What to expect in the log (sanity checks that print automatically)
- `Concatenated: X=(~1.84M, 78)` — 78 feature cols (same as E2). A different number prints a loud warning.
- `Relabelled ~9,144 '- Attempted' flows -> BENIGN` — Engelen relabel fired.
- `~15 classes`, and per seed: `groups A/B (disjoint OK)` + `row test-frac≈0.30`.
  - **`disjoint OK`** is the leakage-free proof — the script raises `AssertionError: GROUP LEAKAGE`
    if any five-tuple were in both sides. It also raises if any identifier column leaks into `X`.
- Per seed, per model: `acc / macroF1 / FPR / FNR`. **macro-F1 should land near E2's, not collapse**
  (the collapse belongs to the *time* split, E3). If it collapses, something is wrong — check the log.

## Outputs (in `results/`; E1/E2/E3 files are never touched)
```
cleaned_e4_corrected_full.pkl                  # cleaned data (retains __group__); reused across seeds
e4_corrected_group_seed<seed>_full.npz         # split+scaled arrays (Xtr/Xte/ytr/yte + classes) — for SHAP (3.1b)
e4_corrected_group_seed<seed>_full_preds.npz   # yte/ytr + per-model predictions (eval reads these; no retrain)
e4_corrected_group_results.csv                 # every seed×model: acc/macro_f1/weighted_f1/fpr/fnr
e4_corrected_group_perclass_<MODEL>.csv        # per-class P/R/F1 mean±std + support + zero-support status
e4_corrected_group_summary.csv                 # per-model mean±std + Δ macro_f1 vs E2  ← the C2 headline
figures/confusion_e4_corrected_<MODEL>.png     # confusion (seed 2, for a visual)
```

## Re-running / cache hygiene (avoids the E2/E3 stale-cache trap)
Caches are tagged `full` / `cap<N>` / `smoke<N>`, so a full run can never silently reuse a subsample,
and a full run under `MIN_EXPECTED_FULL_ROWS` prints a loud warning. To force a clean rebuild:
```bash
rm -f results/cleaned_e4_corrected_full.pkl results/e4_corrected_group_seed*_full*.npz
```

## Phase 2 — original-vs-corrected contrast (the sharpest comparison; run after Phase 1)
```bash
DATASET=original python src/run_e4_group.py     # uses GeneratedLabelledFlows CSVs (have the 5-tuple)
DATASET=original python src/eval_metrics_e4.py
```
`DATASET=original` prints a warning: its column config is wired but **not yet smoke-tested** — verify
the log (feature count, key cols, labels) before trusting the numbers. Note in Methods that `original`
here uses the LabelledFlows CSVs, a different source file than E1's MachineLearningCVE CSVs.

## Caveats to carry into Methods / Limitations
- We adopt Moczkodan & Ragab's **split** but not their windowing/sequence task — E4 is **per-flow**.
- Ratio **70/30** (theirs was 80/20).
- Split is by **conversation-groups** (`GroupShuffleSplit`), so the row-level test fraction is
  *approximately* 0.30 (logged each seed), not exactly.
- We keep all **15 classes** (Moczkodan dropped <2-example classes) → 2 extra near-zero classes,
  reported honestly via the `status` column.
- A single-conversation class can land entirely on one side → **structural** ZERO_TRAIN/ZERO_TEST,
  a property of the data + protocol, flagged per class, not a model failure.
- Two macro-F1 views (same as E3): the **results.csv** macro (over classes present in a seed's test) and
  the **per-class** aggregate; both honest, the per-class CSV has the detail.
