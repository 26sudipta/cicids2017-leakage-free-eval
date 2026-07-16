# How to run E2 (corrected-data comparison) yourself

E2 runs the **exact same modelling pipeline as E1**, but on Engelen's **corrected** CICIDS2017
(the WTMC2021 release) instead of the original public version. It is the controlled comparison
for the "does correcting the data change the reported score?" leg of the paper.

Only two things differ from `run_e1_baseline.py`, both forced by the corrected dataset and both
settled by Engelen's own paper + code (full reasoning: `../Literature/Engelen_2021_E2_Implications_Summary.md`):

1. **Drop host-identifier columns** the corrected CSVs carry but the original ML CSVs never had:
   `Flow ID, Src IP, Src Port, Dst IP, Timestamp`. **Keep** `Dst Port` + `Protocol` + flow
   features. (Leaving the identifiers in would leak attack identity — the shortcut learning
   Engelen §III-D warns about — and that leakage was not in E1, so it would invalidate the
   comparison.) After the drop you get **78 feature columns — the same count as E1.**
2. **Relabel every `X - Attempted` flow to BENIGN** (Engelen's default). These are empty/failed
   connections with no attack payload; a per-flow detector cannot tell them from benign. The
   flows stay in the data (honest evaluation), they just get the label the model can justify.

Everything else is identical to E1: drop duplicate rows, stratified 70/30 split, median imputer +
StandardScaler fit on train only, the same 5 models, `SEED=42`, and the same metrics.

## On your laptop it is ONE command

```bash
cd cicids2017-leakage-free-eval
pip install pandas numpy scikit-learn        # once (same env as E1)
python src/run_e2_corrected.py               # full corrected dataset, all 5 models
```

Outputs (E1 files are never touched):
- `results/e2_corrected_results.csv` — one row per model (accuracy, macro-F1, weighted-F1)
- `results/cleaned_corrected_full.pkl` — cached clean data (tagged `full`)
- `results/e2_prepped_arrays_full.npz` — cached split+scaled arrays (tagged `full`)
- `results/label_classes_e2.npy` — the ~15 class names, for per-class metrics

A full run should take **minutes, not seconds** (E1 took ~15 min for all 5 models). **If it
finishes in seconds, it trained on a subsample — stop and check.** Caches are now tagged by row
setting (`full` vs `cap<N>`), so a full run can never silently reuse a smoke-test subsample, and
the script prints a loud warning if a "full" run has fewer than 1,000,000 rows. The run log also
prints `Train (…, 78)  Test (…, 78)` — the full run's train rows should be ~1.7M, not ~80k.

> Housekeeping: earlier smoke tests left inert untagged files (`cleaned_corrected.pkl`,
> `e2_prepped_arrays.npz`) and a subsample `e2_corrected_results.csv` in `results/`. The full run
> overwrites the results CSV; the two `.pkl`/`.npz` leftovers are ignored — delete them manually
> whenever convenient.

## Then get the per-class / confusion / FPR tables (as in E1's task 2.2)

`eval_metrics.py` was written for E1's arrays. Point it at the E2 cache (or copy the E1 command
you used, swapping the prepped-array and label-class files for the `e2_*` versions). Lead with
**macro-F1 and per-class F1**, never accuracy.

## What to expect (do NOT pre-write it as a collapse)

Based on Engelen's own Table II, correcting the *labels* tends to leave aggregate accuracy /
weighted-F1 ~unchanged (still ~0.99) while **per-class F1 stays flat or improves** — because the
correction removes the mislabelled artefact flows the model was overfitting. The genuine
performance *collapse* the paper argues for is expected in **E3/E3b (the leakage-free split)**,
not here. E2's job is to show that aggregate metrics are blind to a 20%+ change in the data.

## Comparing E1 vs E2 (the actual deliverable)

Put the two `*_results.csv` side by side and read the **macro-F1 and per-class** columns, not
accuracy. Note two honest caveats in Methods:
- The corrected release changed labels **and** regenerated features (patched CICFlowMeter) **and**
  fixed flow construction at once — so this is a "public version vs corrected version" swap, not a
  surgical labels-only change.
- E1 used a 70/30 split and E2 keeps 70/30 for consistency; Engelen used 75/25.

## Constrained-machine flags (same as E1, ignore on a normal laptop)

- `MAX_ROWS=800000` — cap rows to fit small RAM.
- `STAGE=prep` — build the cleaned cache then stop.
- `ONLY_MODELS=RandomForest,KNN` — train just those.
- `KNN_MAX_TRAIN=150000` — stratified reference-set cap for KNN (a lazy learner), same as E1.
- `DATA_DIR=/path/to/corrected_csvs` — override the data folder.
