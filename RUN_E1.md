# How to run E1 (baseline reproduction) yourself

This is the plain-English guide to `src/run_e1_baseline.py`. The goal of E1 is to
reproduce the "~98% accuracy" that the base paper reports on original CICIDS2017,
so later experiments can show how much of that number is an artifact.

## On your laptop it is ONE command

```bash
cd cicids2017-leakage-free-eval
pip install pandas numpy scikit-learn      # once
python src/run_e1_baseline.py              # full dataset, all 5 models
```

Output: `results/e1_baseline_results.csv` (one row per model) and a cached clean
dataset at `results/cleaned_original.pkl`.

Everything below about chunking, `MAX_ROWS`, `ONLY_MODELS`, and per-model runs is
ONLY needed on a tiny/limited machine. On a normal laptop (8 GB+), ignore all of it.

## What the script does, and WHY (5 stages)

1. **Load + concatenate** the 8 daily CSVs into one table (~2.8M network flows).
   - WHY strip column names: every column has a leading space (`" Label"`), so
     lookups silently fail if you don't `df.columns.str.strip()`.

2. **Clean** the features.
   - Coerce every feature column to a number (a few cells contain stray text).
   - Replace `Infinity`/`-Infinity` with `NaN`. These come from `Flow Bytes/s` and
     `Flow Packets/s`, which divide by flow duration — zero-duration flows blow up.
   - Downcast to `float32` (half the memory of float64; harmless here).
   - Drop exact-duplicate rows — a documented CICIDS2017 defect the base paper removes.

3. **Split off the label** and encode it (15 classes: BENIGN + 14 attack types).

4. **Random 70/30 train/test split, stratified** so rare attacks (e.g. Heartbleed,
   ~11 flows total) still appear in both halves.
   - Then fit the median imputer and StandardScaler **on the training set only**.
     WHY: fitting them on all the data leaks test-set statistics into training.
     (The base paper cleaned globally; we do it the correct way.)

5. **Train + evaluate 5 classifiers**: Decision Tree, Random Forest, KNN,
   Logistic Regression, Linear SVM. For each we record accuracy, macro-F1,
   weighted-F1, saved to CSV.

## How to read the result

Accuracy will look great (~96–99.6%). That is the trap. Because ~80% of traffic is
benign, a model that ignores every attack still scores ~80%. **Macro-F1** weights
each class equally, so it exposes the missed minority attacks. The gap between the
two columns is the entire motivation for the paper — lead with macro-F1 and FPR,
never accuracy.

## Two things to restore for the real (laptop) run

The sandbox forced two compromises. On your laptop, undo them for the canonical run:

- **Use the full dataset.** Do NOT set `MAX_ROWS`. The sandbox numbers came from a
  60k subsample; your paper's table must come from the full ~2.8M rows.
- **(Optional) LinearSVC vs SGD.** The script uses `SGDClassifier(loss="hinge")`
  (a linear SVM by SGD) because liblinear was too slow in the sandbox. On a laptop
  you may switch back to `LinearSVC(dual=False)` if you want the exact solver — the
  line is commented in the script. Either is a defensible "linear SVM" baseline;
  just state which one you used in the Methods section.

## Sandbox-only flags (ignore on a laptop)

- `MAX_ROWS=500000` — cap rows so it fits in small RAM.
- `STAGE=prep` — only build the cleaned cache, then stop.
- `ONLY_MODELS=RandomForest,KNN` — train just those models (so each fits a short run).
