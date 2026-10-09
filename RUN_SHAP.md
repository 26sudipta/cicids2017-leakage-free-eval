# RUN_SHAP — task 3.1b (contribution C4): SHAP on the E4 RandomForest

**What this produces:** per-class SHAP feature-importance signatures for the E4 model
(corrected data, leakage-free group split), comparing classes that **degrade** under the
leakage-free split (Infiltration, WA-SQL Injection, WA-XSS) against classes that stay
**stable** (BENIGN, DoS Hulk, DDoS, PortScan, FTP-Patator, WA-Brute Force).
This is the interpretability contribution nobody else has in the corrected × leakage-free
setting.

## Run (laptop)

```bash
pip install shap          # one-time
cd cicids2017-leakage-free-eval
SEED=42 python src/run_shap_e4.py
```

Expected runtime: RF retrain ~5–10 min (same as one E4 seed) + SHAP. **Honest uncertainty:**
TreeSHAP cost grows with tree depth/leaf count and this forest is trained on 1.29M rows, so
SHAP could take minutes or over an hour — it was not benchmarkable at full scale in the
sandbox. The script is built so this doesn't matter:

- Classes run **smallest-first**: the degraded classes (the C4 contribution) finish in the
  first minutes; BENIGN (n=300, least novel) runs last.
- Every finished class is **cached immediately** to `results/shap_cache_RandomForest_seed42_full/`.
  If you interrupt or it crashes, rerun the same command — finished classes are skipped
  (only the RF retrain repeats). `SHAP_FORCE=1` recomputes.
- A single class failing is logged and skipped, never fatal.
- Watch the `s/sample` number on the first classes: if BENIGN's projected time is absurd,
  Ctrl-C after the attack classes finish and rerun with `SAMPLES_PER_CLASS=100`.

If `pip install shap` fails (numba/numpy version clash), try
`pip install --upgrade numba llvmlite shap`.

## What "seed 42" means

A random seed is the number that initialises the random-number generator, so a "random"
process gives the **same** result every time it runs — that's what makes experiments
reproducible. E4's 70/30 group split is random (which conversations land in test), so it
was run three times with seeds 2, 42, and 123 = three different random partitions, and the
paper reports the mean ± std across them. 42 has no special meaning (it's a programming
in-joke from *The Hitchhiker's Guide to the Galaxy*); it is simply one of your three E4
seeds. SHAP explains ONE of those three splits — the seed-42 one — for the reason below.

## Why SEED=42, not 2

Seed 2 (the only seed with saved arrays) has **zero Infiltration test rows** — one of the
three degraded classes would be unexplainable. Seed 42 has all focus classes in test. The
script rebuilds the seed-42 split deterministically from `cleaned_e4_corrected_full.pkl`
and then **verifies the retrained model's test predictions exactly match the saved E4
predictions** (`REPRODUCIBILITY GUARD ... = 1.000000` must appear in the log). If the guard
fails, stop — do not use the outputs.

## Outputs

- `results/shap_e4_RandomForest_seed42_meanabs.csv` — mean |SHAP| for all 78 features × each focus class
- `results/shap_e4_RandomForest_seed42_topk.csv` — top-10 features per class, with share-of-total
- `results/figures/fig3_shap_degraded_vs_stable.png` — the paper figure (grayscale-safe heatmap)

## What to check in the log

1. `REPRODUCIBILITY GUARD: prediction match ... = 1.000000`
2. `skip 'Heartbleed'` does NOT appear (it's not in the focus lists) — but if you add it,
   it must be skipped as ZERO_TRAIN. The model never learned Heartbleed; that fact goes in
   the paper, not under the rug.
3. Each degraded class reports its `n=` — expect tiny numbers (WA-SQL n≈6, WA-XSS n≈6,
   Infiltration n≈8–25 depending on seed). **These are qualitative signatures, not
   population estimates** — say so in the paper.

## Caveats to carry into the Methods/Limitations text

- `feature_perturbation="tree_path_dependent"` (SHAP default without background data).
- Signatures are from ONE seed (42); the E4 performance numbers remain 3-seed means.
- Smoke-tested end-to-end (both the saved-npz path and the rebuild path) on data slices
  in the sandbox, 24 Jul; full run pending on laptop.
- The smoke test caught and fixed a real indexing bug: SHAP output heads follow
  `clf.classes_` (14 learned classes — no Heartbleed), not the full 15-class label set.
