# cicids2017-leakage-free-eval

Code and results for the ICISET 2026 paper "Does Reported Intrusion Detection Performance Survive Honest Evaluation? Label Correction and Leakage-Free Splits on CICIDS2017" (accepted, oral presentation).

The study evaluates five classical per-flow classifiers on the original CICIDS2017 release and on the label-corrected release of Engelen et al. (WTMC 2021), under four split protocols: random, two temporal variants, and a leakage-free split that keeps each network conversation (five-tuple) on one side of the cut. All 15 classes are reported individually with their test support.

## Experiments

| ID | Data | Split | Script | Guide |
| --- | --- | --- | --- | --- |
| E1 | original | random 70/30, stratified | `src/run_e1_baseline.py` | `RUN_E1.md` |
| E2 | corrected | random 70/30, stratified | `src/run_e2_corrected.py` | `RUN_E2.md` |
| E3 | original and corrected | calendar split and time-sorted split | `src/run_e3_temporal.py` | `RUN_E3.md` |
| E4 | corrected | group by five-tuple, 70/30, seeds 2, 42, 123 | `src/run_e4_group.py` | `RUN_E4.md` |
| SHAP | corrected | E4 random forest, seed 42 | `src/run_shap_e4.py` | `RUN_SHAP.md` |

Models: random forest, decision tree, logistic regression, k-NN, linear SVM (SGD). E1 to E3 use seed 42.

Statistics and figures:

- `src/eval_metrics*.py`, `src/eval_perclass.py`, `src/macro_f1.py`: per-class precision, recall, F1, support, macro-F1, FPR, FNR
- `src/bootstrap_ci_e4.py`: bootstrap percentile intervals over 1,000 resamples of test conversations (not individual flows)
- `src/wilcoxon_perclass.py`: per-class paired Wilcoxon test, E2 against E4
- `src/reconcile_table1.py`: rebuilds the paper's Table I from the result CSVs
- `src/make_fig_*.py`: paper figures

`src/preprocessing.py`, `src/train_eval.py` and `src/shap_analysis.py` are early scaffolding. Each `run_*.py` script is self-contained and is what produced the paper's numbers.

## Leakage guards

These run automatically and abort the run on failure:

1. Feature leak: `run_e4_group.py` raises if any identifier column (Flow ID, source/destination IP, source port, timestamp) is in the feature matrix. Destination port and protocol are kept as features.
2. Group leak: `run_e4_group.py` raises `GROUP LEAKAGE` if any five-tuple appears in both train and test.
3. Reproduction: `bootstrap_ci_e4.py` and `reconcile_table1.py` rebuild the saved test partitions and per-seed scores and refuse to report if they do not match exactly.

The five-tuple group key is built before identifiers are dropped.

## Data

Datasets are not included (size and redistribution terms). See `data/README.md`.

| Release | Source |
| --- | --- |
| Original CICIDS2017 (MachineLearningCSV) | https://www.unb.ca/cic/datasets/ids-2017.html |
| Corrected, Engelen et al. WTMC 2021 | https://intrusion-detection.distrinet-research.be/WTMC2021/ |

On the corrected release, flows labelled "- Attempted" are relabelled BENIGN, following Engelen et al. Point the scripts at your copy with `DATA_DIR=/path/to/csvs`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then follow the `RUN_*.md` guide for each experiment.

## Results

`results/` holds the per-model, per-class and confusion-matrix CSVs for every experiment, plus figures in `results/figures/`. Large intermediate files (cleaned data, split arrays, predictions, SHAP cache) are not committed; the scripts regenerate them.

## Citation

Please cite the paper (full reference to be added once the proceedings are published) and both dataset releases above.

## License

MIT, see `LICENSE`.
