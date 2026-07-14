# cicids2017-leakage-free-eval

Does correcting CICIDS2017's known labelling errors change what a leakage-free evaluation reveals? A controlled comparison of classical ML classifiers on **original vs. Engelen-corrected CICIDS2017**, under random, naive chronological, and leakage-free (group-by-five-tuple) splits, with per-attack-class metrics and SHAP interpretability.

Companion code for a submission to ICISET 2026.

## Motivation

Prior work has separately shown that (a) CICIDS2017 contains documented labelling and flow-construction errors (Engelen et al. 2021; Liu et al. 2022; Lanvin et al. 2023), and (b) naive evaluation protocols on this dataset are unreliable — a plain chronological split is degenerate (Moczkodan & Ragab 2026), while a non-degenerate temporal split reveals severe per-class detection collapse on the *original* labels (Bouke et al. 2026). Nobody has re-run this kind of evaluation on Engelen's corrected labels to test whether that collapse is partly a labelling artifact. This repo does that controlled comparison.

## Repo structure

```
.
├── data/                 # NOT checked in — see data/README.md for download instructions
├── src/
│   ├── preprocessing.py  # cleaning, dtype downcasting, feature selection
│   ├── splits.py         # random / naive-chronological / leakage-free group-by-five-tuple
│   ├── train_eval.py     # model training + metrics (per-class F1, macro-F1, FPR, DR)
│   └── shap_analysis.py  # SHAP per-attack-class interpretability
├── results/              # small CSV metric tables + figures (checked in)
├── requirements.txt
└── LICENSE
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then follow `data/README.md` to download the datasets — none are included in this repo.

## Datasets used

| Version | Source |
|---|---|
| Original CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| WTMC2021-corrected (Engelen et al. 2021) | https://intrusion-detection.distrinet-research.be/WTMC2021/ |
| LYCOS-IDS2017 (Rosay et al.) | linked from the Rosay et al. paper, DOI 10.1145/3486622.3493973 |

## Status

Work in progress — experiments not yet run. See commit history / releases for current state.

## Citation

If you use this code, please cite the paper (citation to be added on publication) and the datasets above.

## License

MIT — see LICENSE.
