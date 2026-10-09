# Data

Raw datasets are **not** checked into this repo (size + redistribution restrictions). Download from the official sources below and place them in this folder using the layout shown — the folder is gitignored except this file.

## 1. Original CICIDS2017

Source: https://www.unb.ca/cic/datasets/ids-2017.html
Place CSVs in: `data/original/`

## 2. WTMC2021-corrected (Engelen et al. 2021)

Source: https://intrusion-detection.distrinet-research.be/WTMC2021/
Place CSVs in: `data/corrected_wtmc2021/`

## 3. LYCOS-IDS2017 (Rosay et al.)

Source: linked from the Rosay et al. paper, DOI 10.1145/3486622.3493973
Place CSVs in: `data/lycos_ids2017/`

Note: LYCOS CSVs ship unlabelled ("NeedLabel" placeholder). Run the authors' bundled `labelling.py` then `create_datasets.py` before use.

## Expected layout

```
data/
├── original/
├── corrected_wtmc2021/
└── lycos_ids2017/
```
