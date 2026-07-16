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

## Status (verified 14 Jul 2026)

`data/original`, `data/corrected_wtmc2021`, and `data/lycos_ids2017` are the real data directories, moved here (not copied — instant rename, same filesystem) from `../../Datasets/`. All are gitignored (see .gitignore: `data/*` excluded except this file). Verification results:

- **original** (CICIDS2017_Original_ML_CSVs, 8 files, 844M): row counts match CIC's published per-day statistics (e.g. Thursday Web Attacks: 168,186 BENIGN + 1,507 Brute Force + 652 XSS + 21 SQLi = 170,366, matches exactly). Intact.
- **corrected_wtmc2021** (Engelen et al., 5 files, 1.1G): all 5 day-files present, sizes/line counts plausible, spot-checked via md5 against the duplicate copy in the Research folder — identical. Intact.
- **lycos_ids2017**: **fully processed as of 14 Jul 2026.** `labelling.py` and `create_datasets.py` both ran to completion (three pandas-version bugs fixed along the way — see Paper_Roadmap_and_TODO.md task 1.4 for details). Ready-to-load train/cv/test splits exist at `lycos_ids2017/datasets/cic-ids2017/*.parquet` (440,632 / 220,312 / 220,312 rows, 85 cols) and `lycos_ids2017/datasets/lycos-ids2017/*.parquet` (same row counts, 83 cols, deliberately size-matched to the CIC variant). These still contain identifier columns (flow_id, src/dst addr, ports, timestamp) — the drop-identifiers step in `preprocessing.py` (task 2.1) still applies before training.
- A duplicate ~4.7GB copy of all three dataset sets also exists in the separately-connected Research folder (`/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets/`) — same files, confirmed by checksum. Harmless but redundant; delete one copy manually if you want the disk space back (Claude can't delete files on either connected-folder mount).
- `../../Datasets/` (the project-root folder these moved out of) now only holds `CICIDS2017_Original_LabelledFlows/` (the alternate raw-ID-columns distribution, not used here) — left in place, not moved.
- Three harmless dangling symlinks (`_old_original_symlink`, `_old_corrected_symlink`, `_old_lycos_symlink`) are sitting in this `data/` folder — leftovers from an earlier symlink-based setup that got replaced with real moved directories. They point nowhere now and are gitignored either way, but Claude can't delete files on this mount — delete them yourself whenever convenient, they're not load-bearing.
