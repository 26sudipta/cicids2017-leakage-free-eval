"""
preprocessing.py — clean and standardize CICIDS2017 CSVs (any of the three versions).

Steps (per roadmap task 2.1):
    1. Load per-day CSVs, concatenate.
    2. Drop duplicate rows.
    3. Replace +/-inf with NaN, impute with per-column median.
    4. Downcast numeric dtypes (float64->float32, int64->int32 where safe) to cut RAM.
    5. Drop host-identifier columns (Flow ID, Source/Destination IP, Timestamp) to avoid
       shortcut learning (see Engelen et al. 2021, Reading_Notes.md item 1).
    6. Save cleaned parquet/csv to data/processed/.

TODO: implement — not yet run. See Paper_Roadmap_and_TODO.md task 2.1.
"""

import pandas as pd
from pathlib import Path

IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    # Destination Port: keep or drop is a methodology decision — state explicitly in the paper.
]


def load_and_clean(input_dir: str, output_path: str) -> pd.DataFrame:
    """Load all CSVs in input_dir, clean, and write the result to output_path."""
    raise NotImplementedError("TODO: task 2.1")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see task 2.1 in Paper_Roadmap_and_TODO.md")
