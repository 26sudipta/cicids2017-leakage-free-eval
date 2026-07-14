"""
splits.py — the three evaluation protocols compared in this paper.

1. random_split()
       Standard random 70/30 (or similar). Included as the naive baseline everyone
       else reports, to show the inflation it produces.

2. naive_chronological_split()
       Sort all flows by timestamp, take the first 80% as train / last 20% as test.
       Expected to be near-degenerate on a single-week capture (per Moczkodan & Ragab
       2026 — most attack classes only appear in one part of the week). Report this as
       a documented sanity check, not the paper's main result. See roadmap task 2.4.

3. leakage_free_group_split()
       Group flows by five-tuple (src IP, dst IP, src port, dst port, protocol) —
       i.e. by conversation — before splitting, so no conversation appears on both
       sides of the train/test boundary. This is the validated, non-degenerate
       protocol (per Moczkodan & Ragab 2026; Bouke et al. 2026 use a similar
       Mon-Thu/Fri temporal variant). This is the paper's main experiment (task 2.4b).

TODO: implement — not yet run.
"""

import pandas as pd


def random_split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 0):
    raise NotImplementedError("TODO: task 2.1")


def naive_chronological_split(df: pd.DataFrame, timestamp_col: str, train_frac: float = 0.8):
    """Documented-degenerate sanity check — see Moczkodan & Ragab 2026."""
    raise NotImplementedError("TODO: task 2.4")


def leakage_free_group_split(df: pd.DataFrame, five_tuple_cols: list[str], test_size: float = 0.3, seed: int = 0):
    """Main experiment — group-by-five-tuple split. See task 2.4b."""
    raise NotImplementedError("TODO: task 2.4b")
