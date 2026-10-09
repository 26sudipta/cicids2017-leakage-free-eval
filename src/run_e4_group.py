"""
E4 - LEAKAGE-FREE GROUP-BY-FIVE-TUPLE split experiment (task 2.4b; the paper's novel result).

Same modelling pipeline as E2 (corrected data). The ONLY thing that changes vs. E2's random
70/30 split is HOW train/test is formed: whole CONVERSATIONS (five-tuples) are assigned to one
side, so no conversation straddles the boundary. This removes within-conversation leakage
(CICFlowMeter splits one connection into many flow records that a random split scatters across
train and test). Replicates the leakage-free "group-by-five-tuple" protocol of Moczkodan & Ragab
(arXiv 2606.11098) -- but PER-FLOW (no windowing) and on Engelen-corrected labels (the open gap).

Grouping key (Moczkodan's canonical five-tuple; USED FOR GROUPING ONLY, never a feature):
    (Src IP, Dst IP, Src Port, Dst Port, Protocol)
Built BEFORE identifiers are dropped. Src IP / Src Port / Dst IP are then dropped from the
features; Dst Port + Protocol are KEPT as features (Engelen recipe) -> 78 feature cols, same as E2.

Parameterised by env vars:
    DATASET   = corrected          (Phase 1; 'original' via LabelledFlows is Phase 2 -- untested)
    SEEDS     = 2,42,123           (each seed = a different random group partition + model seed)
    TEST_SIZE = 0.30               (fraction of GROUPS in test; held at 70/30 like E1/E2/E3)

Guards baked in (lessons from E3's bugs):
  * classes embedded INSIDE each npz (never a stale separate .npy)
  * caches tagged full/cap/smoke so a full run can't silently reuse a subsample
  * a <MIN rows guard that warns loudly on a suspiciously small full run
  * KNN train-subsample has a non-stratified fallback (rare classes can have <2 rows)
  * FEATURE-LEAK assert: the identifier cols are never in X (and Dst Port + Protocol ARE kept)
  * LEAKAGE-FREE assert: train and test group-sets are disjoint (the whole point of E4)

Run (laptop, full):   DATASET=corrected python src/run_e4_group.py
Smoke (slice):        DATASET=corrected SMOKE_NROWS=8000 SEEDS=42 ONLY_MODELS=DecisionTree \
                      DATA_DIR=/path/to/corrected python src/run_e4_group.py
Outputs (E1/E2/E3 files are never touched):
   results/cleaned_e4_<dataset>_<cachetag>.pkl            cleaned data, retains __group__
   results/e4_<dataset>_group_seed<seed>_<cachetag>.npz   split+scaled arrays, classes embedded
   results/e4_<dataset>_group_results.csv                 all seeds x models: acc/macroF1/wF1/FPR/FNR
"""
import glob, os, time, collections
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
# Eq. (2) macro-F1 over C+. Do NOT replace with a bare f1_score(average="macro"):
# see the header of src/macro_f1.py. This path hit the bug at seed 2 only.
from macro_f1 import macro_f1_cplus, weighted_f1_cplus, audit_macro

DATASET = os.environ.get("DATASET", "corrected").strip().lower()

# ---- per-dataset config -----------------------------------------------------
# key_cols  : build the five-tuple grouping key (grouping only, NOT features)
# drop_cols : identifier/time cols removed from the feature matrix. Dst Port + Protocol are
#             deliberately NOT here -> they stay as features (Engelen recipe).
CONFIG = {
    "corrected": {   # WTMC2021 corrected CSVs (Phase 1)
        "key_cols":  ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"],
        "drop_cols": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"],
        "keep_feats": ["Dst Port", "Protocol"],
        "label_col": "Label",
        "relabel_attempted": True,     # 'X - Attempted' -> BENIGN (Engelen default)
        "expect_feats": 78,
    },
    # Phase 2 (NOT yet smoke-tested -- verify columns/labels before trusting):
    "original": {    # original GeneratedLabelledFlows CSVs (these HAVE the five-tuple; the
                     # MachineLearningCVE CSVs E1 used do NOT)
        "key_cols":  ["Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol"],
        "drop_cols": ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"],
        "keep_feats": ["Destination Port", "Protocol"],
        "label_col": "Label",
        "relabel_attempted": False,    # original labels have no 'Attempted' variants
        "expect_feats": None,
    },
}
if DATASET not in CONFIG:
    raise SystemExit(f"DATASET must be one of {list(CONFIG)}")
CFG = CONFIG[DATASET]
if DATASET == "original":
    print("!! WARNING: DATASET=original is Phase 2 and has NOT been smoke-tested. "
          "Verify columns/labels before trusting results.", flush=True)

SEEDS = [int(s) for s in os.environ.get("SEEDS", "2,42,123").split(",") if s.strip()]
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.30"))
HERE = os.path.dirname(__file__)
OUT_DIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

_RESEARCH = os.path.expanduser("~/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets")
_CANDIDATES = {
    "corrected": [os.path.join(HERE, "..", "data", "corrected_wtmc2021"),
                  os.path.join(_RESEARCH, "CICIDS2017_Corrected_WTMC2021")],
    "original":  [os.path.join(HERE, "..", "data", "original_labelledflows"),
                  os.path.join(_RESEARCH, "CICIDS2017_Original_LabelledFlows")],
}[DATASET]


def _resolve_data_dir():
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    for c in _CANDIDATES:
        if glob.glob(os.path.join(c, "*.csv")):
            return c
    return _CANDIDATES[0]


DATA_DIR = _resolve_data_dir()

SMOKE_NROWS = int(os.environ.get("SMOKE_NROWS", "0")) or None
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0")) or None
if SMOKE_NROWS:
    CACHE_TAG = f"smoke{SMOKE_NROWS}"
elif MAX_ROWS:
    CACHE_TAG = f"cap{MAX_ROWS}"
else:
    CACHE_TAG = "full"
CLEAN_PICKLE = os.path.join(OUT_DIR, f"cleaned_e4_{DATASET}_{CACHE_TAG}.pkl")
RESULTS_CSV  = os.path.join(OUT_DIR, f"e4_{DATASET}_group_results.csv")
MIN_EXPECTED_FULL_ROWS = 400_000
ONLY_MODELS = [m for m in os.environ.get("ONLY_MODELS", "").split(",") if m]
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))
GROUP_COL, LABEL_KEY = "__group__", "__label__"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def relabel_attempted(y):
    return y.where(~y.str.endswith("- Attempted"), "BENIGN")


# ---- 1. LOAD + CLEAN --------------------------------------------------------
def load_and_clean():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSVs for DATASET={DATASET} in DATA_DIR={DATA_DIR!r}. Candidates: {_CANDIDATES}. "
            f"Set DATA_DIR=/path explicitly.")
    log(f"[{DATASET}] {len(files)} CSVs from {DATA_DIR}")
    key_cols, drop_cols, label_col = CFG["key_cols"], CFG["drop_cols"], CFG["label_col"]

    Xparts, yparts, gparts = [], [], []
    n_rows, n_attempted = 0, 0
    for f in files:
        reader = ([pd.read_csv(f, nrows=SMOKE_NROWS, low_memory=False)] if SMOKE_NROWS
                  else pd.read_csv(f, low_memory=False, chunksize=150_000))
        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()
            miss = [c for c in key_cols if c not in chunk.columns]
            if miss:
                raise KeyError(f"{os.path.basename(f)} missing key cols {miss}; cannot build the "
                               f"five-tuple. First cols: {list(chunk.columns)[:8]}")
            # label ---------------------------------------------------------
            y = chunk[label_col].astype(str).str.strip()
            if CFG["relabel_attempted"]:
                n_attempted += int(y.str.endswith("- Attempted").sum())
                y = relabel_attempted(y)
            # five-tuple group key (built BEFORE dropping identifiers) ------
            g = (chunk[key_cols[0]].astype(str) + "|" + chunk[key_cols[1]].astype(str) + "|"
                 + chunk[key_cols[2]].astype(str) + "|" + chunk[key_cols[3]].astype(str) + "|"
                 + chunk[key_cols[4]].astype(str))
            # features: drop label + identifiers; KEEP Dst Port + Protocol --
            drop = [label_col] + [c for c in drop_cols if c in chunk.columns]
            Xp = (chunk.drop(columns=drop)
                       .apply(pd.to_numeric, errors="coerce")
                       .replace([np.inf, -np.inf], np.nan)
                       .astype(np.float32))
            Xparts.append(Xp)
            yparts.append(y.reset_index(drop=True))
            gparts.append(g.reset_index(drop=True))
            n_rows += len(chunk)
            del chunk
        log(f"  loaded {os.path.basename(f)}  (rows so far: {n_rows:,})")

    X = pd.concat(Xparts, ignore_index=True)
    y = pd.concat(yparts, ignore_index=True)
    g = pd.concat(gparts, ignore_index=True)
    del Xparts, yparts, gparts
    log(f"Concatenated: X={X.shape}  ({X.shape[1]} feature cols)")
    if CFG["relabel_attempted"]:
        log(f"Relabelled {n_attempted:,} '- Attempted' flows -> BENIGN")

    # FEATURE-LEAK GUARD: identifiers gone; Dst Port + Protocol still present.
    leaked = [c for c in ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp",
                          "Source IP", "Source Port", "Destination IP"] if c in X.columns]
    if leaked:
        raise AssertionError(f"identifier columns leaked into features: {leaked}")
    for keep in CFG["keep_feats"]:
        if keep not in X.columns:
            raise AssertionError(f"expected feature '{keep}' missing from X (Engelen keeps it)")
    if CFG["expect_feats"] and X.shape[1] != CFG["expect_feats"]:
        log(f"!! WARNING: {X.shape[1]} feature cols but expected {CFG['expect_feats']} "
            f"(check the CSV schema before trusting results)")

    if MAX_ROWS and len(X) > MAX_ROWS:
        try:
            idx = train_test_split(np.arange(len(X)), train_size=MAX_ROWS, stratify=y,
                                   random_state=42)[0]
        except ValueError:
            idx = train_test_split(np.arange(len(X)), train_size=MAX_ROWS, random_state=42)[0]
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        g = g.iloc[idx].reset_index(drop=True)
        log(f"Subsampled to {len(X):,} rows (MAX_ROWS)")

    # drop exact-duplicate feature+label rows (E2/E3 defect handling); carry group along
    before = len(X)
    key = X.copy(); key[LABEL_KEY] = y.values
    keep_mask = ~key.duplicated(); del key
    X = X[keep_mask].reset_index(drop=True)
    y = y[keep_mask].reset_index(drop=True)
    g = g[keep_mask].reset_index(drop=True)
    log(f"Dropped {before - len(X):,} duplicate rows -> {len(X):,} remain; "
        f"{g.nunique():,} unique five-tuple groups")

    out = X.copy()
    out[LABEL_KEY] = y.values
    out[GROUP_COL] = g.values
    return out


# ---- 2. GROUP SPLIT + PREPROCESS (fit on TRAIN ONLY) ------------------------
def prepare(df, seed, le):
    y_raw = df[LABEL_KEY].values
    groups = df[GROUP_COL].values
    X = df.drop(columns=[LABEL_KEY, GROUP_COL])
    y_enc = le.transform(y_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr, te = next(gss.split(X.values, y_enc, groups=groups))

    # LEAKAGE-FREE GUARD: no conversation may appear on both sides.
    inter = set(groups[tr]) & set(groups[te])
    if inter:
        raise AssertionError(f"GROUP LEAKAGE: {len(inter)} five-tuples in both train and test")

    Xtr, Xte = X.values[tr], X.values[te]
    ytr, yte = y_enc[tr], y_enc[te]
    log(f"seed {seed}: train={len(tr):,} rows / test={len(te):,} rows | "
        f"groups {len(set(groups[tr])):,}/{len(set(groups[te])):,} (disjoint OK) | "
        f"row test-frac={len(te)/(len(tr)+len(te)):.3f}")

    ctr, cte = collections.Counter(ytr), collections.Counter(yte)
    only_test  = sorted(le.classes_[i] for i in set(cte) - set(ctr))
    only_train = sorted(le.classes_[i] for i in set(ctr) - set(cte))
    if only_test:
        log(f"  ZERO-TRAIN (unlearnable -> recall 0): {only_test}")
    if only_train:
        log(f"  ZERO-TEST (not scored): {only_train}")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr, yte


def _guard(n_rows):
    if CACHE_TAG == "full" and n_rows < MIN_EXPECTED_FULL_ROWS:
        log("!" * 70)
        log(f"WARNING: full run but only {n_rows:,} rows (<{MIN_EXPECTED_FULL_ROWS:,}). "
            f"Bad cache/subsample? Delete results/*e4*{CACHE_TAG}* and rerun.")
        log("!" * 70)


def models_for(seed):
    m = {
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    return {k: v for k, v in m.items() if not ONLY_MODELS or k in ONLY_MODELS}


def fpr_fnr(yte, pred, benign_idx):
    yte, pred = np.asarray(yte), np.asarray(pred)
    ben = yte == benign_idx
    atk = ~ben
    fpr = float(np.mean(pred[ben] != benign_idx)) if ben.any() else float("nan")
    fnr = float(np.mean(pred[atk] == benign_idx)) if atk.any() else float("nan")
    return fpr, fnr


# ---- 3. RUN -----------------------------------------------------------------
def run():
    t0 = time.time()
    log(f"E4  DATASET={DATASET}  SEEDS={SEEDS}  TEST_SIZE={TEST_SIZE}  mode={CACHE_TAG}")
    if os.path.exists(CLEAN_PICKLE):
        log(f"Loading cached cleaned data {os.path.basename(CLEAN_PICKLE)}")
        df = pd.read_pickle(CLEAN_PICKLE)
    else:
        df = load_and_clean()
        df.to_pickle(CLEAN_PICKLE)
        log(f"Cached clean data -> {CLEAN_PICKLE}")
    _guard(len(df))

    # ONE label encoder over the FULL label set, shared across seeds (stable indices).
    le = LabelEncoder().fit(df[LABEL_KEY].values)
    log(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    benign_idx = int(le.transform(["BENIGN"])[0]) if "BENIGN" in le.classes_ else -1

    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = prepare(df, seed, le)
        # Big arrays are only needed once (for SHAP, task 3.1b) -> save for the FIRST seed only, to
        # avoid writing ~1.7 GB of duplicate npz across 3 seeds. Set SAVE_ALL_NPZ=1 to override.
        if seed == SEEDS[0] or os.environ.get("SAVE_ALL_NPZ"):
            npz = os.path.join(OUT_DIR, f"e4_{DATASET}_group_seed{seed}_{CACHE_TAG}.npz")
            np.savez(npz, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, classes=le.classes_)
        preds = {}                       # saved so eval never has to retrain (halves compute)
        for name, clf in models_for(seed).items():
            ts = time.time()
            Xf, yf = Xtr, ytr
            if name == "KNN" and 0 < KNN_MAX_TRAIN < len(Xtr):
                try:
                    Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                                    stratify=ytr, random_state=seed)
                except ValueError:
                    Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                                    random_state=seed)
            clf.fit(Xf, yf)
            pred = clf.predict(Xte)
            preds[f"pred_{name}"] = pred.astype(np.int32)
            fpr, fnr = fpr_fnr(yte, pred, benign_idx)
            mf1, n_cplus = macro_f1_cplus(yte, pred)
            row = {"dataset": DATASET, "split": "group5tuple", "seed": seed, "model": name,
                   "accuracy": accuracy_score(yte, pred),
                   "macro_f1": mf1,
                   "n_cplus": n_cplus,
                   "weighted_f1": weighted_f1_cplus(yte, pred),
                   "fpr": fpr, "fnr": fnr,
                   "train_rows": len(Xf), "test_rows": len(Xte), "sec": round(time.time() - ts, 1)}
            rows.append(row)
            # An unstratified group split can leave a class with no test flows
            # (Infiltration does this at seed 2). If a model then predicts that
            # class, the sklearn default silently changes the denominator.
            aud = audit_macro(yte, pred)
            if aud["diverges"]:
                log(f"  NOTE seed {seed} {name}: |C+|={aud['n_cplus']} but the sklearn "
                    f"default would average over {aud['n_union']} labels "
                    f"({aud['macro_f1_sklearn_default']:.4f} vs {mf1:.4f}). "
                    f"Reporting the Eq. (2) value.")
            log(f"seed {seed} {name:20s} acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
                f"(|C+|={n_cplus}) FPR={fpr:.4f} FNR={fnr:.4f} ({row['sec']}s)")
        # per-seed predictions + train support, so eval computes per-class WITHOUT retraining
        np.savez(os.path.join(OUT_DIR, f"e4_{DATASET}_group_seed{seed}_{CACHE_TAG}_preds.npz"),
                 yte=yte, ytr=ytr, classes=le.classes_, **preds)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)   # rewrite after each seed (crash-safe)

    log(f"DONE {round(time.time() - t0)}s -> {RESULTS_CSV}")


if __name__ == "__main__":
    run()
