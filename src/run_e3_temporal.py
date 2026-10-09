"""
E3 - TEMPORAL (chronological) split experiment.

Same modelling pipeline as E1/E2. The ONLY thing that changes vs. E1 (original data) and
E2 (corrected data) is HOW the train/test split is formed: instead of a random stratified
70/30 split, rows are partitioned by *time*. This is the deliberately-degenerate "naive
temporal split" sanity check (roadmap task 2.4): CICIDS2017 schedules one attack type per
day, so a time split puts entire attack classes on only one side of the boundary. We EXPECT
macro-F1 to collapse (~0.50), replicating Moczkodan & Ragab (arXiv 2606.11098). E3's job is
to prove the naive split is broken, which motivates E4 (the leakage-free group split).

Parameterised by two environment variables so all four paper rows come from ONE script:

    DATASET = original | corrected
    SPLIT   = calendar | timesort

  DATASET=original  -> E1 cleaning: MachineLearningCVE CSVs, 78 features, no identifier drop,
                       no "Attempted" relabel (those don't exist in the original ML CSVs).
  DATASET=corrected -> E2 cleaning: WTMC2021 CSVs; relabel every "X - Attempted" -> BENIGN,
                       drop [Flow ID, Src IP, Src Port, Dst IP, Timestamp], KEEP Dst Port +
                       Protocol (Engelen's recipe).

  SPLIT=calendar  -> train = Monday+Tuesday+Wednesday, test = Thursday+Friday.
                     Purely filename-derived; IDENTICAL method for both datasets. This is the
                     primary, fully-robust E3 condition.
  SPLIT=timesort  -> global time order, first 80% train / last 20% test (Moczkodan's naive
                     version). Corrected: ordered by the real parsed Timestamp column.
                     Original: the ML CSVs carry NO timestamp, so order is approximated by
                     chronological FILE order (Mon, Tue, Wed, Thu-AM, Thu-PM, Fri-AM,
                     Fri-PM-PortScan, Fri-PM-DDoS) then within-file row order. This
                     approximation is documented in Methods; it does not affect the
                     degeneracy conclusion because attack classes are day-scheduled.

Day-of-week is taken from the SOURCE FILENAME (100% reliable) because the E1/E2 cached
pickles dropped file provenance and (for original) never had a timestamp.

Everything else is identical to E1/E2: drop exact-duplicate rows, LabelEncoder over the full
label set, median imputer + StandardScaler fit on TRAIN ONLY (no leakage), same 5 models,
SEED=42, same metrics. Per-class / confusion / support come from eval_metrics_e3.py.

Run:   DATASET=corrected SPLIT=calendar python src/run_e3_temporal.py
Outputs (tag = <dataset>_<split>):
   results/e3_<tag>_results.csv        one row per model (accuracy, macro/weighted F1)
   results/e3_<tag>_prepped.npz         cached split+scaled arrays for eval_metrics_e3.py
   results/label_classes_e3_<tag>.npy   class names for that condition
   results/cleaned_e3_<dataset>_<cachetag>.pkl   cleaned data (reused across both splits)
E1/E2 outputs are never touched.
"""
import glob, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
# Eq. (2) macro-F1 over C+. Do NOT replace with a bare f1_score(average="macro"):
# see the header of src/macro_f1.py for the bug that cost us the E3 column.
from macro_f1 import macro_f1_cplus, weighted_f1_cplus, audit_macro

# ---- experiment selection --------------------------------------------------
DATASET = os.environ.get("DATASET", "").strip().lower()
SPLIT   = os.environ.get("SPLIT", "").strip().lower()
if DATASET not in ("original", "corrected"):
    raise SystemExit("Set DATASET=original or DATASET=corrected")
if SPLIT not in ("calendar", "timesort"):
    raise SystemExit("Set SPLIT=calendar or SPLIT=timesort")

SEED = 42
HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Where the per-day CSVs live. If DATA_DIR is not set, auto-detect from known locations
# (the same folders E1/E2 used). Override anytime with DATA_DIR=/path/to/csvs.
_RESEARCH = os.path.expanduser(
    "~/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets")
_CANDIDATES = {
    "original": [
        os.path.join(HERE, "..", "data", "original"),
        os.path.join(_RESEARCH, "CICIDS2017_Original_ML_CSVs"),
    ],
    "corrected": [
        os.path.join(HERE, "..", "data", "corrected_wtmc2021"),
        os.path.join(_RESEARCH, "CICIDS2017_Corrected_WTMC2021"),
    ],
}[DATASET]


def _resolve_data_dir():
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    for c in _CANDIDATES:
        if glob.glob(os.path.join(c, "*.csv")):
            return c
    return _CANDIDATES[0]   # nothing found; return first so the error message is concrete


DATA_DIR = _resolve_data_dir()

MAX_ROWS = int(os.environ.get("MAX_ROWS", "0")) or None
CACHE_TAG = f"cap{MAX_ROWS}" if MAX_ROWS else "full"
TAG = f"{DATASET}_{SPLIT}"
CLEAN_PICKLE = os.path.join(OUT_DIR, f"cleaned_e3_{DATASET}_{CACHE_TAG}.pkl")
PREPPED_NPZ  = os.path.join(OUT_DIR, f"e3_{TAG}_{CACHE_TAG}.npz")
RESULTS_CSV  = os.path.join(OUT_DIR, f"e3_{TAG}_results.csv")
CLASSES_NPY  = os.path.join(OUT_DIR, f"label_classes_e3_{TAG}.npy")
MIN_EXPECTED_FULL_ROWS = 400_000
ONLY_MODELS = [m for m in os.environ.get("ONLY_MODELS", "").split(",") if m]
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))
STAGE = os.environ.get("STAGE", "")

LABEL_COL = "Label"
# E2 identifier drop (corrected only). Timestamp is popped separately (needed for timesort).
DROP_COLS_CORRECTED = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
TS_COL = "Timestamp"                       # corrected data only
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Explicit chronological ordering of the source files (for SPLIT=timesort ordering, and to
# make day assignment unambiguous). Matching is by substring of the lowercased filename.
CHRONO_ORDER = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday-workinghours-morning-webattacks",   # original: Thu AM
    "thursday-workinghours-afternoon-infilteration",  # original: Thu PM
    "thursday",                                   # corrected: whole Thursday
    "friday-workinghours-morning",                # original: Fri AM (Bot)
    "friday-workinghours-afternoon-portscan",     # original: Fri PM PortScan (~13:55)
    "friday-workinghours-afternoon-ddos",         # original: Fri PM DDoS (~15:56)
    "friday",                                     # corrected: whole Friday
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def day_index_of(fname):
    low = os.path.basename(fname).lower()
    for i, d in enumerate(DAYS):
        if d in low:
            return i
    raise ValueError(f"Cannot infer day from filename: {fname}")


def chrono_rank(fname):
    """Position of a file in the documented CICIDS2017 chronology (smaller = earlier).

    Match by the MOST SPECIFIC (longest) key that is a substring of the filename. This is
    essential: the generic keys 'thursday'/'friday' also match the fine-grained
    'friday-...-morning' etc., so we must pick the longest match, not the last one, or every
    Thursday/Friday file would collapse to a single rank and lose within-day time order.
    """
    low = os.path.basename(fname).lower().replace(".pcap_iscx", "").replace(".csv", "")
    best_rank, best_len = None, -1
    for rank, key in enumerate(CHRONO_ORDER):
        if key in low and len(key) > best_len:
            best_rank, best_len = rank, len(key)
    if best_rank is None:
        best_rank = day_index_of(fname)  # fallback: coarse day order
    return best_rank


def relabel_attempted(y):
    """E2 change 2: 'X - Attempted' -> BENIGN (Engelen default)."""
    return y.where(~y.str.endswith("- Attempted"), "BENIGN")


# ---- 1. LOAD + CLEAN (keep day_idx + a time-order key; drop them before training) ------
def load_and_clean():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=chrono_rank)
    log(f"[{DATASET}] {len(files)} CSVs in chronological order:")
    for f in files:
        log(f"    rank {chrono_rank(f)}  day {day_index_of(f)}  {os.path.basename(f)}")
    if not files:
        raise FileNotFoundError(
            f"No CSVs found for DATASET={DATASET}. Looked in DATA_DIR={DATA_DIR!r}. "
            f"Auto-detect candidates were: {_CANDIDATES}. "
            f"Set DATA_DIR=/path/to/csvs explicitly, e.g. "
            f"DATASET={DATASET} SPLIT={SPLIT} DATA_DIR=/full/path python src/run_e3_temporal.py")

    Xparts, yparts, dparts, oparts = [], [], [], []
    seq = 0
    n_attempted = 0
    for f in files:
        d_idx = day_index_of(f)
        f_rank = chrono_rank(f)
        for chunk in pd.read_csv(f, low_memory=False, chunksize=150_000):
            chunk.columns = chunk.columns.str.strip()
            y = chunk[LABEL_COL].astype(str).str.strip()

            # time-order key ---------------------------------------------------
            if DATASET == "corrected" and TS_COL in chunk.columns:
                ts = pd.to_datetime(chunk[TS_COL].astype(str).str.strip(),
                                    format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
                # int64 ns since epoch. Use the numpy array's .view (stable across pandas
                # versions) rather than Series.view, which pandas 3.0 removed. NaT -> int64 min.
                order_key = pd.Series(ts.to_numpy("datetime64[ns]").view("int64"),
                                      index=chunk.index)          # NaT -> min (handled below)
            else:
                # original ML CSVs: no timestamp -> approximate by (file rank, row order)
                order_key = f_rank * 10**12 + (seq + np.arange(len(chunk)))
                order_key = pd.Series(np.asarray(order_key, dtype="int64"),
                                      index=chunk.index)
            seq += len(chunk)

            # labels -----------------------------------------------------------
            if DATASET == "corrected":
                n_attempted += int(y.str.endswith("- Attempted").sum())
                y = relabel_attempted(y)

            # features (drop label + identifiers; Timestamp handled above then dropped) --
            drop = [LABEL_COL]
            if DATASET == "corrected":
                drop += [c for c in DROP_COLS_CORRECTED if c in chunk.columns]
                if TS_COL in chunk.columns:
                    drop += [TS_COL]
            Xp = (chunk.drop(columns=drop)
                       .apply(pd.to_numeric, errors="coerce")
                       .replace([np.inf, -np.inf], np.nan)
                       .astype(np.float32))
            Xparts.append(Xp)
            yparts.append(y.reset_index(drop=True))
            dparts.append(pd.Series(np.full(len(chunk), d_idx, dtype=np.int8)))
            oparts.append(order_key.reset_index(drop=True).astype("int64"))
            del chunk
        log(f"  loaded {os.path.basename(f)}  (rows so far: {seq:,})")

    X = pd.concat(Xparts, ignore_index=True)
    y = pd.concat(yparts, ignore_index=True)
    day = pd.concat(dparts, ignore_index=True)
    order = pd.concat(oparts, ignore_index=True)
    del Xparts, yparts, dparts, oparts
    # NaT timestamps -> push to front deterministically (rare); keeps int64 sortable
    order = order.replace(np.iinfo("int64").min, order[order != np.iinfo("int64").min].min())
    log(f"Concatenated: X={X.shape}  ({X.shape[1]} feature cols)")
    if DATASET == "corrected":
        log(f"Relabelled {n_attempted:,} '- Attempted' flows -> BENIGN")

    if MAX_ROWS and len(X) > MAX_ROWS:
        # keep a stratified subsample but PRESERVE day/order alignment via index.
        # (subsampling is a constrained-machine convenience; the full laptop run leaves
        #  MAX_ROWS unset and skips this entirely.) Fall back to a plain subsample if a
        #  class is too small to stratify.
        try:
            idx = train_test_split(np.arange(len(X)), train_size=MAX_ROWS,
                                   stratify=y, random_state=SEED)[0]
        except ValueError as e:
            log(f"stratified subsample failed ({e}); using non-stratified subsample")
            idx = train_test_split(np.arange(len(X)), train_size=MAX_ROWS,
                                   random_state=SEED)[0]
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        day = day.iloc[idx].reset_index(drop=True)
        order = order.iloc[idx].reset_index(drop=True)
        log(f"Subsampled to {len(X):,} rows (MAX_ROWS set)")

    # drop exact-duplicate feature+label rows (E1/E2 defect handling); keep first (earliest)
    before = len(X)
    key = X.copy()
    key["__label__"] = y.values
    keep_mask = ~key.duplicated()
    del key
    X = X[keep_mask].reset_index(drop=True)
    y = y[keep_mask].reset_index(drop=True)
    day = day[keep_mask].reset_index(drop=True)
    order = order[keep_mask].reset_index(drop=True)
    log(f"Dropped {before - len(X):,} duplicate rows -> {len(X):,} remain")

    out = X.copy()
    out["__label__"] = y.values
    out["__day__"] = day.values
    out["__order__"] = order.values
    return out


# ---- 2. TEMPORAL SPLIT + PREPROCESS (fit on TRAIN ONLY) --------------------
def temporal_masks(day, order):
    if SPLIT == "calendar":
        train_mask = day.isin([0, 1, 2]).values        # Mon, Tue, Wed
        test_mask = day.isin([3, 4]).values             # Thu, Fri
    else:  # timesort: global order, first 80% train / last 20% test
        n = len(order)
        cut = order.sort_values().iloc[int(0.8 * n)]
        train_mask = (order <= cut).values
        test_mask = (order > cut).values
    return train_mask, test_mask


def prepare(df):
    y_raw = df.pop("__label__")
    day = df.pop("__day__")
    order = df.pop("__order__")
    X = df

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)         # over the FULL label set (all classes get an index)
    np.save(CLASSES_NPY, le.classes_)
    log(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    tr, te = temporal_masks(day, order)
    Xtr, Xte = X[tr].values, X[te].values
    ytr, yte = y_enc[tr], y_enc[te]
    log(f"Split={SPLIT}: train={tr.sum():,}  test={te.sum():,}")

    # report class support so degeneracy is visible immediately
    import collections
    ctr = collections.Counter(le.classes_[i] for i in ytr)
    cte = collections.Counter(le.classes_[i] for i in yte)
    only_test = sorted(set(cte) - set(ctr))
    only_train = sorted(set(ctr) - set(cte))
    log(f"  classes with ZERO TRAIN support (unlearnable -> recall 0): {only_test}")
    log(f"  classes with ZERO TEST support (not evaluated): {only_train}")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    log(f"Train {Xtr.shape}  Test {Xte.shape}")
    return Xtr, Xte, ytr, yte, le.classes_


def _guard(n_rows):
    if not MAX_ROWS and n_rows < MIN_EXPECTED_FULL_ROWS:
        log("!" * 70)
        log(f"WARNING: full run but only {n_rows:,} rows (<{MIN_EXPECTED_FULL_ROWS:,}). "
            f"Looks like a subsample/bad cache. Delete results/*_{CACHE_TAG}.* and rerun.")
        log("!" * 70)


# ---- 3. TRAIN + EVALUATE ---------------------------------------------------
def run():
    t0 = time.time()
    log(f"E3  DATASET={DATASET}  SPLIT={SPLIT}  mode={'CAP '+str(MAX_ROWS) if MAX_ROWS else 'FULL'}")
    if STAGE != "prep" and os.path.exists(PREPPED_NPZ):
        d = np.load(PREPPED_NPZ, allow_pickle=True)
        Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
        log(f"Loaded prepped arrays  Train {Xtr.shape}  Test {Xte.shape}")
    else:
        if os.path.exists(CLEAN_PICKLE):
            log(f"Loading cached cleaned data {os.path.basename(CLEAN_PICKLE)}")
            df = pd.read_pickle(CLEAN_PICKLE)
        else:
            df = load_and_clean()
            df.to_pickle(CLEAN_PICKLE)
            log(f"Cached clean data -> {CLEAN_PICKLE}")
        Xtr, Xte, ytr, yte, classes = prepare(df)
        _guard(len(Xtr) + len(Xte))
        del df
        # Embed the class names IN the npz so eval can never read a mismatched class list.
        np.savez(PREPPED_NPZ, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, classes=classes)
        log(f"Cached prepped arrays -> {PREPPED_NPZ}")
        if STAGE == "prep":
            log(f"prep-only done in {round(time.time()-t0)}s")
            return

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if ONLY_MODELS:
        models = {k: v for k, v in models.items() if k in ONLY_MODELS}

    rows = []
    if os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0:
        try:
            rows = [r for r in pd.read_csv(RESULTS_CSV).to_dict("records")
                    if r["model"] not in models]
        except Exception:
            rows = []
    for name, clf in models.items():
        try:
            log(f"training {name} ...")
            ts = time.time()
            Xf, yf = Xtr, ytr
            if name == "KNN" and 0 < KNN_MAX_TRAIN < len(Xtr):
                try:
                    Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                                    stratify=ytr, random_state=SEED)
                except ValueError:
                    # a class with <2 members can't be stratified (rare classes under a
                    # temporal split); fall back to a plain subsample
                    Xf, _, yf, _ = train_test_split(Xtr, ytr, train_size=KNN_MAX_TRAIN,
                                                    random_state=SEED)
            clf.fit(Xf, yf)
            pred = clf.predict(Xte)
            mf1, n_cplus = macro_f1_cplus(yte, pred)
            row = {"dataset": DATASET, "split": SPLIT, "model": name,
                   "accuracy": accuracy_score(yte, pred),
                   "macro_f1": mf1,
                   "n_cplus": n_cplus,
                   "weighted_f1": weighted_f1_cplus(yte, pred),
                   "train_test_sec": round(time.time() - ts, 1)}
            # A macro-F1 on a degenerate split is meaningless without |C+|, so it
            # is logged and stored beside every number, not left to a side note.
            aud = audit_macro(yte, pred)
            if aud["diverges"]:
                log(f"  NOTE {name}: |C+|={aud['n_cplus']} but the sklearn default "
                    f"would average over {aud['n_union']} labels "
                    f"({aud['macro_f1_sklearn_default']:.4f} vs {mf1:.4f}). "
                    f"Reporting the Eq. (2) value.")
            log(f"{name:20s} acc={row['accuracy']:.4f}  macroF1={row['macro_f1']:.4f} "
                f"(|C+|={n_cplus})  weightedF1={row['weighted_f1']:.4f}  "
                f"({row['train_test_sec']}s)")
        except Exception as e:
            row = {"dataset": DATASET, "split": SPLIT, "model": name,
                   "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan,
                   "train_test_sec": np.nan, "error": str(e)[:200]}
            log(f"{name:20s} FAILED: {e}")
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    log(f"DONE in {round(time.time()-t0)}s -> {RESULTS_CSV}")


if __name__ == "__main__":
    run()
