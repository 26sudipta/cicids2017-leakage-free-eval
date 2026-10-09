"""
E2 - Same pipeline as E1, on the CORRECTED CICIDS2017 (Engelen et al. 2021, WTMC2021).

This is the controlled comparison for the paper: E1 ran on the ORIGINAL public CICIDS2017;
E2 runs the IDENTICAL modelling pipeline on Engelen's corrected release. The ONLY things that
differ from run_e1_baseline.py are the two data-side decisions the corrected dataset forces
(both settled by Engelen's own paper + code — see Literature/Engelen_2021_E2_Implications_Summary.md):

  CHANGE 1 (identifier columns). The corrected CSVs carry host-identifier columns that the
    original ML CSVs never had. Per Engelen §III-D (shortcut learning) and their
    MakeDataNumpyFriendly.py, we DROP: Flow ID, Src IP, Src Port, Dst IP, Timestamp.
    We KEEP: Dst Port (E1 kept Destination Port too) + Protocol (Engelen keeps it) + flow
    features. Running E1 unchanged here would feed IPs/timestamp in as junk and let ports/IPs
    leak attack identity — leakage that was NOT in E1, which would invalidate the comparison.

  CHANGE 2 ("- Attempted" labels). Engelen split many attacks into successful vs "X - Attempted"
    (flows with no real attack payload — empty/failed connections). Both their paper (§IV-B) and
    their code default relabel ALL "X - Attempted" flows to BENIGN, because a per-flow detector
    genuinely cannot distinguish an empty/failed connection from benign traffic. We do the same.
    The flows STAY in the dataset (honest evaluation — the hard cases are not deleted), they just
    get the label the model can actually justify. Result: ~15 classes, comparable to Engelen's
    published Table II.

EVERYTHING ELSE IS IDENTICAL TO E1: drop exact-duplicate rows, LabelEncoder, stratified 70/30
split (E1's ratio; Engelen used 75/25 — kept at 70/30 for E1<->E2 consistency, noted in Methods),
median imputer + StandardScaler fit on TRAIN ONLY (no leakage), same 5 models, same SEED=42, same
metrics (accuracy, macro-F1, weighted-F1). Per-class / confusion / FPR come from eval_metrics.py
exactly as for E1.

Run:  python src/run_e2_corrected.py
Outputs: results/e2_corrected_results.csv   (one row per model)
         results/cleaned_corrected.pkl      (cached clean data for reuse)
         results/e2_prepped_arrays.npz       (cached split+scaled arrays)
E1 outputs are never overwritten (all filenames are e2_* / *_corrected).
"""
import glob, os, time, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
# Eq. (2) macro-F1 over C+. Harmless here (a stratified split puts all 15 classes
# in test, so C+ == the full label set) but pinned so the definition is identical
# across E1/E2/E3/E4. See src/macro_f1.py.
from macro_f1 import macro_f1_cplus, weighted_f1_cplus

# ---- paths -----------------------------------------------------------------
# Folder holding the 5 corrected WTMC2021 day-CSVs. Repo-relative default so it is portable.
# Override with:  DATA_DIR=/path/to/corrected_csvs python src/run_e2_corrected.py
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "corrected_wtmc2021"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "e2_corrected_results.csv")

# CHANGE 1: host-identifier columns to drop (Engelen's recipe). Dst Port + Protocol are KEPT.
DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
LABEL_COL = "Label"

# Optional cap for constrained machines: MAX_ROWS=800000 python src/run_e2_corrected.py
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0")) or None

# Caches are TAGGED by the row setting so a FULL run can never silently reuse a capped
# (subsample) cache left over from a smoke test. A full run looks for *_full.* only.
CACHE_TAG = f"cap{MAX_ROWS}" if MAX_ROWS else "full"
CLEAN_PICKLE = os.path.join(OUT_DIR, f"cleaned_corrected_{CACHE_TAG}.pkl")
# Sanity floor: a genuine full corrected-data run has ~2M+ rows after dedup. If a "full"
# run ends up with far fewer, something is wrong (bad cache / wrong DATA_DIR) -> warn loudly.
MIN_EXPECTED_FULL_ROWS = 1_000_000
SEED = 42
STAGE = os.environ.get("STAGE", "")
ONLY_MODELS = [m for m in os.environ.get("ONLY_MODELS", "").split(",") if m]
# KNN is a lazy learner (predict cost ~ train x test); fit it on a stratified reference
# subsample, exactly as in E1, so the comparison is apples-to-apples.
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def relabel_attempted(y):
    """CHANGE 2: relabel every 'X - Attempted' flow to BENIGN (Engelen default).

    The corrected labels look like 'Bot - Attempted', 'Web Attack - Brute Force - Attempted',
    'FTP-Patator - Attempted', etc. All end in '- Attempted'. PortScan has no attempted flows.
    """
    return y.where(~y.str.endswith("- Attempted"), "BENIGN")


# ---- 1. LOAD + CLEAN -------------------------------------------------------
def load_and_clean():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    log(f"Found {len(files)} CSVs in {DATA_DIR}")
    if not files:
        raise FileNotFoundError(
            f"No CSVs in DATA_DIR={DATA_DIR!r}. Point DATA_DIR at the folder holding the 5 "
            f"corrected WTMC2021 CICIDS2017 .csv files.")

    # Chunked reading + per-chunk float32 downcast bounds peak RAM (same fix as E1).
    Xparts, yparts = [], []
    n_attempted = 0
    for f in files:
        for chunk in pd.read_csv(f, low_memory=False, chunksize=150_000):
            chunk.columns = chunk.columns.str.strip()
            y = chunk[LABEL_COL].astype(str).str.strip()
            n_attempted += int(y.str.endswith("- Attempted").sum())
            y = relabel_attempted(y)                       # CHANGE 2
            drop = [c for c in DROP_COLS if c in chunk.columns] + [LABEL_COL]  # CHANGE 1
            Xp = (chunk.drop(columns=drop)
                       .apply(pd.to_numeric, errors="coerce")
                       .replace([np.inf, -np.inf], np.nan)
                       .astype(np.float32))
            Xparts.append(Xp); yparts.append(y)
            del chunk
        log(f"  loaded {os.path.basename(f)}")
    X = pd.concat(Xparts, ignore_index=True)
    y = pd.concat(yparts, ignore_index=True)
    del Xparts, yparts
    log(f"Concatenated: {X.shape}  (kept {X.shape[1]} feature cols incl. Dst Port + Protocol)")
    log(f"Relabelled {n_attempted:,} '- Attempted' flows -> BENIGN (Engelen default)")

    if MAX_ROWS and len(X) > MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS,
                                      stratify=y, random_state=SEED)
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        log(f"Subsampled to {len(X):,} rows (MAX_ROWS set)")

    # Drop exact-duplicate rows (same defect handling as E1).
    before = len(X)
    X["__label__"] = y.values
    X = X.drop_duplicates().reset_index(drop=True)
    y = X.pop("__label__").reset_index(drop=True)
    log(f"Dropped {before - len(X):,} duplicate rows -> {len(X):,} remain")

    return X, y


# ---- 2. SPLIT + PREPROCESS (fit on TRAIN ONLY = no leakage) ----------------
def prepare(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    log(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_enc, test_size=0.30, stratify=y_enc, random_state=SEED)
    except ValueError as e:
        log(f"stratified split failed ({e}); using non-stratified split")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_enc, test_size=0.30, random_state=SEED)

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    log(f"Train {Xtr.shape}  Test {Xte.shape}")
    # Persist the label classes for eval_metrics.py (mirrors E1's label_classes.npy).
    np.save(os.path.join(OUT_DIR, "label_classes_e2.npy"), le.classes_)
    return Xtr, Xte, ytr, yte, le


# ---- 3. TRAIN + EVALUATE ---------------------------------------------------
PREPPED_NPZ = os.path.join(OUT_DIR, f"e2_prepped_arrays_{CACHE_TAG}.npz")


def _rowcount_guard(n_rows):
    """Loudly flag if a FULL run somehow trained on too few rows (stale/bad cache)."""
    if not MAX_ROWS and n_rows < MIN_EXPECTED_FULL_ROWS:
        log("!" * 70)
        log(f"WARNING: full run but only {n_rows:,} rows (<{MIN_EXPECTED_FULL_ROWS:,}). "
            f"This looks like a SUBSAMPLE, not the full corrected dataset. "
            f"Delete results/*_{CACHE_TAG}.* caches and rerun. Results are NOT valid.")
        log("!" * 70)


def run():
    t0 = time.time()
    log(f"Run mode: {'CAPPED MAX_ROWS='+str(MAX_ROWS) if MAX_ROWS else 'FULL dataset'}  "
        f"(cache tag: {CACHE_TAG})")
    if STAGE != "prep" and os.path.exists(PREPPED_NPZ):
        d = np.load(PREPPED_NPZ, allow_pickle=True)
        Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
        log(f"Loaded prepped arrays  Train {Xtr.shape}  Test {Xte.shape}")
        _rowcount_guard(Xtr.shape[0] + Xte.shape[0])
    else:
        if os.path.exists(CLEAN_PICKLE):
            log("Loading cached cleaned data")
            cached = pd.read_pickle(CLEAN_PICKLE)
            y = cached["Label"]; X = cached.drop(columns=["Label"])
        else:
            X, y = load_and_clean()
            out = X.copy(); out["Label"] = y
            out.to_pickle(CLEAN_PICKLE)
            log(f"Cached clean data -> {CLEAN_PICKLE}")
            del out
        Xtr, Xte, ytr, yte, le = prepare(X, y)
        _rowcount_guard(Xtr.shape[0] + Xte.shape[0])
        del X, y
        np.savez(PREPPED_NPZ, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
        log(f"Cached prepped arrays -> {PREPPED_NPZ}")
        if STAGE == "prep":
            log(f"prep-only done in {round(time.time()-t0)}s")
            return

    # Identical model set / hyperparameters to E1.
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                               random_state=SEED),
        "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3,
                                       random_state=SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if ONLY_MODELS:
        models = {k: v for k, v in models.items() if k in ONLY_MODELS}

    rows = []
    if os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0:
        try:
            rows = pd.read_csv(RESULTS_CSV).to_dict("records")
            rows = [r for r in rows if r["model"] not in models]
        except Exception:
            rows = []
    for name, clf in models.items():
        try:
            log(f"training {name} ...")
            ts = time.time()
            Xf, yf = Xtr, ytr
            if name == "KNN" and 0 < KNN_MAX_TRAIN < len(Xtr):
                Xf, _, yf, _ = train_test_split(
                    Xtr, ytr, train_size=KNN_MAX_TRAIN, stratify=ytr,
                    random_state=SEED)
                log(f"  KNN reference set capped to {len(Xf):,} rows")
            clf.fit(Xf, yf)
            pred = clf.predict(Xte)
            acc = accuracy_score(yte, pred)
            mf1, n_cplus = macro_f1_cplus(yte, pred)
            wf1 = weighted_f1_cplus(yte, pred)
            row = {"model": name, "accuracy": acc, "macro_f1": mf1,
                   "n_cplus": n_cplus,
                   "weighted_f1": wf1, "train_test_sec": round(time.time() - ts, 1)}
            log(f"{name:20s} acc={acc:.4f}  macroF1={mf1:.4f} (|C+|={n_cplus})  "
                f"weightedF1={wf1:.4f}  ({row['train_test_sec']}s)")
        except Exception as e:
            row = {"model": name, "accuracy": np.nan, "macro_f1": np.nan,
                   "weighted_f1": np.nan, "train_test_sec": np.nan,
                   "error": str(e)[:200]}
            log(f"{name:20s} FAILED: {e}")
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    log(f"DONE in {round(time.time()-t0)}s -> {RESULTS_CSV}")


if __name__ == "__main__":
    run()
