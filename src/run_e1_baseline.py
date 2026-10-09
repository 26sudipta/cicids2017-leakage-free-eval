"""
E1 - Baseline reproduction on ORIGINAL CICIDS2017 (MachineLearningCVE, 78 features).

Goal: reproduce the ~98% random-split accuracy that the base paper (and dozens of
others) report, so that later experiments (E2 corrected data, E3/E4 temporal /
leakage-free splits) can show how much of that number is an artifact.

Every preprocessing decision below is annotated with WHY, because the whole paper
is an argument about methodology and each step must be defensible to a reviewer.

Run:  python src/run_e1_baseline.py
Outputs: results/e1_baseline_results.csv   (one row per model)
         results/cleaned_original.parquet  (cached clean data for E2/E3 reuse)
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
# Folder holding the 8 original CICIDS2017 MachineLearningCVE CSVs.
# Default is <repo>/data/original (relative to this file, so it's portable).
# Override anytime with:  DATA_DIR=/path/to/csvs python src/run_e1_baseline.py
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "original"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "e1_baseline_results.csv")
CLEAN_PARQUET = os.path.join(OUT_DIR, "cleaned_original.pkl")

# Optional cap for constrained machines: MAX_ROWS=800000 python src/run_e1_baseline.py
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0")) or None
SEED = 42
# STAGE=prep -> only build the cleaned-data cache and exit.
# ONLY_MODELS=RandomForest,KNN -> train just those (so each fits one short call).
STAGE = os.environ.get("STAGE", "")
ONLY_MODELS = [m for m in os.environ.get("ONLY_MODELS", "").split(",") if m]
# KNN is a lazy learner: predict cost scales with (train x test). Fitting it on
# the full 1.75M-row train set is intractable, so we fit KNN on a stratified
# reference subsample. Other models still use the full training set.
KNN_MAX_TRAIN = int(os.environ.get("KNN_MAX_TRAIN", "150000"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- 1. LOAD + CLEAN -------------------------------------------------------
def load_and_clean():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    log(f"Found {len(files)} CSVs")
    if not files:
        raise FileNotFoundError(
            f"No CSVs in DATA_DIR={DATA_DIR!r}. Point DATA_DIR at the folder "
            f"holding the 8 original CICIDS2017 MachineLearningCVE .csv files, e.g. "
            f"DATA_DIR=/path/to/csvs python src/run_e1_baseline.py")
    # WHY chunked reading: pandas' CSV parser peaks ~1.2 GB on ONE of these wide
    # files as float64. Reading in row-chunks and downcasting each chunk to
    # float32 immediately bounds peak RAM to ~one chunk, so total size no longer
    # dictates memory. This is the fix for the earlier OOM.
    Xparts, yparts = [], []
    for f in files:
        for chunk in pd.read_csv(f, low_memory=False, chunksize=150_000):
            chunk.columns = chunk.columns.str.strip()
            yp = chunk["Label"].astype(str).str.strip()
            Xp = (chunk.drop(columns=["Label"])
                       .apply(pd.to_numeric, errors="coerce")
                       .replace([np.inf, -np.inf], np.nan)
                       .astype(np.float32))
            Xparts.append(Xp); yparts.append(yp)
            del chunk
        log(f"  loaded {os.path.basename(f)}")
    X = pd.concat(Xparts, ignore_index=True)
    y = pd.concat(yparts, ignore_index=True)
    del Xparts, yparts
    log(f"Concatenated: {X.shape}")

    # WHY subsample-before-dedup ONLY when MAX_ROWS set: dedup on 2.8M rows needs
    # ~2.6 GB (frame copies) and OOMs a 3.8 GB box. Capping first keeps the sandbox
    # run bounded. On a full-RAM machine MAX_ROWS is unset and we dedup everything
    # first (the scientifically correct order).
    if MAX_ROWS and len(X) > MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS,
                                      stratify=y, random_state=SEED)
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        log(f"Subsampled to {len(X):,} rows (MAX_ROWS set)")

    # WHY: CICIDS2017 contains many exact-duplicate flow rows (documented defect);
    # base paper drops them. In-place label add avoids an extra full-frame copy.
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

    # WHY stratify: preserves the rare-attack proportions in both splits, so tiny
    # classes (e.g. Heartbleed) don't vanish entirely from the test set. If a class
    # is too small to stratify (can happen after subsampling on the sandbox), fall
    # back to a plain split rather than crash.
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_enc, test_size=0.30, stratify=y_enc, random_state=SEED)
    except ValueError as e:
        log(f"stratified split failed ({e}); using non-stratified split")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_enc, test_size=0.30, random_state=SEED)

    # WHY fit imputer+scaler on TRAIN only: fitting on all data leaks test-set
    # statistics into training. The base paper cleaned globally; we do it right.
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    log(f"Train {Xtr.shape}  Test {Xte.shape}")
    return Xtr, Xte, ytr, yte, le


# ---- 3. TRAIN + EVALUATE ---------------------------------------------------
PREPPED_NPZ = os.path.join(OUT_DIR, "prepped_arrays.npz")


def run():
    t0 = time.time()
    # Fast path for per-model calls: load the pre-split, pre-scaled arrays and go
    # straight to training. This skips the ~25s load+clean and ~7s prepare so the
    # whole model call fits inside one short sandbox window.
    if STAGE != "prep" and os.path.exists(PREPPED_NPZ):
        d = np.load(PREPPED_NPZ, allow_pickle=True)
        Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
        log(f"Loaded prepped arrays  Train {Xtr.shape}  Test {Xte.shape}")
    else:
        if os.path.exists(CLEAN_PARQUET):
            log("Loading cached cleaned data")
            cached = pd.read_pickle(CLEAN_PARQUET)
            y = cached["Label"]; X = cached.drop(columns=["Label"])
        else:
            X, y = load_and_clean()
            out = X.copy(); out["Label"] = y
            out.to_pickle(CLEAN_PARQUET)
            log(f"Cached clean data -> {CLEAN_PARQUET}")
            del out
        Xtr, Xte, ytr, yte, le = prepare(X, y)
        del X, y
        # Cache the ready-to-train arrays for subsequent one-model-per-call runs.
        np.savez(PREPPED_NPZ, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
        log(f"Cached prepped arrays -> {PREPPED_NPZ}")
        if STAGE == "prep":
            log(f"prep-only done in {round(time.time()-t0)}s")
            return

    # Cheapest/most-robust models first so partial results survive if a later,
    # heavier model is killed on a tiny box.
    models = {
        # n_jobs=-1 uses all CPU cores (big speedup for RF and KNN on a real
        # machine). It was set to 1 only to survive a tiny 3.8 GB sandbox; on a
        # normal laptop -1 is the right choice.
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                               random_state=SEED),
        # LinearSVM via SGD: liblinear (LinearSVC) is single-threaded and very slow
        # for 15-class OvR here. SGDClassifier(loss="hinge") IS a linear SVM trained
        # by SGD -- same decision boundary family, converges in seconds. On a laptop
        # you can use LinearSVC(dual=False) instead if you prefer the exact solver.
        "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3,
                                       random_state=SEED),
        # max_iter=1000 so lbfgs actually converges (100 was a sandbox shortcut and
        # triggered the ConvergenceWarning). n_jobs dropped: LR ignores it in sklearn 1.8+.
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if ONLY_MODELS:
        models = {k: v for k, v in models.items() if k in ONLY_MODELS}

    # Append mode: keep results from prior per-model calls, replace this model's row.
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
            # KNN: fit on a stratified reference subsample so prediction stays
            # tractable. Everything else fits on the full training set.
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
        except Exception as e:  # keep going if one model OOMs/fails
            row = {"model": name, "accuracy": np.nan, "macro_f1": np.nan,
                   "weighted_f1": np.nan, "train_test_sec": np.nan,
                   "error": str(e)[:200]}
            log(f"{name:20s} FAILED: {e}")
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)  # write incrementally

    log(f"DONE in {round(time.time()-t0)}s -> {RESULTS_CSV}")


if __name__ == "__main__":
    run()
