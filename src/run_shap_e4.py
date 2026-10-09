"""
run_shap_e4.py — SHAP interpretability on the E4 leakage-free group split (task 3.1b, C4).

WHAT: TreeExplainer SHAP attributions for the best E4 model (RandomForest), computed on
corrected-data test flows, comparing the feature-importance *signature* of attack classes
that DEGRADE heavily under the leakage-free split vs. classes that stay STABLE.
Nobody has done this in the corrected-labels x leakage-free-split setting (novelty check
1.6/1.6c) — this is contribution C4.

DESIGN (honest-by-construction):
  * Reuses the EXACT E4 split. If the seed's npz arrays exist they are loaded; otherwise
    the split is rebuilt deterministically from the cleaned cache via run_e4_group.prepare()
    (same GroupShuffleSplit seed, same LabelEncoder over the full label set).
  * REPRODUCIBILITY GUARD: the retrained model's test predictions must EXACTLY match the
    predictions saved by the original E4 run (results/..._preds.npz). If they don't, the
    script aborts — we never explain a model different from the one the paper reports.
  * Heartbleed is auto-skipped: ZERO_TRAIN in all E4 seeds (its single conversation lands
    on one side), so the model literally never learned it — there is nothing to explain.
    Say this in the paper; do not hide it.
  * Micro-class caveat: degraded classes have single-digit test support (e.g. WA-SQL n=6).
    Their SHAP means are qualitative signatures, not population estimates. n is printed in
    every output and figure label.
  * feature_perturbation="tree_path_dependent" (SHAP default without background data) —
    fast and exact for trees; disclose in Methods.

DEFAULT SEED = 42 (not 2): Infiltration has ZERO test rows in seed 2, so the saved seed-2
arrays cannot explain one of the three degraded classes. Seed 42 has all focus classes in
test; the split is rebuilt (a few minutes) and verified against saved predictions.

Run (laptop, full):
    pip install shap
    SEED=42 python src/run_shap_e4.py
Smoke (slice; after a matching run_e4_group.py smoke run):
    CACHE_TAG=smoke6000 SEED=42 SAMPLES_PER_CLASS=50 python src/run_shap_e4.py

Env knobs:
    SEED               E4 seed to explain               (default 42)
    CACHE_TAG          full | smoke<N> | cap<N>          (default full)
    MODEL              RandomForest | DecisionTree       (default RandomForest)
    SAMPLES_PER_CLASS  max test flows explained per class (default 300)
    TOPK               features listed per class in topk CSV (default 10)
    DEGRADED / STABLE  comma-separated class-name overrides
    EXPLAIN_TRAIN=1    for a focus class with 0 test rows, fall back to TRAIN rows
                       (flagged '(train)' everywhere — attribution of what was learned,
                       NOT of generalisation; off by default)
    SHAP_FORCE=1       recompute classes even if a per-class cache file exists

ONE-RUN ROBUSTNESS: classes are explained smallest-first (degraded classes finish in the
first minutes), each class's result is cached to results/shap_cache_*/ immediately, and a
per-class failure is skipped, not fatal. If the run is interrupted, rerunning skips every
finished class (only the RF retrain is repeated).

Outputs (nothing from E1–E4 is touched):
    results/shap_e4_<model>_seed<seed>_meanabs.csv   full 78-feature mean|SHAP| per class
    results/shap_e4_<model>_seed<seed>_topk.csv      top-K features per class, ranked
    results/figures/fig3_shap_degraded_vs_stable.png  heatmap, degraded vs stable signatures
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ---- config -----------------------------------------------------------------
SEED = int(os.environ.get("SEED", "42"))
CACHE_TAG = os.environ.get("CACHE_TAG", "full")
MODEL = os.environ.get("MODEL", "RandomForest")
SAMPLES_PER_CLASS = int(os.environ.get("SAMPLES_PER_CLASS", "300"))
TOPK = int(os.environ.get("TOPK", "10"))
EXPLAIN_TRAIN = os.environ.get("EXPLAIN_TRAIN", "") == "1"
DATASET = "corrected"  # E4 Phase 1; original is Phase 2 and has no full E4 run to explain

DEGRADED = [s.strip() for s in os.environ.get(
    "DEGRADED",
    "Infiltration,Web Attack - Sql Injection,Web Attack - XSS").split(",") if s.strip()]
STABLE = [s.strip() for s in os.environ.get(
    "STABLE",
    "BENIGN,DoS Hulk,DDoS,PortScan,FTP-Patator,Web Attack - Brute Force").split(",") if s.strip()]

HERE = os.path.dirname(__file__)
OUT_DIR = os.environ.get("OUT_DIR") or os.path.join(HERE, "..", "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NPZ_PATH = os.path.join(OUT_DIR, f"e4_{DATASET}_group_seed{SEED}_{CACHE_TAG}.npz")
PREDS_PATH = os.path.join(OUT_DIR, f"e4_{DATASET}_group_seed{SEED}_{CACHE_TAG}_preds.npz")
CLEAN_PICKLE = os.path.join(OUT_DIR, f"cleaned_e4_{DATASET}_{CACHE_TAG}.pkl")
MEANABS_CSV = os.path.join(OUT_DIR, f"shap_e4_{MODEL}_seed{SEED}_meanabs.csv")
TOPK_CSV = os.path.join(OUT_DIR, f"shap_e4_{MODEL}_seed{SEED}_topk.csv")
FIG_PATH = os.path.join(FIG_DIR, "fig3_shap_degraded_vs_stable.png")

MODEL_FACTORY = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                   random_state=SEED),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=SEED),
}
if MODEL not in MODEL_FACTORY:
    raise SystemExit(f"MODEL must be one of {list(MODEL_FACTORY)} (TreeExplainer-compatible)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- 1. feature names + arrays ---------------------------------------------
def load_split():
    """Return Xtr, Xte, ytr, yte, classes, feat_names for SEED — from npz if saved,
    otherwise rebuilt deterministically from the cleaned cache via run_e4_group.prepare()."""
    if not os.path.exists(CLEAN_PICKLE):
        raise FileNotFoundError(
            f"{CLEAN_PICKLE} not found. Run run_e4_group.py first (same CACHE_TAG) — "
            f"it creates the cleaned cache this script needs for feature names/rebuild.")
    log(f"Loading cleaned cache {os.path.basename(CLEAN_PICKLE)} (feature names + rebuild source)")
    df = pd.read_pickle(CLEAN_PICKLE)
    from run_e4_group import GROUP_COL, LABEL_KEY, prepare  # deferred: module reads env on import
    feat_names = [c for c in df.columns if c not in (LABEL_KEY, GROUP_COL)]

    if os.path.exists(NPZ_PATH):
        log(f"Loading saved split arrays {os.path.basename(NPZ_PATH)}")
        d = np.load(NPZ_PATH, allow_pickle=True)
        Xtr, Xte, ytr, yte, classes = d["Xtr"], d["Xte"], d["ytr"], d["yte"], d["classes"]
    else:
        log(f"No saved npz for seed {SEED} — rebuilding split deterministically "
            f"(same GroupShuffleSplit + LabelEncoder as run_e4_group.py)")
        le = LabelEncoder().fit(df[LABEL_KEY].values)
        Xtr, Xte, ytr, yte = prepare(df, SEED, le)
        classes = le.classes_
    if Xtr.shape[1] != len(feat_names):
        raise AssertionError(f"feature-name mismatch: X has {Xtr.shape[1]} cols, "
                             f"cache has {len(feat_names)} names")
    return Xtr, Xte, ytr, yte, np.asarray(classes), feat_names


# ---- 2. retrain + reproducibility guard -------------------------------------
def train_and_verify(Xtr, ytr, Xte, yte):
    clf = MODEL_FACTORY[MODEL]()
    log(f"Training {MODEL} (seed {SEED}) on {len(Xtr):,} rows ...")
    t = time.time()
    clf.fit(Xtr, ytr)
    log(f"  trained in {time.time()-t:.0f}s")
    pred = clf.predict(Xte)

    if os.path.exists(PREDS_PATH):
        d = np.load(PREDS_PATH, allow_pickle=True)
        key = f"pred_{MODEL}"
        if key in d:
            if not np.array_equal(np.asarray(d["yte"]), np.asarray(yte)):
                raise AssertionError(
                    "yte mismatch vs saved preds npz — the rebuilt split is NOT the E4 split. "
                    "Do not proceed; check CACHE_TAG/SEED and the cleaned cache.")
            match = float(np.mean(pred == d[key]))
            log(f"REPRODUCIBILITY GUARD: prediction match vs saved E4 run = {match:.6f}")
            if match < 1.0:
                raise AssertionError(
                    f"Retrained {MODEL} predictions differ from the saved E4 run "
                    f"({match:.6f} match). Refusing to explain a different model. "
                    f"Check sklearn version / seed / arrays.")
        else:
            log(f"!! WARNING: {key} not in preds npz — cannot verify; results are unguarded.")
    else:
        log(f"!! WARNING: {os.path.basename(PREDS_PATH)} not found — cannot verify the model "
            f"against the original E4 run.")
    return clf, pred


# ---- 3. sample selection -----------------------------------------------------
def pick_samples(classes, yte, ytr):
    """Return list of (class_name, group, side, row_indices) for every focus class."""
    rng = np.random.default_rng(SEED)
    focus = [(c, "degraded") for c in DEGRADED] + [(c, "stable") for c in STABLE]
    name_to_idx = {c: i for i, c in enumerate(classes)}
    plan = []
    for cname, grp in focus:
        if cname not in name_to_idx:
            log(f"  skip '{cname}': not in label set")
            continue
        ci = name_to_idx[cname]
        if int(np.sum(ytr == ci)) == 0:
            # ZERO_TRAIN: the model has no output head for this class (it is absent from
            # clf.classes_) — there is literally nothing to explain, even if test rows exist.
            log(f"  skip '{cname}': ZERO_TRAIN — model never learned it (nothing to explain)")
            continue
        te_rows = np.flatnonzero(yte == ci)
        if len(te_rows) == 0:
            if EXPLAIN_TRAIN:
                tr_rows = np.flatnonzero(ytr == ci)
                take = tr_rows if len(tr_rows) <= SAMPLES_PER_CLASS else rng.choice(
                    tr_rows, SAMPLES_PER_CLASS, replace=False)
                log(f"  '{cname}': 0 test rows -> TRAIN fallback, n={len(take)} (flagged)")
                plan.append((cname, grp, "train", np.sort(take)))
            else:
                log(f"  skip '{cname}': 0 test rows this seed (set EXPLAIN_TRAIN=1 to use train rows)")
            continue
        take = te_rows if len(te_rows) <= SAMPLES_PER_CLASS else rng.choice(
            te_rows, SAMPLES_PER_CLASS, replace=False)
        plan.append((cname, grp, "test", np.sort(take)))
        log(f"  '{cname}' ({grp}): explaining n={len(take)} of {len(te_rows)} test rows")
    if not plan:
        raise SystemExit("No focus class has explainable rows — nothing to do.")
    return plan


# ---- 4. SHAP -----------------------------------------------------------------
def _safe_name(cname):
    return "".join(ch if ch.isalnum() else "_" for ch in cname)


def shap_meanabs(clf, Xte, Xtr, classes, plan):
    """One-run robustness (full laptop run):
       * classes processed SMALLEST-FIRST -> the degraded micro-classes (the contribution)
         finish in the first minutes; BENIGN (largest, least novel) goes last;
       * each class's result is CACHED to disk the moment it is computed -> a crash or
         Ctrl-C loses nothing, and a rerun skips finished classes (SHAP_FORCE=1 recomputes);
       * a per-class failure is logged and skipped, never fatal for the other classes."""
    import shap  # imported late so the script can at least print config without shap installed
    explainer = shap.TreeExplainer(clf)  # tree_path_dependent (no background) — exact for trees
    name_to_idx = {c: i for i, c in enumerate(classes)}
    # CRITICAL: SHAP output heads follow clf.classes_ (classes SEEN IN TRAINING), which is a
    # subset of the full label set when any class is ZERO_TRAIN (Heartbleed in the full run).
    # Indexing by full-label position would silently shift every class after the missing one.
    model_pos = {int(c): p for p, c in enumerate(clf.classes_)}

    cache_dir = os.path.join(OUT_DIR, f"shap_cache_{MODEL}_seed{SEED}_{CACHE_TAG}")
    os.makedirs(cache_dir, exist_ok=True)
    force = os.environ.get("SHAP_FORCE", "") == "1"

    plan = sorted(plan, key=lambda p: len(p[3]))  # smallest classes first
    log("Class order (smallest first, so degraded classes finish early): "
        + ", ".join(f"{c}(n={len(r)})" for c, _, _, r in plan))

    out, failures = {}, []
    for cname, grp, side, rows in plan:
        cache_f = os.path.join(cache_dir, f"{_safe_name(cname)}.npz")
        if os.path.exists(cache_f) and not force:
            d = np.load(cache_f, allow_pickle=True)
            out[cname] = (str(d["group"]), str(d["side"]), int(d["n"]), d["vec"])
            log(f"  SHAP '{cname}': loaded from cache (n={int(d['n'])}) — SHAP_FORCE=1 to redo")
            continue
        X = (Xte if side == "test" else Xtr)[rows]
        t = time.time()
        try:
            sv = explainer.shap_values(X, check_additivity=False)
            # shap returns list[n_outputs] of (n, f) in older versions; (n, f, n_outputs) newer.
            pos = model_pos[name_to_idx[cname]]  # position among the model's learned outputs
            if isinstance(sv, list):
                mat = sv[pos]                    # (n, f)
            else:
                sv = np.asarray(sv)
                mat = sv[:, :, pos] if sv.ndim == 3 else sv
            vec = np.abs(mat).mean(axis=0)
        except Exception as e:  # never let one class kill the run
            log(f"  !! SHAP FAILED for '{cname}' ({type(e).__name__}: {e}) — continuing")
            failures.append(cname)
            continue
        dt = time.time() - t
        out[cname] = (grp, side, len(rows), vec)
        np.savez(cache_f, vec=vec, n=len(rows), group=grp, side=side)
        log(f"  SHAP '{cname}': n={len(rows)}  ({dt:.0f}s, {dt/max(len(rows),1):.2f}s/sample) "
            f"-> cached")
    if failures:
        log(f"!! {len(failures)} class(es) failed SHAP and are missing from outputs: {failures}")
    if not out:
        raise SystemExit("SHAP produced nothing — see failures above.")
    return out


# ---- 5. outputs --------------------------------------------------------------
def write_outputs(result, feat_names):
    # full mean|SHAP| matrix: rows = features, one column per class
    cols = {}
    meta_rows = []
    for cname, (grp, side, n, vec) in result.items():
        tag = f"{cname}" + (" (train)" if side == "train" else "")
        cols[tag] = vec
        meta_rows.append({"class": cname, "group": grp, "side": side, "n_explained": n})
    full = pd.DataFrame(cols, index=feat_names)
    full.index.name = "feature"
    full.to_csv(MEANABS_CSV)

    topk_rows = []
    for cname, (grp, side, n, vec) in result.items():
        order = np.argsort(vec)[::-1][:TOPK]
        tot = vec.sum()
        for rank, fi in enumerate(order, 1):
            topk_rows.append({
                "class": cname, "group": grp, "side": side, "n_explained": n, "rank": rank,
                "feature": feat_names[fi], "mean_abs_shap": float(vec[fi]),
                "share_of_class_total": float(vec[fi] / tot) if tot > 0 else np.nan})
    pd.DataFrame(topk_rows).to_csv(TOPK_CSV, index=False)
    log(f"Wrote {MEANABS_CSV}")
    log(f"Wrote {TOPK_CSV}")
    return full, pd.DataFrame(meta_rows)


def make_figure(result, feat_names):
    """Heatmap: rows = focus classes (degraded group first), cols = union of each class's
    top-5 features; cell = share of that class's total |SHAP| (row-normalised, so classes
    with different output scales are comparable). Blues colormap = house style of the other
    figures; sequential light->dark, so it still survives IEEE grayscale printing.

    IEEE figure conventions applied (25 Jul): NO in-figure title (all of it belongs in the
    LaTeX caption — a title inside the image duplicates the caption and wastes plot area in
    a two-column layout), and NO dagger marker (a footnote symbol has to be decoded; right-
    edge group brackets labelled 'degraded'/'stable' are self-explanatory)."""
    ordered = ([(c, v) for c, v in result.items() if v[0] == "degraded"]
               + [(c, v) for c, v in result.items() if v[0] == "stable"])
    feat_union, seen = [], set()
    for _, (_, _, _, vec) in ordered:
        for fi in np.argsort(vec)[::-1][:5]:
            if fi not in seen:
                seen.add(fi)
                feat_union.append(fi)

    M = np.zeros((len(ordered), len(feat_union)))
    ylabels = []
    for r, (cname, (grp, side, n, vec)) in enumerate(ordered):
        tot = vec.sum()
        M[r] = [vec[fi] / tot if tot > 0 else 0 for fi in feat_union]
        ylabels.append(f"{cname}{' (train)' if side=='train' else ''} (n={n})")

    fig_w = max(8.0, 0.55 * len(feat_union) + 3.5)
    fig_h = max(3.5, 0.5 * len(ordered) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # PowerNorm(1.6): SHAP shares only reach ~0.13 (vs the confusion matrices' 0..1 range),
    # so a linear map renders everything mid-pale. Gamma>1 whitens the low background and
    # saturates the peaks -> same visual family as confusion_*.png. Colorbar ticks still
    # show TRUE values (the norm bends the color mapping, not the numbers).
    vmax = float(M.max()) if M.size else 1.0
    im = ax.imshow(M, aspect="auto", cmap="Blues", norm=PowerNorm(gamma=1.6, vmin=0, vmax=vmax))
    for r in range(M.shape[0]):
        for cix in range(M.shape[1]):
            v = M[r, cix]
            if v >= 0.05:  # annotate notable cells, confusion-matrix style
                ax.text(cix, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 * vmax else "black")
    ax.set_xticks(range(len(feat_union)))
    ax.set_xticklabels([feat_names[fi] for fi in feat_union], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ylabels, fontsize=9)
    n_deg = sum(1 for _, v in ordered if v[0] == "degraded")
    if n_deg:
        ax.axhline(n_deg - 0.5, color="black", lw=1.2)

    # group brackets on the right edge — self-explanatory, so no footnote marker is needed
    xr = len(feat_union) - 0.5

    def _bracket(y0, y1, text):
        off, tick = 0.35, 0.25
        ax.plot([xr + off, xr + off + tick, xr + off + tick, xr + off],
                [y0, y0, y1, y1], color="black", lw=1.0, clip_on=False)
        ax.text(xr + off + tick + 0.35, (y0 + y1) / 2, text, rotation=270,
                va="center", ha="left", fontsize=9, clip_on=False)

    if n_deg:
        _bracket(-0.42, n_deg - 0.58, "degraded")
    if len(ordered) > n_deg:
        _bracket(n_deg - 0.42, len(ordered) - 0.58, "stable")
    ax.set_xlim(-0.5, xr + 2.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.09)
    cbar.set_label("share of class total mean |SHAP|", fontsize=8)
    ax.set_xlabel("feature (union of each class's top-5 by mean |SHAP|)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    plt.close(fig)
    log(f"Wrote {FIG_PATH}")


# ---- main --------------------------------------------------------------------
def run():
    t0 = time.time()
    log(f"3.1b SHAP  MODEL={MODEL}  SEED={SEED}  CACHE_TAG={CACHE_TAG}  "
        f"SAMPLES_PER_CLASS={SAMPLES_PER_CLASS}")
    Xtr, Xte, ytr, yte, classes, feat_names = load_split()
    clf, _ = train_and_verify(Xtr, ytr, Xte, yte)
    plan = pick_samples(classes, yte, ytr)
    result = shap_meanabs(clf, Xte, Xtr, classes, plan)
    write_outputs(result, feat_names)
    make_figure(result, feat_names)
    log(f"DONE in {round(time.time()-t0)}s")


if __name__ == "__main__":
    run()
