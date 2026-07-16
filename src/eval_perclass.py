"""
2.2 - Per-class metrics for E1.

Opens the macro-F1 average back into per-class precision / recall / F1 so we can see
WHICH of the 15 classes each model detects or misses. Retrains one model on the cached
prepped arrays, prints the report, and saves it to CSV (never retype numbers).

Run:   python src/eval_perclass.py
       MODEL=DecisionTree python src/eval_perclass.py    # pick another model
Output: results/e1_perclass_<MODEL>.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report

SEED = 42
HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, "..", "results", "prepped_arrays.npz")

CLASSES = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
           'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
           'SSH-Patator', 'Web Attack Brute Force', 'Web Attack Sql Injection',
           'Web Attack XSS']

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVM_SGD": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=SEED),
}

name = os.environ.get("MODEL", "RandomForest")
clf = MODELS[name]

d = np.load(NPZ, allow_pickle=True)
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
print(f"Training {name} on {Xtr.shape[0]:,} rows ...", flush=True)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

# labels=range(15) ensures every class shows even if a rare one is absent from a split.
present = np.unique(np.concatenate([ytr, yte]))
names = [CLASSES[i] for i in present]

print("\n" + classification_report(yte, pred, labels=present,
                                    target_names=names, digits=3, zero_division=0))

# Save the same numbers to CSV so nothing is ever retyped.
rep = classification_report(yte, pred, labels=present, target_names=names,
                            digits=3, zero_division=0, output_dict=True)
out = os.path.join(HERE, "..", "results", f"e1_perclass_{name}.csv")
pd.DataFrame(rep).T.to_csv(out)
print(f"\nSaved -> {out}")
