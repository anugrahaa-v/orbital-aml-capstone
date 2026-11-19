"""Classification experiment. Labels come from CelesTrak group membership (DEBRIS vs PAYLOAD): a defensible
but NOT ground-truth label. The task therefore measures how separable catalogued fragmentation-debris groups
are from active payloads by orbit/drag features, not 'true' object type detection."""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
                             accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import chronological_split


def classifiers(seed=42):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=40, random_state=seed, n_jobs=1, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "svc_rbf": SVC(probability=True, class_weight="balanced", random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=seed),
    }


def run_classification(df: pd.DataFrame, features: list[str], seed: int = 42, time_col: str = "epoch",
                       positive="DEBRIS", negative="PAYLOAD", max_rows=2000) -> dict:
    warnings.filterwarnings("ignore")
    d = df[df["obj_class"].isin([positive, negative])].dropna(subset=features).copy()
    if d["obj_class"].nunique() < 2:
        return {"error": "need both DEBRIS and PAYLOAD objects; ingest at least one debris group and 'active'"}
    if len(d) > max_rows:
        d = d.sort_values(time_col).iloc[np.linspace(0, len(d) - 1, max_rows).astype(int)]
    d["y"] = (d["obj_class"] == positive).astype(int)
    tr, va, te = chronological_split(d, time_col, (0.7, 0.1, 0.2))
    trva = pd.concat([tr, va])
    if trva["y"].nunique() < 2 or te["y"].nunique() < 2:
        return {"error": "a chronological split left only one class in train or test; ingest more data",
                "class_counts": d["y"].value_counts().to_dict()}
    out = {"features": features, "seed": seed, "label_note": "labels = CelesTrak group membership (not ground truth)",
           "n_train": len(trva), "n_test": len(te), "positive_class": positive,
           "test_class_counts": te["y"].value_counts().to_dict(), "models": []}
    Xtr, ytr, Xte, yte = trva[features].to_numpy(), trva["y"].to_numpy(), te[features].to_numpy(), te["y"].to_numpy()
    for name, clf in classifiers(seed).items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)]).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, proba)
        step = max(1, len(fpr) // 200)
        out["models"].append({
            "model": name, "accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0)), "roc_auc": float(roc_auc_score(yte, proba)),
            "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
            "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}})
    out["models"].sort(key=lambda m: -m["roc_auc"])
    out["confusion_labels"] = [negative, positive]
    return out



