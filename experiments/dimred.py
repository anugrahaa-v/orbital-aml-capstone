"""PCA / LDA / ICA with downstream evaluation (reducer fit on train only)."""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .common import chronological_split


def run_dimred(df: pd.DataFrame, features: list[str], label_col="obj_class", time_col="epoch", seed=42,
               n_components=2, max_rows=2000) -> dict:
    warnings.filterwarnings("ignore")
    d = df.dropna(subset=features + [label_col]).copy()
    if len(d) > max_rows:
        d = d.sort_values(time_col).iloc[np.linspace(0, len(d) - 1, max_rows).astype(int)]
    classes = sorted(d[label_col].unique())
    tr, _, te = chronological_split(d, time_col, (0.8, 0.0, 0.2))
    sc = StandardScaler().fit(tr[features]); Xtr, Xte = sc.transform(tr[features]), sc.transform(te[features])
    ytr, yte = tr[label_col].to_numpy(), te[label_col].to_numpy()
    out = {"classes": classes, "features": features, "methods": {}, "n_train": len(tr), "n_test": len(te),
           "lda_note": None}
    def evaluate(name, red, supervised):
        Ztr = red.fit_transform(Xtr, ytr) if supervised else red.fit_transform(Xtr)
        Zte = red.transform(Xte)
        knn = KNeighborsClassifier(n_neighbors=15).fit(Ztr, ytr); pred = knn.predict(Zte)
        r = {"n_components": int(Ztr.shape[1]), "knn_accuracy": float(accuracy_score(yte, pred)),
             "knn_macro_f1": float(f1_score(yte, pred, average="macro")),
             "sil_by_class_test": float(silhouette_score(Zte, yte)) if len(set(yte)) > 1 and len(Zte) > 5 else None}
        if hasattr(red, "explained_variance_ratio_"):
            r["explained_variance_ratio"] = red.explained_variance_ratio_.tolist()
        pts = np.linspace(0, len(Zte) - 1, min(1500, len(Zte))).astype(int)
        zz = Zte[pts]
        if zz.shape[1] == 1:
            zz = np.hstack([zz, np.zeros((len(zz), 1))])
        r["projection"] = {"x": zz[:, 0].tolist(), "y": zz[:, 1].tolist(), "label": yte[pts].tolist()}
        out["methods"][name] = r
    evaluate("PCA", PCA(n_components=min(n_components, len(features)), random_state=seed), False)
    evaluate("ICA", FastICA(n_components=min(n_components, len(features)), random_state=seed, max_iter=300), False)
    k_lda = min(len(classes) - 1, len(features), n_components)
    if k_lda >= 1 and len(set(ytr)) >= 2:
        evaluate("LDA", LinearDiscriminantAnalysis(n_components=k_lda), True)
        if k_lda == 1:
            out["lda_note"] = ("Only 2 classes present => LDA yields 1 discriminant axis (max classes-1). "
                               "Labels are group-derived, not ground truth.")
    else:
        out["lda_note"] = "LDA skipped: needs >= 2 labelled classes."
    return out


