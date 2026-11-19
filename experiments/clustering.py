"""K-Means (elbow, silhouette, optimum K), hierarchical (Ward/complete/average, dendrogram), validation metrics."""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["inclination", "eccentricity", "perigee_alt", "apogee_alt"]


def _prep(df, features, max_rows, seed):
    d = df.dropna(subset=features).copy()
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    X = StandardScaler().fit_transform(d[features].to_numpy())
    return d.reset_index(drop=True), X


def metrics(X, labels, inertia=None, seed=42) -> dict:
    n_lab = len(set(labels))
    if n_lab < 2 or n_lab >= len(X):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, "inertia": inertia}
    ss = min(500, len(X))
    return {"silhouette": float(silhouette_score(X, labels, sample_size=ss, random_state=seed)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)), "inertia": inertia}


def run_kmeans(df, features=None, k_min=2, k_max=10, seed=42, max_rows=3000) -> dict:
    warnings.filterwarnings("ignore")
    features = features or CLUSTER_FEATURES
    d, X = _prep(df, features, max_rows, seed)
    if len(d) < k_max + 5:
        return {"error": f"too few objects ({len(d)}) for K up to {k_max}"}
    sweep = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
        sweep.append({"k": k, **metrics(X, km.labels_, float(km.inertia_), seed)})
    best_k = max(sweep, key=lambda s: s["silhouette"] if s["silhouette"] is not None else -1)["k"]
    # elbow: point of max distance from the chord joining first and last inertia values
    ks = np.array([s["k"] for s in sweep], float); inn = np.array([s["inertia"] for s in sweep])
    kx = (ks - ks.min()) / (ks.max() - ks.min()); iy = (inn - inn.min()) / max(1e-12, inn.max() - inn.min())
    chord = np.array([kx[-1] - kx[0], iy[-1] - iy[0]]); chord /= np.linalg.norm(chord)
    dist = [abs(chord[0] * (iy[j] - iy[0]) - chord[1] * (kx[j] - kx[0])) for j in range(len(ks))]
    elbow_k = int(ks[int(np.argmax(dist))])
    km = KMeans(n_clusters=best_k, n_init=3, random_state=seed).fit(X)
    pca3 = PCA(n_components=min(3, len(features)), random_state=seed).fit(X)
    Z = pca3.transform(X)
    if Z.shape[1] < 3:
        Z = np.hstack([Z, np.zeros((len(Z), 3 - Z.shape[1]))])
    idx = np.linspace(0, len(d) - 1, min(2500, len(d))).astype(int)
    cross = pd.crosstab(km.labels_, d["obj_class"]).to_dict() if "obj_class" in d else {}
    return {"features": features, "n_objects": len(d), "sweep": sweep, "optimal_k_silhouette": int(best_k),
            "elbow_k": elbow_k, "metrics_at_optimum": next(s for s in sweep if s["k"] == best_k),
            "pca_explained_variance": pca3.explained_variance_ratio_.tolist(),
            "points": {"x": Z[idx, 0].tolist(), "y": Z[idx, 1].tolist(), "z": Z[idx, 2].tolist(),
                       "cluster": km.labels_[idx].tolist(),
                       "name": d["name"].to_numpy()[idx].tolist() if "name" in d else []},
            "cluster_vs_class": {str(k): v for k, v in cross.items()},
            "note": "Clusters are unsupervised; class composition is shown only for interpretation."}


def run_hierarchical(df, features=None, n_clusters=None, seed=42, max_rows=600, dendro_points=150) -> dict:
    warnings.filterwarnings("ignore")
    features = features or CLUSTER_FEATURES
    d, X = _prep(df, features, max_rows, seed)
    if len(d) < 20:
        return {"error": f"too few objects ({len(d)})"}
    k = n_clusters or 4
    out = {"features": features, "n_objects": len(d), "n_clusters": k, "linkages": {}}
    for link in ("ward", "complete", "average"):
        lab = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X)
        out["linkages"][link] = metrics(X, lab, None, seed)
        out["linkages"][link]["cluster_sizes"] = np.bincount(lab).tolist()
    sub = X[np.linspace(0, len(X) - 1, min(dendro_points, len(X))).astype(int)]
    Zl = linkage(sub, method="ward")
    dd = dendrogram(Zl, no_plot=True, truncate_mode="lastp", p=40)
    out["dendrogram"] = {"icoord": dd["icoord"], "dcoord": dd["dcoord"], "leaves": [str(x) for x in dd["ivl"]],
                         "linkage": "ward", "n_points": len(sub)}
    out["best_linkage_by_silhouette"] = max(out["linkages"], key=lambda l: out["linkages"][l]["silhouette"] or -1)
    return out





