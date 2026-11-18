"""Feature selection: correlation, variance threshold, SelectKBest, mutual information, RFE,
random-forest importance, L1. Selectors are fit on the TRAIN segment only (chronological)."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, RFE)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler

from .common import chronological_split


def run_feature_selection(df: pd.DataFrame, features: list[str], target: str, time_col="epoch", seed=42,
                          k: int = 5) -> dict:
    d = df.dropna(subset=features + [target])
    tr, _, _ = chronological_split(d, time_col, (0.7, 0.1, 0.2))
    if len(tr) > 6000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 6000).astype(int)]
    X = StandardScaler().fit_transform(tr[features].to_numpy()); y = tr[target].to_numpy()
    k = min(k, len(features))
    corr = pd.DataFrame(X, columns=features).corr().round(4)
    ytc = [float(np.corrcoef(X[:, i], y)[0, 1]) if X[:, i].std() > 0 else 0.0 for i in range(len(features))]
    vt = VarianceThreshold(threshold=0.0).fit(tr[features].to_numpy())
    kb = SelectKBest(f_regression, k=k).fit(X, y)
    mi = mutual_info_regression(X, y, random_state=seed)
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X, y)
    rf = RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=-1, max_depth=10).fit(X, y)
    lasso = LassoCV(cv=3, random_state=seed, max_iter=20000).fit(X, y)
    table = []
    for i, f in enumerate(features):
        table.append({"feature": f, "corr_with_target": ytc[i], "variance_kept": bool(vt.get_support()[i]),
                      "f_score": float(np.nan_to_num(kb.scores_[i])), "selectkbest_selected": bool(kb.get_support()[i]),
                      "mutual_info": float(mi[i]), "rfe_rank": int(rfe.ranking_[i]), "rfe_selected": bool(rfe.support_[i]),
                      "rf_importance": float(rf.feature_importances_[i]), "lasso_coef": float(lasso.coef_[i]),
                      "lasso_selected": bool(abs(lasso.coef_[i]) > 1e-8)})
    # consensus: number of methods that select the feature
    for r in table:
        r["votes"] = int(r["selectkbest_selected"]) + int(r["rfe_selected"]) + int(r["lasso_selected"]) + \
            int(r["rf_importance"] >= np.sort(rf.feature_importances_)[-k]) + \
            int(r["mutual_info"] >= np.sort(mi)[-k])
    return {"target": target, "k": k, "n_fit_rows": len(tr), "lasso_alpha": float(lasso.alpha_),
            "correlation_matrix": {"labels": features, "values": corr.to_numpy().tolist()},
            "table": sorted(table, key=lambda r: -r["votes"]),
            "fit_on": "chronological training segment only"}
