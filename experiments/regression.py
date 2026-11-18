"""All requested regression models with time-aware tuning and leakage-safe pipelines."""
from __future__ import annotations
import time, warnings
import numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .common import chronological_split, regression_metrics

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:   # pragma: no cover
    HAVE_XGB = False

FAMILY = {"simple_linear": "linear", "multiple_linear": "linear", "polynomial": "linear", "ridge": "linear",
          "lasso": "linear", "elastic_net": "linear", "bayesian_ridge": "linear", "svr": "nonlinear",
          "decision_tree": "nonlinear", "random_forest": "nonlinear", "gradient_boosting": "nonlinear",
          "xgboost": "nonlinear", "knn": "nonlinear"}


def model_zoo(seed: int = 42) -> dict:
    """name -> (estimator, grid). Grid keys refer to the estimator step named 'model'."""
    z = {
        "simple_linear": (Pipeline([("sel", SelectKBest(f_regression, k=1)), ("model", LinearRegression())]), {}),
        "multiple_linear": (LinearRegression(), {}),
        "polynomial": (Pipeline([("poly", PolynomialFeatures(2, include_bias=False)), ("model", Ridge())]),
                       {"alpha": [1.0, 10.0, 100.0]}),
        "ridge": (Ridge(), {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "lasso": (Lasso(max_iter=20000), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
        "elastic_net": (ElasticNet(max_iter=20000), {"alpha": [1e-3, 1e-2, 1e-1], "l1_ratio": [0.2, 0.5, 0.8]}),
        "bayesian_ridge": (BayesianRidge(), {}),
        "svr": (SVR(kernel="rbf"), {"C": [1.0, 10.0], "epsilon": [0.01, 0.1]}),
        "decision_tree": (DecisionTreeRegressor(random_state=seed), {"max_depth": [3, 6, 10]}),
        "random_forest": (RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=-1),
                          {"max_depth": [6, 12]}),
        "gradient_boosting": (GradientBoostingRegressor(random_state=seed),
                              {"n_estimators": [150], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "knn": (KNeighborsRegressor(), {"n_neighbors": [5, 15, 30]}),
    }
    if HAVE_XGB:
        z["xgboost"] = (XGBRegressor(random_state=seed, n_jobs=2, verbosity=0),
                        {"n_estimators": [200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]})
    return z


def _wrap(est, grid):
    """Scaler (+ target scaler) inside the pipeline => fitted per CV fold only (no preprocessing leakage)."""
    if isinstance(est, Pipeline):
        steps = [("scaler", StandardScaler())] + est.steps
        # route grid to final 'model' step
        inner = Pipeline(steps)
    else:
        inner = Pipeline([("scaler", StandardScaler()), ("model", est)])
    ttr = TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
    return ttr, {f"regressor__model__{k}": v for k, v in grid.items()}


def run_regression(df: pd.DataFrame, features: list[str], target: str, time_col: str = "epoch", seed: int = 42,
                   models: list[str] | None = None, n_splits: int = 3, max_rows_svr: int = 4000,
                   max_rows: int | None = 12000) -> dict:
    warnings.filterwarnings("ignore")
    d = df.dropna(subset=features + [target]).copy()
    if len(d) < 60:
        return {"error": f"insufficient rows ({len(d)}) for a chronological split", "n_rows": len(d)}
    if max_rows and len(d) > max_rows:       # evenly thinned, order preserved => still chronological
        d = d.sort_values(time_col).iloc[np.linspace(0, len(d) - 1, max_rows).astype(int)]
    tr, va, te = chronological_split(d, time_col, (0.7, 0.1, 0.2))
    trva = pd.concat([tr, va])
    X_trva, y_trva = trva[features].to_numpy(), trva[target].to_numpy()
    X_te, y_te = te[features].to_numpy(), te[target].to_numpy()
    cv = TimeSeriesSplit(n_splits=n_splits)

    out = {"target": target, "features": features, "seed": seed, "n_train": len(trva), "n_test": len(te),
           "train_epoch_max": str(tr[time_col].max()), "test_epoch_min": str(te[time_col].min()),
           "split": "chronological 70/10/20 (tuning on first 80% with TimeSeriesSplit, test on final 20%)"}
    base = DummyRegressor().fit(X_trva, y_trva)
    out["baseline_mean"] = regression_metrics(y_te, base.predict(X_te))
    results = []
    for name, (est, grid) in model_zoo(seed).items():
        if models and name not in models:
            continue
        t0 = time.time()
        Xf, yf = X_trva, y_trva
        if name == "svr" and len(Xf) > max_rows_svr:
            idx = np.linspace(0, len(Xf) - 1, max_rows_svr).astype(int)
            Xf, yf = Xf[idx], yf[idx]
        model, pg = _wrap(est, grid)
        try:
            if pg:
                gs = GridSearchCV(model, pg, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, refit=True)
                gs.fit(Xf, yf)
                best, cv_mae, params = gs.best_estimator_, -gs.best_score_, {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
            else:
                best = model.fit(Xf, yf)
                cv_mae = float(np.mean([regression_metrics(yf[v], model.fit(Xf[t], yf[t]).predict(Xf[v]))["mae"]
                                        for t, v in cv.split(Xf)]))
                best = model.fit(Xf, yf)
                params = {}
            p = best.predict(X_te)
            m = regression_metrics(y_te, p)
            sel = np.linspace(0, len(y_te) - 1, min(400, len(y_te))).astype(int)
            results.append({"model": name, "family": FAMILY[name], "best_params": params, "cv_mae": float(cv_mae),
                            "test": m, "fit_seconds": round(time.time() - t0, 2),
                            "actual_vs_pred": {"y_true": y_te[sel].tolist(), "y_pred": p[sel].tolist(),
                                               "epoch": [str(x) for x in te[time_col].to_numpy()[sel]]},
                            "residuals": (y_te[sel] - p[sel]).tolist()})
        except Exception as exc:
            results.append({"model": name, "family": FAMILY[name], "error": str(exc)})
    results.sort(key=lambda r: r.get("test", {}).get("rmse", float("inf")))
    out["models"] = results
    out["xgboost_available"] = HAVE_XGB
    return out
