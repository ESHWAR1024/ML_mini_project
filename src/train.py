"""Train and evaluate Linear Regression, SVR and a Neural Network (MLP) to predict
daily ozone from meteorology + precursors.  Also trains a Random Forest as an extra baseline.

Usage:  python src/train.py --data data/ozone_daily.csv --outdir outputs
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

FEATURES = ["temperature", "temp_p90", "pressure", "wind", "no2", "voc",
            "month_sin", "month_cos", "latitude", "longitude"]   # absent columns are dropped automatically
TARGET = "ozone"


def load(path):
    df = pd.read_csv(path)
    req = ["ozone", "temperature", "temp_p90", "pressure", "wind", "no2", "voc", "latitude", "longitude"]
    df = df.dropna(subset=[c for c in req if c in df.columns])   # 'voc' may be absent (--no-voc)
    df = df[df.ozone >= 0]
    if "date_local" in df.columns:                  # DAILY data: add cyclic month features
        df["date_local"] = pd.to_datetime(df["date_local"])
        m = df.date_local.dt.month
        df["month_sin"] = np.sin(2 * np.pi * m / 12)   # month is cyclic (Dec is next to Jan)
        df["month_cos"] = np.cos(2 * np.pi * m / 12)
        df["year"] = df.date_local.dt.year
    elif "year" not in df.columns:                  # ANNUAL data already has a 'year' column
        raise SystemExit("CSV needs a 'date_local' or a 'year' column.")
    return df


def split(df, val_year, test_year):
    """Temporal split: past -> train, then validation year, then test year (no leakage)."""
    tr = df[df.year < val_year]
    va = df[df.year == val_year]
    te = df[df.year >= test_year]
    return tr, va, te


def metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))


def scatter(y, p, title, path):
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y, p, s=3, alpha=0.3)
    lim = [0, max(y.max(), p.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Actual ozone (ppm)")
    plt.ylabel("Predicted ozone (ppm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(a):
    global FEATURES
    os.makedirs(a.outdir, exist_ok=True)
    df = load(a.data)
    FEATURES = [f for f in FEATURES if f in df.columns]     # drops 'voc' if the CSV has no VOC column
    print("features used:", FEATURES)
    tr, va, te = split(df, a.val_year, a.test_year)
    print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("A split is empty - adjust --val-year/--test-year to years present in the data.")

    scaler = MinMaxScaler().fit(tr[FEATURES])           # fit on train only
    X = {k: scaler.transform(d[FEATURES]) for k, d in dict(tr=tr, va=va, te=te).items()}
    y = {k: d[TARGET].values for k, d in dict(tr=tr, va=va, te=te).items()}

    results, preds = {}, {}

    # 1) Linear regression -------------------------------------------------
    lr = LinearRegression().fit(X["tr"], y["tr"])
    models = {"Linear Regression": lr}

    # 2) SVR (RBF) - tuned on a subsample because SVR is O(n^2) -------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr), size=min(a.svr_samples, len(tr)), replace=False)
    ystd = y["tr"].std()
    search = RandomizedSearchCV(
        SVR(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 0.5, 1.0],
         "epsilon": [ystd * f for f in (0.01, 0.05, 0.1, 0.2)]},   # epsilon tied to target scale!
        n_iter=a.svr_iter, cv=3, scoring="r2", random_state=0, n_jobs=-1)
    search.fit(X["tr"][idx], y["tr"][idx])
    print("SVR best params:", search.best_params_)
    models["SVR (RBF)"] = search.best_estimator_

    # 3) Neural network ------------------------------------------------------
    # solver defaults to "adam", which adapts step sizes itself (sklearn only uses the
    # learning_rate="adaptive" schedule with solver="sgd"), so we only set the initial rate.
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="relu",
                       learning_rate_init=1e-3,
                       early_stopping=True, validation_fraction=0.1, max_iter=200,
                       random_state=0)
    mlp.fit(X["tr"], y["tr"])
    models["Neural Network (MLP)"] = mlp

    # 4) Extra baseline: Random Forest ---------------------------------------
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
    rf.fit(X["tr"], y["tr"])
    models["Random Forest"] = rf

    for name, m in models.items():
        t0 = time.time()
        results[name] = {k: metrics(y[k], m.predict(X[k])) for k in ("tr", "va", "te")}
        preds[name] = m.predict(X["te"])
        print(f"{name:22s} R2 train/val/test = "
              f"{results[name]['tr']['r2']:.3f} / {results[name]['va']['r2']:.3f} / {results[name]['te']['r2']:.3f}")
        scatter(y["te"], preds[name], f"{name} (test)",
                os.path.join(a.outdir, f"scatter_{name.split()[0].lower()}.png"))

    # Feature importance for best model (by validation R2) --------------------
    best = max(results, key=lambda n: results[n]["va"]["r2"])
    pi = permutation_importance(models[best], X["va"], y["va"], n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(pi.importances_mean)
    plt.figure(figsize=(5.5, 4))
    plt.barh(np.array(FEATURES)[order], pi.importances_mean[order])
    plt.xlabel("Drop in R2 when feature is shuffled")
    plt.title(f"Permutation importance - {best}")
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "feature_importance.png"), dpi=150)
    plt.close()

    # Save outputs ------------------------------------------------------------
    summary = dict(n_train=len(tr), n_val=len(va), n_test=len(te), best_model=best,
                   svr_params={k: float(v) if not isinstance(v, str) else v
                               for k, v in search.best_params_.items()},
                   results=results,
                   feature_importance=dict(zip(FEATURES, map(float, pi.importances_mean))))
    with open(os.path.join(a.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump(dict(model=models[best], scaler=scaler, features=FEATURES, name=best),
                os.path.join(a.outdir, "best_model.joblib"))

    rows = ["| Model | R2 train | R2 val | R2 test | RMSE test | MAE test |", "|---|---|---|---|---|---|"]
    for n, r in results.items():
        rows.append(f"| {n} | {r['tr']['r2']:.3f} | {r['va']['r2']:.3f} | {r['te']['r2']:.3f} | "
                    f"{r['te']['rmse']:.4f} | {r['te']['mae']:.4f} |")
    open(os.path.join(a.outdir, "results_table.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"Best model by validation R2: {best}. Outputs in {a.outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ozone_daily.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--svr-samples", type=int, default=15000)
    ap.add_argument("--svr-iter", type=int, default=8)
    main(ap.parse_args())