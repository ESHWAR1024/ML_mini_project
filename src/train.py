"""Train and evaluate Linear Regression, SVR, a (much improved) Neural Network ensemble, Random Forest
and Gradient Boosting to predict daily ozone from meteorology + precursors (+ yesterday's ozone).

Usage:
    python src/train.py --data data/ozone_daily.csv --outdir outputs
    python src/train.py --no-lag            # paper-style: today's weather/NO2 only, no yesterday-ozone
    python src/train.py --seeds 1           # faster NN (1 network instead of an ensemble)

What changed vs the first version (see docs/PROJECT_GUIDE.md for the why):
  1. Features: peak temperature, temp spread, peak wind/NO2, day-of-year (instead of month),
     per-site average ozone (learned from TRAIN only), and lag features (yesterday's ozone/temp/NO2).
  2. Neural network: target is standardised (ozone ~0.03 ppm is tiny -> bad for NN training),
     wider net, L2 regularisation, mini-batches, and an ensemble of several seeds averaged.
  3. Inputs scaled with StandardScaler (z-scores) instead of Min-Max (less sensitive to outliers).
  4. Extra model: HistGradientBoosting.  Random Forest regularised to overfit less.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGET = "ozone"
BASE = ["temperature", "temp_max", "temp_spread", "pressure", "wind", "wind_max", "no2", "no2_max",
        "voc", "rh", "doy_sin", "doy_cos", "latitude", "longitude", "site_mean_ozone"]
LAG = ["ozone_lag1", "ozone_max_lag1", "ozone_roll7", "temp_lag1", "temp_roll3", "no2_lag1"]
# absent columns are dropped automatically (e.g. voc / rh if the CSV does not have them)


# ----------------------------------------------------------------------------- data
def add_features(df):
    df = df.copy()
    df["date_local"] = pd.to_datetime(df["date_local"])
    df["year"] = df.date_local.dt.year
    doy = df.date_local.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)     # day of year is cyclic (31 Dec next to 1 Jan)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["site_id"] = (df.state_code.astype(str) + "-" + df.county_code.astype(str) + "-"
                     + df.site_num.astype(str))
    if "temp_max" in df.columns:
        df["temp_spread"] = df.temp_max - df.temperature
    df = df.sort_values(["site_id", "date_local"]).reset_index(drop=True)
    g = df.groupby("site_id")

    def lag(col, k):
        """Value of `col` k days earlier at the SAME site; NaN if that exact day is missing."""
        s = g[col].shift(k)
        ok = g["date_local"].shift(k) == df["date_local"] - pd.Timedelta(days=k)
        return s.where(ok)

    df["ozone_lag1"] = lag("ozone", 1)
    if "ozone_max" in df.columns:
        df["ozone_max_lag1"] = lag("ozone_max", 1)
    df["temp_lag1"] = lag("temperature", 1)
    df["temp_roll3"] = pd.concat([df.temperature, df.temp_lag1, lag("temperature", 2)],
                                 axis=1).mean(axis=1, skipna=False)
    df["no2_lag1"] = lag("no2", 1)
    # mean ozone over the previous 7 days (needs at least 3 of them to exist) = recent "ozone climate"
    past = pd.concat([lag("ozone", k) for k in range(1, 8)], axis=1)
    df["ozone_roll7"] = past.mean(axis=1).where(past.notna().sum(axis=1) >= 3)
    return df


def load(path, use_lag):
    df = pd.read_csv(path)
    if "date_local" not in df.columns:
        raise SystemExit("CSV needs a 'date_local' column (build it with data_epa_csv.py).")
    df = df[df.ozone >= 0]
    df = add_features(df)
    feats = [f for f in BASE + (LAG if use_lag else []) if f in df.columns or f == "site_mean_ozone"]
    need = [f for f in feats if f != "site_mean_ozone"] + [TARGET, "latitude", "longitude"]
    df = df.dropna(subset=need)
    return df, feats


def split(df, val_year, test_year):
    """Temporal split: past -> train, then validation year, then test year (no leakage)."""
    return df[df.year < val_year].copy(), df[df.year == val_year].copy(), df[df.year >= test_year].copy()


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


NN_DEFAULT = dict(hidden=(256, 128, 64), alpha=1e-3, lr=1e-3, batch=512)


def tune_nn(Xtr, ytr, Xva, yva, n_iter, seed=0):
    """Random search over NN settings. Each candidate is trained on a subsample of TRAIN and
    scored on the VALIDATION year (so tuning respects time order and never touches test)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(40000, len(Xtr)), replace=False)
    grid = dict(hidden=[(128, 64), (256, 128, 64), (256, 256, 128), (512, 256, 128), (128, 128, 64, 32)],
                alpha=[1e-5, 1e-4, 1e-3, 1e-2, 3e-2], lr=[3e-4, 1e-3, 3e-3], batch=[256, 512, 1024])
    best, best_r2 = dict(NN_DEFAULT), -9
    for i in range(n_iter):
        cfg = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        m = make_mlp(0, **cfg).fit(Xtr[idx], ytr[idx])
        r2 = r2_score(yva, m.predict(Xva))
        print(f"  [{(time.time() - T0) / 60:5.1f} min] tune {i+1}/{n_iter}: val R2={r2:.4f}  {cfg}")
        if r2 > best_r2:
            best, best_r2 = cfg, r2
    print(f"best NN settings: {best} (val R2 {best_r2:.4f})")
    return best


def make_mlp(seed, hidden=(256, 128, 64), alpha=1e-3, lr=1e-3, batch=512):
    # TransformedTargetRegressor standardises y (mean 0, std 1) for training and converts predictions
    # back to ppm automatically.  Without this, ozone ~0.03 gives microscopic gradients.
    return TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                               alpha=alpha,                # L2 penalty -> less overfitting
                               batch_size=batch, learning_rate_init=lr,
                               early_stopping=True, validation_fraction=0.1,
                               n_iter_no_change=15, max_iter=300, random_state=seed),
        transformer=StandardScaler())


# ----------------------------------------------------------------------------- main
T0 = time.time()


def say(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    df, feats = load(a.data, use_lag=not a.no_lag)
    tr, va, te = split(df, a.val_year, a.test_year)
    print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("A split is empty - adjust --val-year/--test-year to years present in the data.")

    # Per-site average ozone, learned from TRAIN rows only (target encoding). Unseen site -> global mean.
    site_mean = tr.groupby("site_id")[TARGET].mean()
    glob_mean = float(tr[TARGET].mean())
    for d in (tr, va, te):
        d["site_mean_ozone"] = d.site_id.map(site_mean).fillna(glob_mean)
    print(f"features used ({len(feats)}):", feats)

    scaler = StandardScaler().fit(tr[feats])             # fit on train only
    X = {k: scaler.transform(d[feats]) for k, d in dict(tr=tr, va=va, te=te).items()}
    y = {k: d[TARGET].values for k, d in dict(tr=tr, va=va, te=te).items()}
    models = {}

    say("stage 1/5: Linear Regression")
    # 1) Linear regression -------------------------------------------------
    models["Linear Regression"] = LinearRegression().fit(X["tr"], y["tr"])

    search = None
    # 2) SVR (RBF) - tuned on a subsample because SVR is O(n^2) -------------
    if not a.nn_only:
        say("stage 2/5: SVR search (slow)")
        rng = np.random.default_rng(0)
        idx = rng.choice(len(tr), size=min(a.svr_samples, len(tr)), replace=False)
        ystd = y["tr"].std()
        search = RandomizedSearchCV(
            SVR(kernel="rbf"),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 0.5, 1.0],
             "epsilon": [ystd * f for f in (0.01, 0.05, 0.1, 0.2)]},
            n_iter=a.svr_iter, cv=3, scoring="r2", random_state=0, n_jobs=-1, verbose=2)
        search.fit(X["tr"][idx], y["tr"][idx])
        print("SVR best params:", search.best_params_)
        models["SVR (RBF)"] = search.best_estimator_

    # 3) Neural network: ensemble of `seeds` MLPs, predictions averaged ------
    say("stage 3/5: Neural network" + (" - tuning first" if a.tune_nn else ""))
    cfg = tune_nn(X["tr"], y["tr"], X["va"], y["va"], a.tune_iter) if a.tune_nn else dict(NN_DEFAULT)
    say(f"training neural network ensemble ({a.seeds} net(s)) on all rows - no output until done")
    mlp = VotingRegressor([(f"mlp{s}", make_mlp(s, **cfg)) for s in range(a.seeds)], n_jobs=-1, verbose=True)
    mlp.fit(X["tr"], y["tr"])
    models["Neural Network (MLP)"] = mlp

    if not a.nn_only:
        say("stage 4/5: Random Forest")
        # 4) Random Forest (a bit more regularised than before) ----------------
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.6, n_jobs=-1, random_state=0, verbose=1).fit(X["tr"], y["tr"])
        say("stage 5/5: Gradient Boosting")
        # 5) Gradient boosting -------------------------------------------------
        models["Gradient Boosting"] = HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=0).fit(X["tr"], y["tr"])

    say("scoring all models + drawing plots")
    results, preds = {}, {}
    for name, m in models.items():
        results[name] = {k: metrics(y[k], m.predict(X[k])) for k in ("tr", "va", "te")}
        preds[name] = m.predict(X["te"])
        r = results[name]
        print(f"{name:22s} R2 train/val/test = {r['tr']['r2']:.3f} / {r['va']['r2']:.3f} / {r['te']['r2']:.3f}")
        scatter(y["te"], preds[name], f"{name} (test)",
                os.path.join(a.outdir, f"scatter_{name.split()[0].lower()}.png"))

    # Feature importance for best model (by validation R2) --------------------
    best = max(results, key=lambda n: results[n]["va"]["r2"])
    say(f"permutation importance for {best} (last step)")
    pi = permutation_importance(models[best], X["va"], y["va"], n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(pi.importances_mean)
    plt.figure(figsize=(6, 5))
    plt.barh(np.array(feats)[order], pi.importances_mean[order])
    plt.xlabel("Drop in R2 when feature is shuffled")
    plt.title(f"Permutation importance - {best}")
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "feature_importance.png"), dpi=150)
    plt.close()

    # Save outputs ------------------------------------------------------------
    summary = dict(n_train=len(tr), n_val=len(va), n_test=len(te), best_model=best, features=feats,
                   use_lag=not a.no_lag,
                   svr_params=({k: float(v) if not isinstance(v, str) else v
                                for k, v in search.best_params_.items()} if search else None),
                   nn_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                   results=results,
                   feature_importance=dict(zip(feats, map(float, pi.importances_mean))))
    with open(os.path.join(a.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    site_table = (tr.groupby("site_id").agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
                  .join(site_mean.rename("site_mean_ozone")).reset_index(drop=True))
    defaults = {f: float(tr[f].median()) for f in feats if f != "site_mean_ozone"}
    joblib.dump(dict(model=models[best], scaler=scaler, features=feats, name=best,
                     defaults=defaults, site_table=site_table, global_mean=glob_mean),
                os.path.join(a.outdir, "best_model.joblib"))

    rows = ["| Model | R2 train | R2 val | R2 test | RMSE test | MAE test |", "|---|---|---|---|---|---|"]
    for n, r in results.items():
        rows.append(f"| {n} | {r['tr']['r2']:.3f} | {r['va']['r2']:.3f} | {r['te']['r2']:.3f} | "
                    f"{r['te']['rmse']:.4f} | {r['te']['mae']:.4f} |")
    with open(os.path.join(a.outdir, "results_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    say("DONE")
    print(f"Best model by validation R2: {best}. Outputs in {a.outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ozone_daily.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--svr-samples", type=int, default=15000)
    ap.add_argument("--svr-iter", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5, help="number of MLPs averaged in the NN ensemble")
    ap.add_argument("--tune-nn", action="store_true", help="random-search NN hyperparameters on the validation year")
    ap.add_argument("--tune-iter", type=int, default=12, help="number of NN settings to try with --tune-nn")
    ap.add_argument("--nn-only", action="store_true", help="skip SVR/RF/GB, train only Linear + NN (fast)")
    ap.add_argument("--no-lag", action="store_true",
                    help="drop yesterday's-ozone/lag features (closest to the original paper setup)")
    main(ap.parse_args())