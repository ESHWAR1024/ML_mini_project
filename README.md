# Predicting Daily Ground-Level Ozone from Weather and Precursor Pollutants

UE24CS352A Machine Learning mini-project. We predict the **daily mean ozone (ppm) at a US EPA air-quality
monitoring station** from weather, NO2, season, location and recent ozone history, using EPA AirData.
Five models are compared (Linear Regression, SVR, Neural Network, Random Forest, Gradient Boosting), and the
neural network is specifically improved beyond the assigned reference paper (Sutton, CS229 2019).

## Results (test year 2019, never seen during training or model selection)

| Model | R2 train | R2 val | R2 test | RMSE test (ppm) | MAE test (ppm) |
|---|---|---|---|---|---|
| Linear Regression | 0.685 | 0.713 | 0.653 | 0.0065 | 0.0050 |
| SVR (RBF) | 0.708 | 0.689 | 0.639 | 0.0066 | 0.0051 |
| **Neural Network (MLP)** | **0.782** | **0.777** | **0.737** | **0.0057** | **0.0044** |
| Random Forest | 0.893 | 0.758 | 0.713 | 0.0059 | 0.0046 |
| Gradient Boosting | 0.767 | 0.765 | 0.723 | 0.0058 | 0.0045 |

| Neural network, test R2 | Value |
|---|---|
| Reference paper (Sutton, CS229 2019) | 0.654 |
| **Our final model** | **0.737** |

Our final model also uses yesterday's ozone (one-day-ahead forecasting), and our dataset/test period differ
from the paper's, so this comparison is indicative, not a strict like-for-like. Run with `--no-lag` for the
closer same-day-only comparison.

## What we improved over the paper
1. **Target standardisation** inside the neural network — ozone is only ~0.03 ppm, which starves gradient
   training if left raw.
2. **Automated tuning**: random search over NN hyperparameters, scored on the 2018 validation year only
   (never the test year). SVR tuned via 3-fold `RandomizedSearchCV`, with epsilon tied to the target's own
   spread — the paper's SVR failed (R2 = -2.36) because its epsilon was far too large.
3. **3-network ensemble**, averaged, instead of one hand-tuned network.
4. **19 engineered features** vs. the paper's 7: per-site average ozone, day-of-year seasonality, daily peak
   values (temperature/wind/NO2), and lag features (yesterday's ozone, 7-day rolling ozone, etc.).
5. **Two extra models** (Random Forest, Gradient Boosting) added as stronger baselines, plus permutation
   feature importance.

Full explanation, worked examples and viva prep: **docs/UNDERSTAND_THE_PROJECT.md**.

## Project structure
```
ML_mini_project/
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- UNDERSTAND_THE_PROJECT.md     how everything works, paper comparison, viva prep
|   |-- REPORT_SUMMARY.md             same content as the generated PDF write-up
|-- src/
|   |-- data_epa_csv.py               EPA daily files -> data/ozone_daily.csv
|   |-- train.py                      features, split, tuning, 5 models, evaluation, plots
|   |-- demo.py                       predict ozone for one day you type in
|   |-- make_report.py                metrics.json -> 2-page PDF write-up
|-- data/
|   |-- raw/                          downloaded EPA zip files (you put them here)
|   |-- ozone_daily.csv               created by data_epa_csv.py
|-- outputs_final/                    created by train.py (table, plots, metrics.json, best_model.joblib)
```

## 1. Setup

Requires Python 3.9+.

**Windows (PowerShell)**
```powershell
cd ML_mini_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

**macOS / Linux**
```bash
cd ML_mini_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the data (EPA AirData, free, no account)
1. Open https://aqs.epa.gov/aqsweb/airdata/download_files.html and go to **Daily Summary Data**.
2. Download one zip per year (we used **2010-2019**) for each:

| File name starts with | Content | Required |
|---|---|---|
| `daily_44201_` | Ozone (the target) | yes |
| `daily_42602_` | NO2 | yes |
| `daily_TEMP_` | Temperature | yes |
| `daily_PRESS_` | Barometric pressure | yes |
| `daily_WIND_` | Wind | yes |
| `daily_VOCS_` | VOCs (few stations) | no, skipped with `--no-voc` |
| `daily_RH_DP_` | Humidity | no, only with `--with-rh` |

3. Put all zips in **one folder**: `data/raw/` (leave them zipped).

## 3. Run the pipeline

**Step 1 — build the dataset** (keeps daily mean *and* daily peak values per parameter):
```bash
python src/data_epa_csv.py --raw data/raw --out data/ozone_daily.csv --no-voc
```
Optional: add `--with-rh` for humidity (fewer stations, fewer rows).

**Step 2 — train and evaluate**

Fast check (a few minutes — Linear + 1 network only, no tuning):
```bash
python src/train.py --data data/ozone_daily.csv --outdir outputs_fast --nn-only --seeds 1
```
Best / final run (about 20 minutes — tunes the network, trains all 5 models):
```bash
python src/train.py --data data/ozone_daily.csv --outdir outputs_final --tune-nn --tune-iter 8 --seeds 3
```
Paper-style comparison (drops yesterday's-ozone/lag features):
```bash
python src/train.py --data data/ozone_daily.csv --outdir outputs_nolag --no-lag --tune-nn --tune-iter 8 --seeds 3
```
Progress prints with timestamps at every stage (data loading, SVR search, NN tuning, NN ensemble, RF, GB,
permutation importance), so it's clear the run hasn't frozen.

**Step 3 — live demo**
```bash
python src/demo.py --model outputs_final/best_model.joblib --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --month 7 --latitude 34.05 --longitude -118.24 --ozone-yesterday 0.055
```
Units: temperature deg F, pressure millibar, wind knots, NO2 ppb, ozone in ppm (0.055 ppm = 55 ppb).
Prints: the inputs you gave, every other feature the model needs (filled in automatically, labelled with
where each value came from), then the target and the prediction. Anything you don't pass falls back to a
training-set median, the nearest training station's average, or a same-as-today/same-as-yesterday assumption
— see the printed output for exactly which.

**Step 4 — PDF write-up (2 pages)**
```bash
python src/make_report.py --data data/ozone_daily.csv --team "Name1 (SRN1), Name2 (SRN2)" --problem "17" --out report.pdf
```
Reads straight from `outputs_final/metrics.json` (auto-detected if you don't pass `--metrics`); re-run any
time after retraining so the numbers stay in sync.

## 4. Command options (train.py)
| Option | Meaning | Default |
|---|---|---|
| `--data` | input CSV | `data/ozone_daily.csv` |
| `--outdir` | where outputs are saved | `outputs` |
| `--val-year`, `--test-year` | temporal split | 2018, 2019 |
| `--tune-nn` | random-search the NN settings on the validation year | off |
| `--tune-iter N` | how many NN settings to try | 12 |
| `--seeds N` | networks averaged in the NN ensemble | 5 |
| `--nn-only` | skip SVR / Random Forest / Gradient Boosting (much faster) | off |
| `--no-lag` | drop yesterday's-ozone/lag features (closest to the paper's setup) | off |
| `--with-rh` | also build/use humidity (fewer stations) | off |
| `--svr-samples`, `--svr-iter` | SVR tuning subsample size / number of trials | 15000, 8 |

## 5. What you get in the output folder
| File | What it is |
|---|---|
| `results_table.md` | R2 / RMSE / MAE for every model (compare the test column) |
| `scatter_*.png` | predicted vs actual ozone on the test year, per model |
| `feature_importance.png` | which features matter most (permutation importance) |
| `metrics.json` | all numbers, feature list, NN/SVR settings (read by make_report.py) |
| `best_model.joblib` | best model + scaler + feature list + defaults (read by demo.py) |

## 6. The 19 features
**8 from the paper's design** (temperature, pressure, wind, NO2, day-of-year sin/cos in place of month,
latitude, longitude — VOC dropped, too few stations measure it) **+ 11 engineered**: daily peak temperature/
wind/NO2, peak-minus-mean temperature spread, per-site average ozone (training years only), and lag features
(yesterday's ozone, yesterday's peak ozone, 7-day rolling ozone, yesterday's temperature, 3-day rolling
temperature, yesterday's NO2).

## 7. How it works in 30 seconds
`data_epa_csv.py` joins the EPA files into one row per station per day. `train.py` builds features (including
lags), splits **by year** (train 2010-2017, validation 2018, test 2019) to avoid leakage, scales with
z-score statistics from training data only, tunes and trains five models, and picks the best on the
validation year — the test year is only the final exam. `demo.py` and `make_report.py` reuse the saved
results.

## 8. Known limitations
- Final model uses yesterday's ozone, so it's a one-day-ahead forecast (use `--no-lag` for a same-day-only
  setup closer to the paper).
- Missing drivers: sunlight, humidity (unless `--with-rh`), traffic and industry, VOCs.
- Target is the daily mean, not the regulatory 8-hour maximum.
- Only stations measuring all required parameters are included (143 sites).
- All evaluation stations were also seen during training — results reflect forecasting at known stations,
  not generalising to an unmonitored one.
- Different dataset and test period from the paper, so the comparison is indicative.

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| `No files found for 'ozone'...` | The zips are not in `--raw`, or names don't start with `daily_44201_` etc. |
| `A split is empty` | Your data lacks the validation/test year; change `--val-year` / `--test-year`. |
| `ModuleNotFoundError: reportlab` (or another) | `pip install -r requirements.txt` with the venv active. |
| Only 14 features used, no peak-value features | The CSV was built with an older script version; re-run Step 1. |
| `make_report.py: FileNotFoundError` for metrics.json | Run without `--metrics` — it auto-detects any `outputs*/metrics.json`; check you're in the project root and that `train.py` actually finished. |
| `demo.py: Cannot predict: no value for [...]` | Pass the named option, or re-train with the current `train.py` so defaults are stored in the model file. |
| Training feels frozen | The NN-ensemble line prints nothing until it finishes training in parallel; check CPU usage, not the terminal. |
| Too slow | Use `--nn-only --seeds 1`, or lower `--tune-iter` and `--svr-samples`. |

## References
- M. Sutton, *Tropospheric ozone prediction*, CS229 Final Project Report, 2019 (assigned problem).
- US EPA, AirData daily summary files, https://aqs.epa.gov/aqsweb/airdata
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.

## Team
Name 1 (SRN) | Name 2 (SRN) | Problem no. __
